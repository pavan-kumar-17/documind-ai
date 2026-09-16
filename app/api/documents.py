import os
import uuid
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.domain import User, Document, DocumentChunk
from app.schemas.pydantic_schemas import DocumentResponse, DocumentDetailResponse
from app.api.deps import get_current_user
from app.services.document_service import DocumentService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger("documind.documents_api")

def process_and_index_document(doc_id: str, db_session_factory):
    db: Session = db_session_factory()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return

        doc.status = "Processing"
        db.commit()

        # Step 1: Text Extraction
        extracted_doc = DocumentService.extract_text(doc.file_path, doc.filename, doc.file_size)
        doc.page_count = extracted_doc.total_pages

        # Step 2: Sentence-Aware Chunking
        chunker = ChunkingService()
        raw_chunks = chunker.chunk_extracted_document(extracted_doc)

        if not raw_chunks:
            doc.status = "Failed"
            doc.error_message = "No readable text content extracted from document."
            db.commit()
            return

        # Step 3: Remove old chunks if re-indexing
        existing_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
        if existing_chunks:
            vector_service.delete_document_chunks(doc.id)
            for c in existing_chunks:
                db.delete(c)
            db.commit()

        # Step 4: Batch Embeddings Generation
        chunk_texts = [c.text for c in raw_chunks]
        embeddings = embedding_service.generate_embeddings_batch(chunk_texts)

        # Step 5: Add to Vector DB
        vector_metas = []
        db_chunk_objects = []

        for idx, (raw_c, vec) in enumerate(zip(raw_chunks, embeddings)):
            chunk_uuid = str(uuid.uuid4())
            db_chunk = DocumentChunk(
                id=chunk_uuid,
                document_id=doc.id,
                page_number=raw_c.page_number,
                chunk_index=raw_c.chunk_index,
                text=raw_c.text,
                char_count=raw_c.char_count,
                token_count=raw_c.token_count
            )
            db_chunk_objects.append(db_chunk)

            vector_metas.append({
                "chunk_id": chunk_uuid,
                "document_id": doc.id,
                "document_name": doc.filename,
                "page_number": raw_c.page_number,
                "chunk_index": raw_c.chunk_index,
                "text": raw_c.text
            })

        db.add_all(db_chunk_objects)
        db.commit()

        vector_ids = vector_service.add_chunks(vector_metas, embeddings)

        # Assign vector_ids back to chunks
        for db_chunk, vec_id in zip(db_chunk_objects, vector_ids):
            db_chunk.vector_id = vec_id
        
        doc.status = "Indexed"
        doc.chunk_count = len(db_chunk_objects)
        doc.error_message = None
        db.commit()

        logger.info(f"Successfully processed & indexed document {doc.filename} ({doc.id}) with {doc.chunk_count} chunks.")

    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "Failed"
            doc.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Read file content length
    contents = await file.read()
    file_size = len(contents)
    
    # Validate format & size
    is_valid, err_msg = DocumentService.validate_file(file.filename, file_size)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)

    doc_id = str(uuid.uuid4())
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, current_user.id)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    file_path = os.path.join(user_upload_dir, f"{doc_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(contents)

    ext = os.path.splitext(file.filename)[1].lower().replace(".", "")

    new_doc = Document(
        id=doc_id,
        user_id=current_user.id,
        filename=file.filename,
        file_type=ext,
        file_size=file_size,
        file_path=file_path,
        status="Uploaded",
        chunk_count=0,
        page_count=0
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    # Process indexing synchronously for instant UI update
    from app.core.database import SessionLocal
    process_and_index_document(new_doc.id, SessionLocal)
    db.refresh(new_doc)

    return new_doc


@router.get("", response_model=List[DocumentResponse])
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    docs = db.query(Document).filter(Document.user_id == current_user.id).order_by(Document.created_at.desc()).all()
    return docs


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # Remove from FAISS Vector Store
    vector_service.delete_document_chunks(doc.id)

    # Remove file from disk
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception as e:
            logger.warning(f"Could not remove document file {doc.file_path}: {e}")

    db.delete(doc)
    db.commit()
    return None


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
def reindex_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    from app.core.database import SessionLocal
    process_and_index_document(doc.id, SessionLocal)
    db.refresh(doc)
    return doc
