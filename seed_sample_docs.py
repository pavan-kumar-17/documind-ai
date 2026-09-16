import sys
import os
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.services.document_service import DocumentService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

def seed_sample_documents():
    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents"))
    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        return

    sample_files = [f for f in os.listdir(sample_dir) if f.endswith((".txt", ".pdf", ".docx"))]
    print(f"Seeding {len(sample_files)} sample documents into FAISS vector database...")

    for fname in sample_files:
        fpath = os.path.join(sample_dir, fname)
        fsize = os.path.getsize(fpath)
        doc_id = f"doc_{fname.replace('.', '_')}"

        extracted = DocumentService.extract_text(fpath, fname, fsize)
        chunker = ChunkingService(chunk_size=800, chunk_overlap=150)
        chunks = chunker.chunk_extracted_document(extracted)

        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.generate_embeddings_batch(chunk_texts)

        vector_metas = []
        for raw_c in chunks:
            chunk_uuid = str(uuid.uuid4())
            vector_metas.append({
                "chunk_id": chunk_uuid,
                "document_id": doc_id,
                "document_name": fname,
                "page_number": raw_c.page_number,
                "chunk_index": raw_c.chunk_index,
                "text": raw_c.text
            })

        vector_service.add_chunks(vector_metas, embeddings)
        print(f"  Indexed {fname}: {len(chunks)} chunks added to FAISS index.")

    print(f"Total vector count in FAISS index: {vector_service.vector_count}")

if __name__ == "__main__":
    seed_sample_documents()
