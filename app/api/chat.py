import json
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.domain import User, ChatSession, ChatMessage, QueryHistory, Document
from app.schemas.pydantic_schemas import (
    ChatQueryRequest, ChatQueryResponse, ChatSessionResponse,
    ChatSessionDetailResponse, ChatMessageResponse, SourceCitation,
    DocumentCompareRequest, DocumentCompareResponse
)
from app.api.deps import get_current_user
from app.services.rag_service import rag_service

router = APIRouter(prefix="/chat", tags=["AI RAG Chat"])

@router.post("/query", response_model=ChatQueryResponse)
def ask_question(
    query_in: ChatQueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not query_in.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query text cannot be empty.")

    # Get or create chat session
    session_id = query_in.session_id
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
        if not session:
            session_id = None

    if not session_id:
        session_id = str(uuid.uuid4())
        # Generate summary title from query
        title = query_in.query[:40] + ("..." if len(query_in.query) > 40 else "")
        session = ChatSession(id=session_id, user_id=current_user.id, title=title)
        db.add(session)
        db.commit()

    # Save User message
    user_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        sender="user",
        content=query_in.query,
        sources_json=None
    )
    db.add(user_msg)
    db.commit()

    # Execute RAG Pipeline
    rag_result = rag_service.execute_rag_pipeline(
        query=query_in.query,
        top_k=query_in.top_k or 5,
        document_ids=query_in.document_ids
    )

    answer_text = rag_result["answer"]
    sources: List[SourceCitation] = rag_result["sources"]
    sources_json_str = json.dumps([s.model_dump() for s in sources])

    # Save Assistant message
    assistant_msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        sender="assistant",
        content=answer_text,
        sources_json=sources_json_str
    )
    db.add(assistant_msg)

    # Save Query History metric
    history = QueryHistory(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        query_text=query_in.query,
        answer=answer_text,
        retrieved_chunks_count=rag_result["retrieved_count"],
        response_time_ms=rag_result["response_time_ms"]
    )
    db.add(history)
    db.commit()

    return ChatQueryResponse(
        session_id=session_id,
        query=query_in.query,
        answer=answer_text,
        sources=sources,
        response_time_ms=rag_result["response_time_ms"]
    )


@router.post("/compare", response_model=DocumentCompareResponse)
def compare_documents(
    req: DocumentCompareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc_a = db.query(Document).filter(Document.id == req.document_a_id, Document.user_id == current_user.id).first()
    doc_b = db.query(Document).filter(Document.id == req.document_b_id, Document.user_id == current_user.id).first()

    if not doc_a or not doc_b:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or both selected documents were not found.")

    res = rag_service.compare_documents(
        doc_a_id=doc_a.id,
        doc_a_name=doc_a.filename,
        doc_b_id=doc_b.id,
        doc_b_name=doc_b.filename,
        user_prompt=req.prompt or "What are the main similarities, differences, and key changes between these two documents?"
    )
    return res


@router.get("/sessions", response_model=List[ChatSessionResponse])
def get_chat_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.updated_at.desc()).all()
    res = []
    for s in sessions:
        msg_count = db.query(ChatMessage).filter(ChatMessage.session_id == s.id).count()
        res.append(ChatSessionResponse(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=msg_count
        ))
    return res


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailResponse)
def get_chat_session_details(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.asc()).all()

    msg_responses = []
    for m in messages:
        sources_list = []
        if m.sources_json:
            try:
                raw_sources = json.loads(m.sources_json)
                sources_list = [SourceCitation(**s) for s in raw_sources]
            except Exception:
                sources_list = []

        msg_responses.append(ChatMessageResponse(
            id=m.id,
            session_id=m.session_id,
            sender=m.sender,
            content=m.content,
            sources=sources_list,
            created_at=m.created_at
        ))

    return ChatSessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(msg_responses),
        messages=msg_responses
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")

    db.delete(session)
    db.commit()
    return None
