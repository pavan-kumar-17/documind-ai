from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.models.domain import User, Document, DocumentChunk, ChatSession, QueryHistory
from app.schemas.pydantic_schemas import DashboardStatsResponse, EvaluationRunResponse, EvaluationRunRequest
from app.api.deps import get_current_user
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Admin"])

@router.get("/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    total_docs = db.query(Document).filter(Document.user_id == current_user.id).count()
    indexed_docs = db.query(Document).filter(Document.user_id == current_user.id, Document.status == "Indexed").count()
    processing_docs = db.query(Document).filter(Document.user_id == current_user.id, Document.status == "Processing").count()
    failed_docs = db.query(Document).filter(Document.user_id == current_user.id, Document.status == "Failed").count()
    
    total_chunks = db.query(DocumentChunk).join(Document).filter(Document.user_id == current_user.id).count()
    total_queries = db.query(QueryHistory).filter(QueryHistory.user_id == current_user.id).count()
    total_sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).count()

    avg_time = db.query(func.avg(QueryHistory.response_time_ms)).filter(QueryHistory.user_id == current_user.id).scalar() or 0.0

    return DashboardStatsResponse(
        total_documents=total_docs,
        indexed_documents=indexed_docs,
        processing_documents=processing_docs,
        failed_documents=failed_docs,
        total_chunks=total_chunks,
        total_queries=total_queries,
        total_sessions=total_sessions,
        average_response_time_ms=round(float(avg_time), 2)
    )

@router.post("/run", response_model=EvaluationRunResponse)
def run_rag_evaluation(
    req: EvaluationRunRequest = EvaluationRunRequest(),
    current_user: User = Depends(get_current_user)
):
    res = EvaluationService.run_retrieval_evaluation(top_k=req.top_k)
    return res
