from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_db
from app.core.config import settings
from app.services.vector_service import vector_service

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "app_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": db_status,
        "vector_store": {
            "type": "FAISS",
            "indexed_vectors": vector_service.vector_count,
            "dimension": vector_service.dim
        },
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER
    }
