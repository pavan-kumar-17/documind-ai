from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.domain import User
from app.schemas.pydantic_schemas import SemanticSearchRequest, SemanticSearchResponse, SemanticSearchResult
from app.api.deps import get_current_user
from app.services.retrieval_service import retrieval_service

router = APIRouter(prefix="/search", tags=["Semantic Search"])

@router.post("", response_model=SemanticSearchResponse)
def semantic_search(
    req: SemanticSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not req.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    retrieved = retrieval_service.retrieve(
        query=req.query,
        top_k=req.top_k or 10,
        document_ids=req.document_ids
    )

    results = [
        SemanticSearchResult(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            document_name=r.document_name,
            page_number=r.page_number,
            chunk_index=r.chunk_index,
            text=r.text,
            similarity_score=round(r.similarity_score, 4)
        ) for r in retrieved
    ]

    return SemanticSearchResponse(
        query=req.query,
        total_results=len(results),
        results=results
    )
