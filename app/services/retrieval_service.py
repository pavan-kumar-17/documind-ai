from typing import List, Optional
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service, VectorSearchResult

class RetrievalService:
    @classmethod
    def retrieve(
        cls,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[VectorSearchResult]:
        if not query.strip():
            return []

        # Step 1: Embed query
        query_embedding = embedding_service.generate_embedding(query)

        # Step 2: Query vector index
        results = vector_service.similarity_search(
            query_vector=query_embedding,
            top_k=top_k,
            document_ids=document_ids
        )

        return results

retrieval_service = RetrievalService()
