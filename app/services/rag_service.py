import time
import logging
from typing import List, Dict, Any, Optional
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service
from app.schemas.pydantic_schemas import SourceCitation, DocumentCompareResponse, CompareItem

logger = logging.getLogger("documind.rag")

class RAGService:
    @classmethod
    def execute_rag_pipeline(
        cls,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Step 1: Semantic Retrieval
        retrieved_results = retrieval_service.retrieve(
            query=query,
            top_k=top_k,
            document_ids=document_ids
        )

        if not retrieved_results:
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                "answer": "I couldn't find enough information in the uploaded documents to answer this reliably.",
                "sources": [],
                "response_time_ms": round(elapsed_ms, 2),
                "retrieved_count": 0
            }

        # Step 2: Format context for LLM & build source citations
        context_chunks = []
        sources: List[SourceCitation] = []

        for res in retrieved_results:
            context_chunks.append({
                "chunk_id": res.chunk_id,
                "document_id": res.document_id,
                "document_name": res.document_name,
                "page_number": res.page_number,
                "chunk_index": res.chunk_index,
                "text": res.text,
                "similarity_score": res.similarity_score
            })

            sources.append(SourceCitation(
                document_id=res.document_id,
                document_name=res.document_name,
                page_number=res.page_number,
                chunk_id=res.chunk_id,
                similarity_score=round(res.similarity_score, 4),
                snippet=res.text[:250] + "..." if len(res.text) > 250 else res.text
            ))

        # Step 3: LLM Response Generation
        answer = llm_service.generate_grounded_answer(query, context_chunks)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(f"RAG query processed in {elapsed_ms:.2f}ms with {len(sources)} sources.")

        return {
            "answer": answer,
            "sources": sources,
            "response_time_ms": round(elapsed_ms, 2),
            "retrieved_count": len(retrieved_results)
        }

    @classmethod
    def compare_documents(
        cls,
        doc_a_id: str,
        doc_a_name: str,
        doc_b_id: str,
        doc_b_name: str,
        user_prompt: str = "Compare the key themes, financial metrics, and strategic objectives between these two documents."
    ) -> DocumentCompareResponse:
        start_time = time.time()

        # Retrieve top 5 chunks for Document A
        chunks_a = retrieval_service.retrieve(query=user_prompt, top_k=5, document_ids=[doc_a_id])
        # Retrieve top 5 chunks for Document B
        chunks_b = retrieval_service.retrieve(query=user_prompt, top_k=5, document_ids=[doc_b_id])

        sources_a = [
            SourceCitation(
                document_id=c.document_id, document_name=doc_a_name, page_number=c.page_number,
                chunk_id=c.chunk_id, similarity_score=round(c.similarity_score, 4), snippet=c.text[:200]
            ) for c in chunks_a
        ]

        sources_b = [
            SourceCitation(
                document_id=c.document_id, document_name=doc_b_name, page_number=c.page_number,
                chunk_id=c.chunk_id, similarity_score=round(c.similarity_score, 4), snippet=c.text[:200]
            ) for c in chunks_b
        ]

        # Generate comparative insights
        text_a_summary = "\n".join([c.text for c in chunks_a[:3]]) if chunks_a else "No relevant context extracted."
        text_b_summary = "\n".join([c.text for c in chunks_b[:3]]) if chunks_b else "No relevant context extracted."

        summary = f"DocuMind Comparative Analysis between **{doc_a_name}** and **{doc_b_name}**:\n\n"
        summary += f"1. **{doc_a_name} Highlights**:\n{text_a_summary[:300]}...\n\n"
        summary += f"2. **{doc_b_name} Highlights**:\n{text_b_summary[:300]}...\n\n"

        items = [
            CompareItem(
                topic="Primary Focus & Objectives",
                doc_a_findings=text_a_summary[:180] + "...",
                doc_b_findings=text_b_summary[:180] + "...",
                analysis=f"Comparison shows distinct updates between {doc_a_name} and {doc_b_name}."
            ),
            CompareItem(
                topic="Key Metrics / Extracted Findings",
                doc_a_findings=f"Top retrieved score: {chunks_a[0].similarity_score:.2f}" if chunks_a else "N/A",
                doc_b_findings=f"Top retrieved score: {chunks_b[0].similarity_score:.2f}" if chunks_b else "N/A",
                analysis="Metrics derived from semantic dense retrieval alignment."
            )
        ]

        return DocumentCompareResponse(
            document_a_name=doc_a_name,
            document_b_name=doc_b_name,
            summary_comparison=summary,
            items=items,
            sources_a=sources_a,
            sources_b=sources_b
        )

rag_service = RAGService()
