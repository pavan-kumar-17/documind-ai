from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

def test_embedding_and_faiss_vector_indexing():
    text_chunks = [
        {"chunk_id": "c1", "document_id": "doc1", "document_name": "Report.pdf", "page_number": 1, "chunk_index": 0, "text": "DocuMind total revenue grew by 28.4 percent in fiscal year 2025."},
        {"chunk_id": "c2", "document_id": "doc1", "document_name": "Report.pdf", "page_number": 2, "chunk_index": 1, "text": "Net income reached 28.4 million with strong cash flow generation."}
    ]

    texts = [c["text"] for c in text_chunks]
    embeddings = embedding_service.generate_embeddings_batch(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 dimension

    vector_ids = vector_service.add_chunks(text_chunks, embeddings)
    assert len(vector_ids) == 2

    # Query search
    query_emb = embedding_service.generate_embedding("What was the revenue growth percentage?")
    results = vector_service.similarity_search(query_emb, top_k=2)

    assert len(results) > 0
    assert results[0].document_id == "doc1"
    assert "revenue" in results[0].text.lower()
