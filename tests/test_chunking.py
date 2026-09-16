from app.services.chunking_service import ChunkingService
from app.services.document_service import ExtractedDocument

def test_chunking_service_sliding_window():
    text = (
        "DocuMind AI is an enterprise document intelligence platform. "
        "It processes PDF, DOCX, and TXT files using sentence transformers. "
        "The system breaks documents into configurable chunks with overlap. "
        "Each chunk contains document ID, page number, and chunk index. "
        "FAISS vector store stores dense embeddings for high-speed similarity search."
    )
    doc = ExtractedDocument("test.txt", "txt", 500, [(1, text)], 1)
    
    chunker = ChunkingService(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_extracted_document(doc)

    assert len(chunks) >= 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert "DocuMind" in chunks[0].text
    assert chunks[0].token_count > 0
