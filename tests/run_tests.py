import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.document_service import DocumentService, ExtractedDocument
from app.services.chunking_service import ChunkingService, TextChunk

def test_document_validation():
    print("[TEST] Document Validation...")
    valid, msg = DocumentService.validate_file("annual_report.pdf", 1024 * 1024)
    assert valid is True, f"Expected valid, got {msg}"
    
    invalid, msg = DocumentService.validate_file("malicious.exe", 100)
    assert invalid is False, "Expected invalid for exe"
    
    invalid_size, msg = DocumentService.validate_file("huge.pdf", 30 * 1024 * 1024)
    assert invalid_size is False, "Expected invalid for size > 25MB"
    print("  -> PASSED: Document Validation")

def test_sentence_chunking():
    print("[TEST] Sentence-Aware Chunking...")
    sample_text = (
        "DocuMind AI is an enterprise document intelligence and RAG platform. "
        "It processes PDF, DOCX, and TXT files using sentence transformers. "
        "The system breaks documents into configurable chunks with overlap. "
        "Each chunk contains document ID, page number, and chunk index. "
        "FAISS vector store stores dense embeddings for high-speed similarity search."
    )
    doc = ExtractedDocument("test.txt", "txt", 500, [(1, sample_text)], 1)
    chunker = ChunkingService(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_extracted_document(doc)

    assert len(chunks) >= 1, "Expected at least 1 chunk"
    assert chunks[0].page_number == 1, "Expected page 1"
    assert chunks[0].chunk_index == 0, "Expected index 0"
    assert "DocuMind" in chunks[0].text, "Expected 'DocuMind' in chunk text"
    assert chunks[0].token_count > 0, "Expected word count > 0"
    print(f"  -> PASSED: Sentence Chunking ({len(chunks)} chunks produced)")

def test_text_extraction_txt():
    print("[TEST] TXT Text Extraction...")
    sample_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/sample_documents/Annual_Report_2025.txt"))
    if os.path.exists(sample_file):
        size = os.path.getsize(sample_file)
        extracted = DocumentService.extract_text(sample_file, "Annual_Report_2025.txt", size)
        assert len(extracted.pages) >= 1, "Expected extracted pages"
        assert extracted.total_pages >= 1
        assert "DocuMind Enterprise Corp" in extracted.pages[0][1]
        print(f"  -> PASSED: Text Extraction ({extracted.total_pages} page(s) extracted)")
    else:
        print("  -> SKIPPED: Sample TXT file not found")

if __name__ == "__main__":
    print("=" * 50)
    print("RUNNING DOCUMIND AI LIGHTWEIGHT VERIFICATION SUITE")
    print("=" * 50)
    try:
        test_document_validation()
        test_sentence_chunking()
        test_text_extraction_txt()
        print("=" * 50)
        print("ALL LIGHTWEIGHT UNIT TESTS PASSED SUCCESSFULLY!")
        print("=" * 50)
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
