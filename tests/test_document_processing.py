import pytest
from app.services.document_service import DocumentService

def test_file_validation():
    valid, msg = DocumentService.validate_file("report.pdf", 1024 * 1024)
    assert valid is True
    assert msg == ""

    invalid_ext, msg_ext = DocumentService.validate_file("malicious.exe", 1024)
    assert invalid_ext is False
    assert "Unsupported file format" in msg_ext

    invalid_size, msg_size = DocumentService.validate_file("huge.pdf", 30 * 1024 * 1024)
    assert invalid_size is False
    assert "File size exceeds" in msg_size
