from backend.app.storage.local import detect_content
import pytest
def test_detects_binary_signatures_without_filename_trust():
    assert detect_content(b"%PDF-1.7\n") == "application/pdf"
    assert detect_content(b"\x89PNG\r\n\x1a\nabc") == "image/png"
def test_rejects_invalid_binary_as_text():
    with pytest.raises(ValueError): detect_content(b"\xff\x00\x81")
