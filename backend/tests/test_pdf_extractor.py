import pymupdf as fitz
import pytest

from app.core.exceptions import InvalidDocumentError
from app.services.pdf_extractor import PdfExtractor
from tests.fakes import make_pdf_bytes


def write_pdf(tmp_path, pages):
    path = tmp_path / "report.pdf"
    path.write_bytes(make_pdf_bytes(pages))
    return path


def test_extracts_text_layer_without_ocr(tmp_path):
    path = write_pdf(tmp_path, ["A roof fall occurred in the east stope during the night shift of the mine."])
    pages = PdfExtractor(ocr_enabled=False).extract(path)
    assert len(pages) == 1
    assert "roof fall" in pages[0].text
    assert pages[0].used_ocr is False


def test_ocr_is_used_for_pages_without_text(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()

    extractor = PdfExtractor(ocr_enabled=True)
    monkeypatch.setattr(extractor, "_ocr_page", lambda page, number: "Recovered text from a scanned incident report")
    pages = extractor.extract(path)
    assert pages[0].used_ocr is True
    assert "scanned incident report" in pages[0].text


def test_text_pages_do_not_trigger_ocr(tmp_path, monkeypatch):
    path = write_pdf(tmp_path, ["Plenty of readable text about methane monitoring in the return airway."])
    extractor = PdfExtractor(ocr_enabled=True, min_chars=20)

    def fail(page, number):
        raise AssertionError("OCR should not run")

    monkeypatch.setattr(extractor, "_ocr_page", fail)
    assert extractor.extract(path)[0].used_ocr is False


def test_corrupt_file_raises_invalid_document(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(InvalidDocumentError):
        PdfExtractor(ocr_enabled=False).extract(path)
