"""PDF text extraction with an OCR fallback for scanned pages."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
import pytesseract
from PIL import Image

from app.core.exceptions import InvalidDocumentError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    used_ocr: bool


class PdfExtractor:
    """Read text from each page, running OCR when a page has almost no text layer."""

    def __init__(self, ocr_enabled: bool = True, ocr_dpi: int = 200, ocr_language: str = "eng", min_chars: int = 40) -> None:
        self.ocr_enabled = ocr_enabled
        self.ocr_dpi = ocr_dpi
        self.ocr_language = ocr_language
        self.min_chars = min_chars

    def extract(self, path: Path) -> list[PageText]:
        """Return the text of every page in order."""
        pages: list[PageText] = []
        try:
            with fitz.open(path) as pdf:
                if pdf.needs_pass:
                    raise InvalidDocumentError("Password-protected PDFs are not supported")
                for number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    used_ocr = False
                    if self.ocr_enabled and len(text) < self.min_chars:
                        ocr_text = self._ocr_page(page, number)
                        if len(ocr_text) > len(text):
                            text, used_ocr = ocr_text, True
                    pages.append(PageText(page_number=number, text=text, used_ocr=used_ocr))
        except InvalidDocumentError:
            raise
        except Exception as exc:
            raise InvalidDocumentError(f"Could not read the PDF: {exc}") from exc
        return pages

    def _ocr_page(self, page: "fitz.Page", number: int) -> str:
        """Render a page to an image and run Tesseract on it."""
        try:
            pixmap = page.get_pixmap(dpi=self.ocr_dpi)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            return pytesseract.image_to_string(image, lang=self.ocr_language).strip()
        except pytesseract.TesseractNotFoundError:
            logger.warning("Tesseract is not installed; page %s was skipped", number)
        except Exception:
            logger.exception("OCR failed on page %s", number)
        return ""
