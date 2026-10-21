"""Text cleaning and page-aware chunking."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN_CHUNK_CHARS = 20

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\n(\w)")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_SPACES = re.compile(r"[ \t]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk of text before it is stored, tagged with the page it came from."""

    page_number: int
    chunk_index: int
    text: str


def clean_text(raw: str) -> str:
    """Normalise text extracted from a PDF page.

    Joins words split by a hyphen at a line break, turns single line breaks
    into spaces, collapses repeated whitespace and removes control characters.
    Paragraph breaks (blank lines) are kept.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)
    text = _HYPHEN_LINE_BREAK.sub(r"\1\2", text)
    text = _SINGLE_NEWLINE.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def chunk_pages(pages: Sequence[tuple[int, str]], chunk_size: int, chunk_overlap: int) -> list[ChunkDraft]:
    """Split page texts into overlapping chunks.

    Each page is chunked separately so every chunk keeps an exact page number
    for citations. ``chunk_index`` counts chunks across the whole document.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    drafts: list[ChunkDraft] = []
    for page_number, raw in pages:
        cleaned = clean_text(raw)
        if not cleaned:
            continue
        for piece in splitter.split_text(cleaned):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            drafts.append(ChunkDraft(page_number=page_number, chunk_index=len(drafts), text=piece))
    return drafts
