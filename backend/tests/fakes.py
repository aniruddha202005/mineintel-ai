"""Small deterministic stand-ins for the embedding model and the LLM."""

import re
import zlib
from collections.abc import Sequence

import pymupdf as fitz
import numpy as np
from langchain_core.runnables import RunnableLambda

from app.services.embeddings import l2_normalize

DIMENSION = 256


class FakeEmbedder:
    """Hashes words into a fixed-size vector, so texts sharing words score higher."""

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DIMENSION, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(token.encode()) % DIMENSION] += 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._vector(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text).reshape(1, -1))[0]


def fake_llm(reply: str = "Loose strata caused the roof fall [1].") -> RunnableLambda:
    """A runnable that ignores the prompt and returns a fixed reply."""
    return RunnableLambda(lambda _: reply)


def make_pdf_bytes(pages: Sequence[str]) -> bytes:
    """Build a small text PDF with one page per string."""
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data
