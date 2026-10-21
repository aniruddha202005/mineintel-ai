"""Embedding backends. Both return L2-normalised float32 vectors."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.exceptions import EmbeddingError


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so inner product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class LocalEmbedder:
    """Runs a sentence-transformers model on the local CPU."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._load().encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return l2_normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class HuggingFaceApiEmbedder:
    """Calls the Hugging Face Inference API so the server does not need torch."""

    def __init__(self, model_name: str, token: str) -> None:
        from huggingface_hub import InferenceClient

        self.model_name = model_name
        self._client = InferenceClient(token=token or None)

    def _embed_one(self, text: str) -> np.ndarray:
        try:
            raw = self._client.feature_extraction(text, model=self.model_name)
        except Exception as exc:
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc
        vector = np.asarray(raw, dtype=np.float32)
        if vector.ndim > 1:
            vector = vector.reshape(-1, vector.shape[-1]).mean(axis=0)
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalize(np.vstack([self._embed_one(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._embed_one(text).reshape(1, -1))[0]


def build_embedder(settings: Settings) -> Embedder:
    """Create the embedder selected by configuration."""
    provider = settings.embedding_provider.lower()
    if provider == "local":
        return LocalEmbedder(settings.embedding_model)
    if provider == "hf_api":
        return HuggingFaceApiEmbedder(settings.embedding_model, settings.hf_api_token)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
