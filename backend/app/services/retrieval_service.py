"""Semantic search over stored chunks."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.repositories.chunk_repository import ChunkRepository
from app.services.embeddings import Embedder
from app.services.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    score: float


class RetrievalService:
    """Embeds a query, searches FAISS and loads the matching chunks from the database."""

    def __init__(self, db: Session, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store
        self.chunks = ChunkRepository(db)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, best first."""
        if top_k <= 0 or self.store.size == 0:
            return []
        hits = self.store.search(self.embedder.embed_query(query), top_k)
        rows = self.chunks.get_by_ids([chunk_id for chunk_id, _ in hits])

        results: list[RetrievedChunk] = []
        for chunk_id, score in hits:
            row = rows.get(chunk_id)
            if row is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    document_name=row.document.filename,
                    page_number=row.page_number,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    score=score,
                )
            )
        return results
