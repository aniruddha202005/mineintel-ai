"""Data access for chunks."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.db.models import Chunk


class ChunkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_many(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        self.db.add_all(chunks)
        self.db.flush()
        return list(chunks)

    def get_by_ids(self, chunk_ids: Sequence[int]) -> dict[int, Chunk]:
        if not chunk_ids:
            return {}
        statement = select(Chunk).options(joinedload(Chunk.document)).where(Chunk.id.in_(chunk_ids))
        return {chunk.id: chunk for chunk in self.db.scalars(statement).unique()}

    def ids_for_document(self, document_id: int) -> list[int]:
        return list(self.db.scalars(select(Chunk.id).where(Chunk.document_id == document_id)))

    def all_embeddings(self) -> list[tuple[int, bytes]]:
        """Return (chunk id, raw embedding bytes) for every chunk that has an embedding."""
        statement = select(Chunk.id, Chunk.embedding).where(Chunk.embedding.is_not(None)).order_by(Chunk.id)
        return [(row[0], row[1]) for row in self.db.execute(statement)]
