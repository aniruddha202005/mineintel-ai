"""Data access for documents."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, document: Document) -> Document:
        self.db.add(document)
        self.db.flush()
        return document

    def get(self, document_id: int) -> Document | None:
        return self.db.get(Document, document_id)

    def get_by_hash(self, sha256: str) -> Document | None:
        return self.db.scalar(select(Document).where(Document.sha256 == sha256))

    def list_all(self) -> list[Document]:
        return list(self.db.scalars(select(Document).order_by(Document.created_at.desc(), Document.id.desc())))

    def delete(self, document: Document) -> None:
        self.db.delete(document)
        self.db.flush()
