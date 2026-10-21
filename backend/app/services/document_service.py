"""Document ingestion and removal."""

import hashlib
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidDocumentError,
)
from app.db.models import Chunk, Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.embeddings import Embedder
from app.services.pdf_extractor import PdfExtractor
from app.services.text_processing import chunk_pages
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentService:
    """Runs the ingestion pipeline: validate, extract, chunk, embed, store, index."""

    def __init__(
        self,
        db: Session,
        settings: Settings,
        extractor: PdfExtractor,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self.db = db
        self.settings = settings
        self.extractor = extractor
        self.embedder = embedder
        self.store = store
        self.documents = DocumentRepository(db)
        self.chunks = ChunkRepository(db)

    def list_documents(self) -> list[Document]:
        return self.documents.list_all()

    def ingest(self, filename: str, content: bytes) -> Document:
        """Process an uploaded PDF and make it searchable."""
        safe_name = Path(filename).name or "upload.pdf"
        self._validate(safe_name, content)

        digest = hashlib.sha256(content).hexdigest()
        if self.documents.get_by_hash(digest) is not None:
            raise DuplicateDocumentError("This document has already been uploaded")

        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        stored_path = self.settings.upload_dir / f"{digest}.pdf"
        stored_path.write_bytes(content)

        try:
            pages = self.extractor.extract(stored_path)
            drafts = chunk_pages(
                [(page.page_number, page.text) for page in pages],
                self.settings.chunk_size,
                self.settings.chunk_overlap,
            )
            if not drafts:
                raise EmptyDocumentError("No readable text was found in this document")

            vectors = self.embedder.embed_documents([draft.text for draft in drafts])

            document = self.documents.add(
                Document(
                    filename=safe_name,
                    stored_path=str(stored_path),
                    sha256=digest,
                    page_count=len(pages),
                    ocr_page_count=sum(1 for page in pages if page.used_ocr),
                    chunk_count=len(drafts),
                )
            )
            rows = self.chunks.add_many(
                [
                    Chunk(
                        document_id=document.id,
                        page_number=draft.page_number,
                        chunk_index=draft.chunk_index,
                        text=draft.text,
                        embedding=vector.astype("float32").tobytes(),
                    )
                    for draft, vector in zip(drafts, vectors)
                ]
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            stored_path.unlink(missing_ok=True)
            raise

        self.store.add([row.id for row in rows], vectors)
        self.store.save()
        logger.info("Ingested %s: %s pages, %s chunks", safe_name, document.page_count, document.chunk_count)
        return document

    def delete(self, document_id: int) -> None:
        """Remove a document, its chunks, its stored file and its vectors."""
        document = self.documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} not found")

        chunk_ids = self.chunks.ids_for_document(document_id)
        stored_path = Path(document.stored_path)
        self.documents.delete(document)
        self.db.commit()

        self.store.remove(chunk_ids)
        self.store.save()
        stored_path.unlink(missing_ok=True)

    def _validate(self, filename: str, content: bytes) -> None:
        if not filename.lower().endswith(".pdf"):
            raise InvalidDocumentError("Only PDF files are supported")
        if len(content) > self.settings.max_upload_bytes:
            raise FileTooLargeError(f"File exceeds the {self.settings.max_upload_mb} MB limit")
        if not content.startswith(b"%PDF"):
            raise InvalidDocumentError("The file is not a valid PDF")
