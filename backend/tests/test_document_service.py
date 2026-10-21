import pytest

from app.core.config import Settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidDocumentError,
)
from app.db.models import Chunk
from app.repositories.chunk_repository import ChunkRepository
from app.services.document_service import DocumentService
from app.services.index_sync import sync_vector_store
from app.services.pdf_extractor import PdfExtractor
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore
from tests.fakes import FakeEmbedder, make_pdf_bytes

ROOF_FALL = "A roof fall trapped two miners in the east stope after loose strata were not scaled before the shift."
GAS = "Methane accumulated in the return airway because the auxiliary fan had stopped working during maintenance."


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", ocr_enabled=False, chunk_size=300, chunk_overlap=30, max_upload_mb=1)


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "data" / "faiss.index")


@pytest.fixture
def service(db_session, settings, store):
    extractor = PdfExtractor(ocr_enabled=False)
    return DocumentService(db_session, settings, extractor, FakeEmbedder(), store)


def test_ingest_stores_document_chunks_and_vectors(service, db_session, store):
    document = service.ingest("east-stope.pdf", make_pdf_bytes([ROOF_FALL, GAS]))
    assert document.page_count == 2
    assert document.chunk_count == 2
    assert store.size == 2
    assert len(ChunkRepository(db_session).ids_for_document(document.id)) == 2


def test_ingested_text_can_be_retrieved(service, db_session, store):
    service.ingest("report.pdf", make_pdf_bytes([ROOF_FALL, GAS]))
    retrieval = RetrievalService(db_session, FakeEmbedder(), store)
    best = retrieval.retrieve("what caused the methane in the airway", top_k=2)[0]
    assert best.page_number == 2
    assert best.document_name == "report.pdf"


def test_duplicate_upload_is_rejected(service):
    content = make_pdf_bytes([ROOF_FALL])
    service.ingest("a.pdf", content)
    with pytest.raises(DuplicateDocumentError):
        service.ingest("b.pdf", content)


def test_validation_errors(service):
    with pytest.raises(InvalidDocumentError):
        service.ingest("notes.txt", b"%PDF-1.4")
    with pytest.raises(InvalidDocumentError):
        service.ingest("fake.pdf", b"not a pdf at all")
    with pytest.raises(FileTooLargeError):
        service.ingest("big.pdf", b"%PDF" + b"0" * (2 * 1024 * 1024))


def test_pdf_without_text_is_rejected_and_nothing_is_kept(service, db_session, store, settings):
    with pytest.raises(EmptyDocumentError):
        service.ingest("blank.pdf", make_pdf_bytes([""]))
    assert store.size == 0
    assert service.list_documents() == []
    assert list(settings.upload_dir.glob("*.pdf")) == []


def test_delete_removes_everything(service, db_session, store):
    document = service.ingest("report.pdf", make_pdf_bytes([ROOF_FALL, GAS]))
    stored_file = document.stored_path
    service.delete(document.id)
    assert store.size == 0
    assert service.list_documents() == []
    assert not __import__("pathlib").Path(stored_file).exists()
    with pytest.raises(DocumentNotFoundError):
        service.delete(document.id)


def test_index_is_rebuilt_from_database_when_missing(service, db_session, store, tmp_path):
    service.ingest("report.pdf", make_pdf_bytes([ROOF_FALL, GAS]))
    fresh_store = VectorStore(tmp_path / "somewhere-else" / "faiss.index")
    assert fresh_store.size == 0

    assert sync_vector_store(db_session, fresh_store) is True
    assert fresh_store.size == 2
    assert sync_vector_store(db_session, fresh_store) is False


def test_chunks_without_embeddings_are_ignored_by_sync(db_session, service, tmp_path):
    document = service.ingest("report.pdf", make_pdf_bytes([ROOF_FALL]))
    db_session.add(Chunk(document_id=document.id, page_number=1, chunk_index=9, text="no embedding here at all", embedding=None))
    db_session.commit()
    store = VectorStore(tmp_path / "other" / "faiss.index")
    sync_vector_store(db_session, store)
    assert store.size == 1
