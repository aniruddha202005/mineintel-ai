"""FastAPI dependencies. Heavy objects are created once and reused."""

from functools import lru_cache

from fastapi import Depends
from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.services.document_service import DocumentService
from app.services.embeddings import Embedder, build_embedder
from app.services.llm import build_llm
from app.services.pdf_extractor import PdfExtractor
from app.services.qa_service import QAService
from app.services.query_service import QueryService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore


@lru_cache
def get_embedder() -> Embedder:
    return build_embedder(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(get_settings().index_path)


@lru_cache
def get_llm() -> BaseChatModel:
    return build_llm(get_settings())


def get_pdf_extractor(settings: Settings = Depends(get_settings)) -> PdfExtractor:
    return PdfExtractor(
        ocr_enabled=settings.ocr_enabled,
        ocr_dpi=settings.ocr_dpi,
        ocr_language=settings.ocr_language,
        min_chars=settings.min_page_chars,
    )


def get_document_service(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    extractor: PdfExtractor = Depends(get_pdf_extractor),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> DocumentService:
    return DocumentService(db, settings, extractor, embedder, store)


def get_retrieval_service(
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
) -> RetrievalService:
    return RetrievalService(db, embedder, store)


def get_qa_service(
    retrieval: RetrievalService = Depends(get_retrieval_service),
    llm: BaseChatModel = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> QAService:
    return QAService(retrieval, llm, settings)


def get_query_service(
    db: Session = Depends(get_db),
    qa: QAService = Depends(get_qa_service),
) -> QueryService:
    return QueryService(db, qa)
