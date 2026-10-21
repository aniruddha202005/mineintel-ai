"""Document upload, listing and deletion."""

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from app.api.deps import get_document_service
from app.core.config import Settings, get_settings
from app.schemas.document import DocumentOut
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
) -> DocumentOut:
    content = file.file.read(settings.max_upload_bytes + 1)
    return service.ingest(file.filename or "upload.pdf", content)


@router.get("", response_model=list[DocumentOut])
def list_documents(service: DocumentService = Depends(get_document_service)) -> list[DocumentOut]:
    return service.list_documents()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, service: DocumentService = Depends(get_document_service)) -> Response:
    service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
