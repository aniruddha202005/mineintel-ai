"""Document response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    page_count: int
    ocr_page_count: int
    chunk_count: int
    created_at: datetime
