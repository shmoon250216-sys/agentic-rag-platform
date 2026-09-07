from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=20, max_length=20000)


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    chunk_count: int
    content_length: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total_chunks: int


class DocumentCreateResponse(BaseModel):
    document: DocumentSummary
