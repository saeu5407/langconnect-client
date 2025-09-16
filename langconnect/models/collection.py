import datetime
from typing import Any

from pydantic import BaseModel, Field
# pydantic BaseModel은 클래스 __init__을 자동으로 생성

# =====================
# Collection Schemas
# =====================

# 콜렉션 생성 클래스(DB 콜렉션을 의미)
class CollectionCreate(BaseModel):
    """Schema for creating a new collection."""

    name: str = Field(..., description="The unique name of the collection.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for the collection."
    )

# 수정 클래스(디폴트를 None으로 해둠)
class CollectionUpdate(BaseModel):
    """Schema for updating an existing collection."""

    name: str | None = Field(None, description="New name for the collection.")
    metadata: dict[str, Any] | None = Field(
        None, description="Updated metadata for the collection."
    )


class CollectionResponse(BaseModel):
    """Schema for representing a collection from PGVector."""

    # PGVector table has uuid (id), name (str), and cmetadata (JSONB)
    # We get these from list/get db functions
    uuid: str = Field(
        ..., description="The unique identifier of the collection in PGVector."
    )
    name: str = Field(..., description="The name of the collection.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with the collection."
    )
    document_count: int = Field(0, description="The number of documents in the collection.")
    chunk_count: int = Field(0, description="The number of chunks in the collection.")

    class Config:
        # Allows creating model from dict like
        # {'uuid': '...', 'name': '...', 'metadata': {...}}
        from_attributes = True


# =====================
# Document Schemas
# =====================


class DocumentBase(BaseModel):
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    collection_id: str
    embedding: list[float] | None = (
        None  # Embedding can be added during creation or later
    )


class DocumentUpdate(BaseModel):
    page_content: str | None = None
    metadata: dict[str, Any] | None = None
    embedding: list[float] | None = None


class DocumentResponse(DocumentBase):
    id: str
    collection_id: str
    embedding: list[float] | None = None  # Represent embedding as list of floats
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True
        from_attributes = True  # Pydantic v2 way
