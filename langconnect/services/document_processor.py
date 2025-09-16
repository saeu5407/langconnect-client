import logging
import uuid

from fastapi import UploadFile
from langchain_community.document_loaders.parsers import (
    BS4HTMLParser,
    PDFPlumberParser,
)
from langchain_community.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_core.documents.base import Blob, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOGGER = logging.getLogger(__name__)

# Document Parser Configuration
HANDLERS = {
    "application/pdf": PDFPlumberParser(),
    "text/plain": TextParser(),
    "text/html": BS4HTMLParser(),
    "text/markdown": TextParser(),  # Markdown files
    "text/x-markdown": TextParser(),  # Alternative markdown MIME type
    "application/msword": MsWordParser(),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        MsWordParser()
    ),
}

SUPPORTED_MIMETYPES = sorted(HANDLERS.keys())

# 여러 파서들을 모아놓고, MIME 타입에 따라 적절한 파서를 골라주는 라우터 클래스
# HANDLERS를 입력으로 받음(위의 내용)
# 핸들러의 키값은 MIME(타입/서브타입 형식) 타입으로 설정해야 함
MIMETYPE_BASED_PARSER = MimeTypeBasedParser(
    handlers=HANDLERS,
    fallback_parser=None,
)


async def process_document(
    file: UploadFile,
    metadata: dict | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Process an uploaded file into LangChain documents."""
    # Generate a unique ID for this file processing instance : 키값 하나 생성해둠
    file_id = uuid.uuid4()

    contents = await file.read()

    # Determine the actual mime type
    mime_type = file.content_type or "text/plain"

    # Handle application/octet-stream by checking file extension
    if mime_type == "application/octet-stream" and file.filename:
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".md") or filename_lower.endswith(".markdown"):
            mime_type = "text/markdown"
        elif filename_lower.endswith(".txt"):
            mime_type = "text/plain"
        elif filename_lower.endswith(".html") or filename_lower.endswith(".htm"):
            mime_type = "text/html"
        elif filename_lower.endswith(".pdf"):
            mime_type = "application/pdf"
        elif filename_lower.endswith(".doc"):
            mime_type = "application/msword"
        elif filename_lower.endswith(".docx"):
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    # MimeTypeBasedParser에서 입력받아야 하는 형식 : Blob 컨테이너(랭체인에서 쓰는 객체)
    blob = Blob(data=contents, mimetype=mime_type)

    docs = MIMETYPE_BASED_PARSER.parse(blob) # 파싱 수행

    # Add provided metadata to each document
    # 메타데이터가 있는데 doc에 정의되지 않았다면 doc.metadata에 정의
    if metadata:
        for doc in docs:
            # Ensure metadata attribute exists and is a dict
            if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
                doc.metadata = {}
            # Update with provided metadata, preserving existing keys if not overridden
            doc.metadata.update(metadata)

    # Create text splitter with provided parameters
    # 리커시브스플리터 적용
    # TODO : 스플리터 개선 영억
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split documents
    split_docs = text_splitter.split_documents(docs)

    # Add the generated file_id to all split documents' metadata
    # 메타데이터에 file_id(키값) 저장
    for split_doc in split_docs:
        if not hasattr(split_doc, "metadata") or not isinstance(
            split_doc.metadata, dict
        ):
            split_doc.metadata = {}  # Initialize if it doesn't exist
        split_doc.metadata["file_id"] = str(
            file_id
        )  # Store as string for compatibility

    return split_docs