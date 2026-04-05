"""Chunking utilities."""

from .line_chunker import MAX_DOCUMENT_BYTES, chunk_record, chunk_records

__all__ = ["MAX_DOCUMENT_BYTES", "chunk_record", "chunk_records"]
