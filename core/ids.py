"""Stable identifier construction.

IDs must be reproducible across runs and machines, readable enough to debug a
retrieval result by eye, and safe to use in filenames. Raw chunk text is never
used as an identifier.
"""

from __future__ import annotations

import hashlib
import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(value: str, max_len: int = 48) -> str:
    """Filesystem- and log-safe rendering of an arbitrary string.

    Long values keep a readable prefix and gain a hash suffix so that two
    documents with a shared prefix (common in CUAD contract titles) cannot
    collide.
    """
    cleaned = _UNSAFE.sub("_", value.strip()).strip("_")
    if not cleaned:
        cleaned = "x"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:max_len]}~{digest}"


def doc_id(dataset: str, split: str, local_id: str) -> str:
    return f"{dataset}/{split}/{slug(local_id)}"


def query_id(dataset: str, split: str, local_id: str) -> str:
    return f"{dataset}/{split}/q/{slug(local_id, max_len=64)}"


def chunk_id(document_id: str, ordinal: int) -> str:
    """Chunk ids are stable for a given (document, chunking configuration).

    The chunking configuration is not encoded here; it is encoded in the corpus
    build directory, so a chunk id is unambiguous within the corpus that
    contains it while staying short and readable.
    """
    return f"{document_id}#c{ordinal:05d}"


def content_hash(text: str) -> str:
    """Hash used to de-duplicate source documents that share identical text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def run_id(prefix: str = "run") -> str:
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:12]}"
