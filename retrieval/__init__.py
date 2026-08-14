"""Retrieval strategies under comparison."""

from .base import Retriever
from .bm25 import BM25Retriever, tokenize
from .dense import DenseRetriever
from .hybrid import HybridRetriever, reciprocal_rank_fusion

METHODS = ("bm25", "dense", "hybrid")

__all__ = [
    "Retriever", "BM25Retriever", "DenseRetriever", "HybridRetriever",
    "reciprocal_rank_fusion", "tokenize", "METHODS",
]
