"""Embedding models for vector representations."""

from .sentence_transformer import SentenceTransformerEmbeddings, HuggingFaceEmbeddings

__all__ = [
    "SentenceTransformerEmbeddings",
    "HuggingFaceEmbeddings",
]
