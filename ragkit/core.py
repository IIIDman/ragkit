"""Core data structures for RAGKit."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class Document:
    """A document with content and metadata."""
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


@dataclass
class Chunk:
    """A chunk of a document with optional embedding."""
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        has_emb = self.embedding is not None
        return f"Chunk(content='{preview}', has_embedding={has_emb})"


@dataclass
class Answer:
    """An answer from the RAG pipeline with sources."""
    
    text: str
    sources: List[str] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    
    def __repr__(self) -> str:
        preview = self.text[:100] + "..." if len(self.text) > 100 else self.text
        return f"Answer(text='{preview}', sources={self.sources})"
    
    def __str__(self) -> str:
        return self.text
