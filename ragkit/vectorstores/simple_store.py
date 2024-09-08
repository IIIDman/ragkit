"""Simple in-memory vector store using NumPy."""

from typing import List, Optional, Tuple
import numpy as np
import pickle
from pathlib import Path

from ..core import Chunk


class SimpleStore:
    """
    Simple in-memory vector store using NumPy.
    
    Good for small datasets (< 10k chunks) where FAISS is overkill.
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize simple store.
        
        Args:
            embedding_model: Embedding model with embed_documents method
        """
        self.embedding_model = embedding_model
        self._chunks: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None
    
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to the store."""
        if not chunks:
            return
        
        # Get texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        if self.embedding_model:
            new_embeddings = self.embedding_model.embed_documents(texts)
            
            # Store in chunks
            for chunk, emb in zip(chunks, new_embeddings):
                chunk.embedding = emb
        else:
            new_embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Append to existing
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
        
        self._chunks.extend(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 4
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if not self._chunks or self._embeddings is None:
            return []
        
        # Embed query
        if self.embedding_model:
            query_embedding = self.embedding_model.embed_query(query)
        else:
            raise ValueError("No embedding model provided")
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarities (embeddings assumed normalized)
        similarities = np.dot(self._embeddings, query_embedding)
        
        # Get top-k indices
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return chunks with scores
        results = []
        for idx in top_indices:
            results.append((self._chunks[idx], float(similarities[idx])))
        
        return results
    
    def save(self, path: str) -> None:
        """Save store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self._embeddings is not None:
            np.save(path / "embeddings.npy", self._embeddings)
        
        # Save chunks
        chunks_data = []
        for chunk in self._chunks:
            chunks_data.append({
                "content": chunk.content,
                "metadata": chunk.metadata
            })
        
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(chunks_data, f)
    
    @classmethod
    def load(cls, path: str, embedding_model=None) -> "SimpleStore":
        """Load store from disk."""
        path = Path(path)
        
        store = cls(embedding_model=embedding_model)
        
        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            store._embeddings = np.load(embeddings_path)
        
        # Load chunks
        with open(path / "chunks.pkl", "rb") as f:
            chunks_data = pickle.load(f)
        
        store._chunks = [
            Chunk(content=d["content"], metadata=d["metadata"])
            for d in chunks_data
        ]
        
        return store
    
    def __len__(self) -> int:
        return len(self._chunks)
