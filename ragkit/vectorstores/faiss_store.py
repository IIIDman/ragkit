"""FAISS vector store for efficient similarity search."""

from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path
import pickle

from ..core import Chunk


class FAISSStore:
    """
    Vector store using Facebook's FAISS library.
    
    Fast and memory-efficient similarity search.
    """
    
    def __init__(self, embedding_model=None, dimension: int = 384):
        """
        Initialize FAISS store.
        
        Args:
            embedding_model: Embedding model with embed_documents method
            dimension: Embedding dimension (default 384 for MiniLM)
        """
        self.embedding_model = embedding_model
        self.dimension = dimension
        self._index = None
        self._chunks: List[Chunk] = []
    
    @property
    def index(self):
        """Lazy initialize FAISS index."""
        if self._index is None:
            try:
                import faiss
            except ImportError:
                raise ImportError(
                    "faiss-cpu is required. "
                    "Install it with: pip install faiss-cpu"
                )
            
            # Use inner product (equivalent to cosine with normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index
    
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector store."""
        if not chunks:
            return
        
        # Get texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings if model provided
        if self.embedding_model:
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Update dimension if needed
            if embeddings.shape[1] != self.dimension:
                self.dimension = embeddings.shape[1]
                self._index = None  # Reset index with new dimension
            
            # Store embeddings in chunks
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
        else:
            # Expect embeddings to already be in chunks
            embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Add to index
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        
        # Store chunks for retrieval
        self._chunks.extend(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 4
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if not self._chunks:
            return []
        
        # Embed query
        if self.embedding_model:
            query_embedding = self.embedding_model.embed_query(query)
        else:
            raise ValueError("No embedding model provided for query embedding")
        
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search
        k = min(top_k, len(self._chunks))
        distances, indices = self.index.search(query_embedding, k)
        
        # Return chunks with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._chunks):
                results.append((self._chunks[idx], float(dist)))
        
        return results
    
    def save(self, path: str) -> None:
        """Save vector store to disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save chunks (without embeddings to save space)
        chunks_data = []
        for chunk in self._chunks:
            chunks_data.append({
                "content": chunk.content,
                "metadata": chunk.metadata
            })
        
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(chunks_data, f)
        
        # Save config
        config = {
            "dimension": self.dimension,
            "num_chunks": len(self._chunks)
        }
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(config, f)
    
    @classmethod
    def load(cls, path: str, embedding_model=None) -> "FAISSStore":
        """Load vector store from disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required")
        
        path = Path(path)
        
        # Load config
        with open(path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        # Create store
        store = cls(embedding_model=embedding_model, dimension=config["dimension"])
        
        # Load FAISS index
        store._index = faiss.read_index(str(path / "index.faiss"))
        
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
