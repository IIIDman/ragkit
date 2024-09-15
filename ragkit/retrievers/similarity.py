"""Retriever implementations."""

from typing import List, Tuple
from ..core import Chunk


class SimilarityRetriever:
    """
    Basic similarity retriever using vector store search.
    """
    
    def __init__(self, vectorstore, top_k: int = 4):
        """
        Initialize retriever.
        
        Args:
            vectorstore: Vector store with search method
            top_k: Number of chunks to retrieve
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Chunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant chunks
        """
        results = self.vectorstore.search(query, top_k=self.top_k)
        return [chunk for chunk, score in results]
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Chunk, float]]:
        """
        Retrieve chunks with similarity scores.
        
        Args:
            query: Query string
            
        Returns:
            List of (chunk, score) tuples
        """
        return self.vectorstore.search(query, top_k=self.top_k)


class MMRRetriever:
    """
    Maximum Marginal Relevance retriever.
    
    Balances relevance with diversity to reduce redundancy.
    """
    
    def __init__(
        self,
        vectorstore,
        top_k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ):
        """
        Initialize MMR retriever.
        
        Args:
            vectorstore: Vector store with search method
            top_k: Number of chunks to return
            fetch_k: Number of chunks to fetch before MMR
            lambda_mult: Balance between relevance (1) and diversity (0)
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
    
    def retrieve(self, query: str) -> List[Chunk]:
        """
        Retrieve diverse relevant chunks using MMR.
        """
        import numpy as np
        
        # Fetch more candidates than needed
        results = self.vectorstore.search(query, top_k=self.fetch_k)
        
        if not results:
            return []
        
        # Get query embedding
        query_embedding = self.vectorstore.embedding_model.embed_query(query)
        
        # Get candidate embeddings
        candidates = [chunk for chunk, score in results]
        candidate_embeddings = np.array([
            chunk.embedding if chunk.embedding is not None
            else self.vectorstore.embedding_model.embed_query(chunk.content)
            for chunk in candidates
        ])
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        for _ in range(min(self.top_k, len(candidates))):
            if not remaining_indices:
                break
            
            if not selected_indices:
                # First selection: most similar to query
                similarities = np.dot(candidate_embeddings[remaining_indices], query_embedding)
                best_idx = remaining_indices[np.argmax(similarities)]
            else:
                # MMR selection
                best_score = float("-inf")
                best_idx = remaining_indices[0]
                
                for idx in remaining_indices:
                    # Similarity to query
                    query_sim = np.dot(candidate_embeddings[idx], query_embedding)
                    
                    # Max similarity to already selected
                    selected_embs = candidate_embeddings[selected_indices]
                    max_selected_sim = np.max(np.dot(selected_embs, candidate_embeddings[idx]))
                    
                    # MMR score
                    mmr_score = self.lambda_mult * query_sim - (1 - self.lambda_mult) * max_selected_sim
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [candidates[i] for i in selected_indices]
