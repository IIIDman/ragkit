"""Two-stage retrieval: fetch candidates, then rerank them."""

from typing import List, Tuple

from ..core import Chunk
from .hybrid import min_max_normalize


class RerankRetriever:
    """
    Wrap a first-stage retriever with a reranker.

    The base retriever's top_k is the candidate depth: that many chunks are
    rescored by the reranker, and the best top_k are returned. A reranker can
    only reorder what the first stage found, so recall at the candidate depth
    is its ceiling.

    With alpha < 1 the final score blends both stages:
        alpha * norm(reranker score) + (1 - alpha) * norm(first-stage score)
    Useful when the reranker was trained on a different domain and is not
    reliable on its own. Tune alpha on held-out queries, not the test set.

    Example:
        hybrid = HybridRetriever([dense, bm25], top_k=20)
        retriever = RerankRetriever(hybrid, CrossEncoderReranker(), top_k=5)
    """

    def __init__(self, base_retriever, reranker, top_k: int = 4, alpha: float = 1.0):
        """
        Initialize retriever.

        Args:
            base_retriever: Retriever with retrieve_with_scores(query)
            reranker: Object with score(query, chunks) -> [float]
            top_k: Number of chunks to return after reranking
            alpha: Weight of the reranker score, 1.0 = reranker only
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.top_k = top_k
        self.alpha = alpha

    def retrieve_with_scores(self, query: str) -> List[Tuple[Chunk, float]]:
        """Retrieve candidates, rescore, return the top_k with final scores."""
        candidates = self.base_retriever.retrieve_with_scores(query)
        if not candidates:
            return []
        chunks = [chunk for chunk, _ in candidates]
        scores = self.reranker.score(query, chunks)

        if self.alpha < 1.0:
            rerank_norm = min_max_normalize(scores)
            base_norm = min_max_normalize([score for _, score in candidates])
            scores = [
                self.alpha * r + (1 - self.alpha) * b for r, b in zip(rerank_norm, base_norm)
            ]

        ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
        return ranked[: self.top_k]

    def retrieve(self, query: str) -> List[Chunk]:
        """Retrieve the top_k reranked chunks."""
        return [chunk for chunk, _ in self.retrieve_with_scores(query)]
