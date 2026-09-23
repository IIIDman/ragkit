"""Hybrid retriever: fuse ranked lists from several retrievers."""

from typing import Callable, Dict, List, Optional, Tuple

from ..core import Chunk


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).

    score(d) = sum over lists i of weight_i / (k + rank_i(d)), rank starting at 1.
    A doc missing from a list gets nothing from it. Only ranks are used, so
    lists with incomparable scores (cosine vs BM25) can be merged directly.

    Args:
        ranked_lists: One list of item keys per retriever, best first
        weights: Per-list weights (default all 1.0)
        k: Damping constant. Larger k flattens the gap between top ranks

    Returns:
        {key: fused score}
    """
    weights = weights or [1.0] * len(ranked_lists)
    fused: Dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, key in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
    return fused


def min_max_normalize(scores: List[float]) -> List[float]:
    """Rescale scores to [0, 1]. All-equal scores map to 1.0."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    span = high - low
    return [(s - low) / span if span > 0 else 1.0 for s in scores]


def min_max_fusion(
    scored_lists: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Weighted sum of min-max normalized scores.

    Each list's scores are rescaled to [0, 1] per query, then combined as
    sum_i weight_i * norm_score_i(d). A doc missing from a list counts as 0.

    Args:
        scored_lists: One list of (key, score) per retriever
        weights: Per-list weights (default all 1.0)

    Returns:
        {key: fused score}
    """
    weights = weights or [1.0] * len(scored_lists)
    fused: Dict[str, float] = {}
    for scored, weight in zip(scored_lists, weights):
        norms = min_max_normalize([score for _, score in scored])
        for (key, _), norm in zip(scored, norms):
            fused[key] = fused.get(key, 0.0) + weight * norm
    return fused


class HybridRetriever:
    """
    Combine several retrievers (e.g. dense + BM25) into one ranking.

    Each sub-retriever returns its own candidates (set their top_k to the
    candidate depth you want, e.g. 50-100). The lists are fused and the
    top_k best are returned.

    Example:
        dense = SimilarityRetriever(store, top_k=50)
        bm25 = BM25Retriever(chunks, top_k=50)
        hybrid = HybridRetriever([dense, bm25], top_k=5)
    """

    def __init__(
        self,
        retrievers: List,
        top_k: int = 4,
        weights: Optional[List[float]] = None,
        fusion: str = "minmax",
        rrf_k: int = 60,
        key: Optional[Callable[[Chunk], str]] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            retrievers: Retrievers with a retrieve_with_scores(query) method
            top_k: Number of chunks to return after fusion
            weights: Per-retriever weights (default equal)
            fusion: "minmax" (normalized score sum) or "rrf" (rank based).
                minmax scored higher on SciFact and NFCorpus, in line with
                Bruch et al. 2023; rrf ignores score scales entirely, which is
                safer when a retriever's scores are badly calibrated
            rrf_k: RRF damping constant
            key: Maps a chunk to the id used to match it across retrievers.
                Default is the chunk text, which works even when retrievers
                hold separate copies of the same chunks.
        """
        if fusion not in ("rrf", "minmax"):
            raise ValueError(f"Unknown fusion: {fusion}. Use 'rrf' or 'minmax'")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must have one entry per retriever")

        self.retrievers = retrievers
        self.top_k = top_k
        self.weights = weights
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.key = key or (lambda chunk: chunk.content)

    def retrieve_with_scores(self, query: str) -> List[Tuple[Chunk, float]]:
        """Retrieve fused results with fusion scores, best first."""
        chunks_by_key: Dict[str, Chunk] = {}
        scored_lists = []
        for retriever in self.retrievers:
            scored = []
            for chunk, score in retriever.retrieve_with_scores(query):
                key = self.key(chunk)
                chunks_by_key.setdefault(key, chunk)
                scored.append((key, score))
            scored_lists.append(scored)

        if self.fusion == "rrf":
            ranked_lists = [[key for key, _ in scored] for scored in scored_lists]
            fused = reciprocal_rank_fusion(ranked_lists, self.weights, self.rrf_k)
        else:
            fused = min_max_fusion(scored_lists, self.weights)

        best = sorted(fused.items(), key=lambda item: item[1], reverse=True)[: self.top_k]
        return [(chunks_by_key[key], score) for key, score in best]

    def retrieve(self, query: str) -> List[Chunk]:
        """Retrieve the top_k fused chunks."""
        return [chunk for chunk, _ in self.retrieve_with_scores(query)]
