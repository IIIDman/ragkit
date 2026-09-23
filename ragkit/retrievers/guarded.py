"""Retriever wrapper that drops chunks flagged by a guard."""

import logging
from typing import List, Tuple

from ..core import Chunk

logger = logging.getLogger(__name__)


class GuardedRetriever:
    """
    Screen retrieved chunks with a guard and drop the flagged ones.

    Dropped chunks are logged with their score and kept on
    self.last_flagged for inspection.

    Example:
        retriever = GuardedRetriever(hybrid, InjectionGuard())
    """

    def __init__(self, base_retriever, guard):
        """
        Initialize retriever.

        Args:
            base_retriever: Retriever with retrieve_with_scores(query)
            guard: Object with filter(chunks) -> (safe, flagged)
        """
        self.base_retriever = base_retriever
        self.guard = guard
        self.last_flagged: List[Chunk] = []

    def retrieve_with_scores(self, query: str) -> List[Tuple[Chunk, float]]:
        """Retrieve, screen, return the safe chunks with their original scores."""
        results = self.base_retriever.retrieve_with_scores(query)
        safe, flagged = self.guard.filter([chunk for chunk, _ in results])
        self.last_flagged = flagged
        for chunk in flagged:
            logger.warning(
                "Dropped chunk from %s (injection score %.2f)",
                chunk.metadata.get("source", "unknown"),
                chunk.metadata.get("injection_score", float("nan")),
            )
        safe_ids = {id(chunk) for chunk in safe}
        return [(chunk, score) for chunk, score in results if id(chunk) in safe_ids]

    def retrieve(self, query: str) -> List[Chunk]:
        """Retrieve the chunks that pass the guard."""
        return [chunk for chunk, _ in self.retrieve_with_scores(query)]
