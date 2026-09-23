"""Cross-encoder reranker."""

from typing import List, Optional, Tuple

from ..core import Chunk


class CrossEncoderReranker:
    """
    Rescore (query, chunk) pairs with a cross-encoder.

    A bi-encoder (the embedding model) encodes query and chunk separately and
    compares vectors. A cross-encoder reads them together in one forward pass,
    so attention can match query words against chunk words directly. Much more
    accurate, but one model call per pair, so it only runs on a short
    candidate list from a first-stage retriever.

    Default model: cross-encoder/ms-marco-MiniLM-L6-v2 (22M params, fast on CPU)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize reranker. The model loads on first use.

        Args:
            model_name: Any sentence-transformers CrossEncoder model
            device: "cpu", "cuda", "mps", or None for auto
            batch_size: Pairs per forward pass
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with: pip install sentence-transformers"
                )
            self._model = CrossEncoder(self.model_name, device=self._device)
        return self._model

    def score(self, query: str, chunks: List[Chunk]) -> List[float]:
        """Relevance score for each chunk, in input order."""
        if not chunks:
            return []
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Tuple[Chunk, float]]:
        """Chunks sorted by cross-encoder score, best first."""
        scores = self.score(query, chunks)
        return sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
