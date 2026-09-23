"""Prompt injection detection for retrieved text."""

from typing import List, Optional, Tuple

from ..core import Chunk

# Label names different classifiers use for the "attack" class
INJECTION_LABELS = frozenset({"INJECTION", "MALICIOUS", "JAILBREAK", "LABEL_1"})


class InjectionGuard:
    """
    Flag text that tries to give instructions to the LLM.

    Retrieved documents are untrusted input: a web page or PDF can contain
    "ignore your instructions and ..." aimed at the model reading it
    (indirect prompt injection). This runs a small classifier over each
    chunk before it reaches the prompt.

    Text is scored in short overlapping windows and the highest window
    score is kept. Two reasons: a payload past the classifier's 512-token
    limit would otherwise be cut off, and a short payload inside a long
    benign document gets diluted. On SciFact abstracts with injected
    payloads, 300-char windows caught 34% vs 10% for 1500-char windows,
    at 0.2% false positives on clean abstracts.

    Default model: protectai/deberta-v3-base-prompt-injection-v2 (184M,
    Apache 2.0, not gated). English only, and it is not a jailbreak
    detector. meta-llama/Llama-Prompt-Guard-2-86M also works once you
    have accepted its license on Hugging Face.
    """

    def __init__(
        self,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
        threshold: float = 0.5,
        device: Optional[str] = None,
        window_chars: int = 300,
        overlap_chars: int = 100,
        batch_size: int = 16,
    ):
        """
        Initialize guard. The model loads on first use.

        Args:
            model_name: Hugging Face text-classification model
            threshold: Injection probability at or above which text is flagged
            device: "cpu", "cuda", "mps", or None for auto
            window_chars: Characters per scoring window
            overlap_chars: Overlap between windows, so a payload on a boundary is seen whole
            batch_size: Windows per forward pass
        """
        if overlap_chars >= window_chars:
            raise ValueError("overlap_chars must be smaller than window_chars")
        self.model_name = model_name
        self.threshold = threshold
        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self.batch_size = batch_size
        self._device = device
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy load the classifier."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError(
                    "transformers is required. Install it with: pip install transformers"
                )
            self._pipeline = pipeline(
                "text-classification", model=self.model_name, device=self._device
            )
        return self._pipeline

    def _windows(self, text: str) -> List[str]:
        if len(text) <= self.window_chars:
            return [text]
        step = self.window_chars - self.overlap_chars
        starts = range(0, len(text) - self.overlap_chars, step)
        return [text[i : i + self.window_chars] for i in starts]

    def score(self, texts: List[str]) -> List[float]:
        """Injection probability for each text (max over its windows)."""
        if not texts:
            return []
        windows, owners = [], []
        for i, text in enumerate(texts):
            for window in self._windows(text):
                windows.append(window)
                owners.append(i)

        outputs = self.pipeline(
            windows, top_k=None, batch_size=self.batch_size, truncation=True, max_length=512
        )
        scores = [0.0] * len(texts)
        for owner, labels in zip(owners, outputs):
            p = sum(item["score"] for item in labels if item["label"].upper() in INJECTION_LABELS)
            scores[owner] = max(scores[owner], p)
        return scores

    def check(self, chunks: List[Chunk]) -> List[Tuple[Chunk, float, bool]]:
        """(chunk, injection score, flagged) for each chunk, in input order."""
        scores = self.score([chunk.content for chunk in chunks])
        return [(c, s, s >= self.threshold) for c, s in zip(chunks, scores)]

    def filter(self, chunks: List[Chunk]) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Split chunks into (safe, flagged).

        Every chunk gets metadata["injection_score"] so callers can log or
        inspect decisions.
        """
        safe, flagged = [], []
        for chunk, score, is_flagged in self.check(chunks):
            chunk.metadata["injection_score"] = score
            (flagged if is_flagged else safe).append(chunk)
        return safe, flagged
