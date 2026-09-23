"""BM25 keyword retriever."""

import math
import pickle
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core import Chunk

# Lucene / Elasticsearch default English stopwords
ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such "
    "that the their then there these they this to was will with".split()
)

_TOKEN_RE = re.compile(r"\w+")


def simple_tokenize(text: str) -> List[str]:
    """Lowercase, split on non-word characters, drop English stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOPWORDS]


@lru_cache(maxsize=1)
def _porter():
    import snowballstemmer

    return snowballstemmer.stemmer("porter")


def english_tokenize(text: str) -> List[str]:
    """simple_tokenize plus Porter stemming, so "mutations" matches "mutation"."""
    return _porter().stemWords(simple_tokenize(text))


class BM25Retriever:
    """
    Keyword retriever using Okapi BM25 over an inverted index.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    with idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)), the Lucene variant,
    which is always positive.
    """

    def __init__(
        self,
        chunks: Optional[List[Chunk]] = None,
        top_k: int = 4,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], List[str]] = english_tokenize,
    ):
        """
        Initialize retriever.

        Args:
            chunks: Chunks to index (more can be added later with add())
            top_k: Number of chunks to retrieve
            k1: Term frequency saturation. Higher = repeated terms keep adding score
            b: Length normalization, 0 = none, 1 = full
            tokenizer: Function mapping text to a list of terms
        """
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer

        self._chunks: List[Chunk] = []
        self._doc_tfs: List[Counter] = []
        self._doc_lens = np.zeros(0, dtype=np.float32)
        self._postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._idf: Dict[str, float] = {}
        self._dirty = False

        if chunks:
            self.add(chunks)

    def add(self, chunks: List[Chunk]) -> None:
        """Tokenize and queue chunks. The index is rebuilt on the next search."""
        if not chunks:
            return
        for chunk in chunks:
            self._doc_tfs.append(Counter(self.tokenizer(chunk.content)))
        self._chunks.extend(chunks)
        # Adding docs changes N, df and avgdl for every term, so the index is
        # rebuilt, but lazily: many add() calls in a row cost one rebuild
        self._dirty = True

    def _build_index(self) -> None:
        # Inverted index: term -> (doc indices, term counts in those docs)
        doc_ids: Dict[str, List[int]] = {}
        tfs: Dict[str, List[int]] = {}
        for i, counts in enumerate(self._doc_tfs):
            for term, tf in counts.items():
                doc_ids.setdefault(term, []).append(i)
                tfs.setdefault(term, []).append(tf)

        n = len(self._doc_tfs)
        self._doc_lens = np.array([sum(c.values()) for c in self._doc_tfs], dtype=np.float32)
        self._postings = {
            term: (np.array(doc_ids[term]), np.array(tfs[term], dtype=np.float32))
            for term in doc_ids
        }
        self._idf = {
            term: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, ids in doc_ids.items()
        }
        self._dirty = False

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every indexed chunk for the query."""
        scores = np.zeros(len(self._chunks), dtype=np.float32)
        if not self._chunks:
            return scores
        if self._dirty:
            self._build_index()

        avgdl = self._doc_lens.mean() or 1.0
        # Unique terms: repeating a word in the query does not boost it
        for term in set(self.tokenizer(query)):
            if term not in self._postings:
                continue
            ids, tf = self._postings[term]
            length_norm = 1 - self.b + self.b * self._doc_lens[ids] / avgdl
            scores[ids] += self._idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return scores

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve chunks with BM25 scores, best first.

        Chunks sharing no terms with the query are never returned.

        Args:
            query: Query string
            top_k: Override the retriever's top_k for this call
        """
        scores = self.score(query)
        matched = np.flatnonzero(scores > 0)
        if len(matched) == 0:
            return []

        k = min(top_k or self.top_k, len(matched))
        # argpartition finds the top k in O(n), then only those k get sorted
        top = matched[np.argpartition(-scores[matched], k - 1)[:k]]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Retrieve the top_k chunks for a query."""
        return [chunk for chunk, _ in self.retrieve_with_scores(query, top_k)]

    def save(self, path: str) -> None:
        """Save chunks and parameters. The index is rebuilt on load."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [{"content": c.content, "metadata": c.metadata} for c in self._chunks],
            "top_k": self.top_k,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(
        cls, path: str, tokenizer: Callable[[str], List[str]] = english_tokenize
    ) -> "BM25Retriever":
        """Load a saved retriever. Pass the same tokenizer that was used to build it."""
        with open(Path(path) / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        chunks = [Chunk(content=d["content"], metadata=d["metadata"]) for d in data["chunks"]]
        return cls(chunks, top_k=data["top_k"], k1=data["k1"], b=data["b"], tokenizer=tokenizer)

    def __len__(self) -> int:
        return len(self._chunks)
