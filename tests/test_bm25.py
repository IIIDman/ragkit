"""Tests for the BM25 retriever."""

import math

import pytest

from ragkit import BM25Retriever, Chunk
from ragkit.retrievers import english_tokenize, simple_tokenize


def make_chunks(*texts):
    return [Chunk(content=t, metadata={"doc_id": f"d{i}"}) for i, t in enumerate(texts)]


class TestTokenizers:
    def test_simple_tokenize_drops_stopwords_and_punctuation(self):
        assert simple_tokenize("The cat, and THE dog!") == ["cat", "dog"]

    def test_english_tokenize_stems(self):
        assert english_tokenize("mutations") == english_tokenize("mutation")


class TestBM25:
    def test_score_matches_formula(self):
        # "apple" appears in 1 of 3 docs; all docs have 2 terms, so length_norm = 1
        retriever = BM25Retriever(
            make_chunks("apple pie", "banana split", "cherry tart"),
            tokenizer=simple_tokenize,
        )
        idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
        tf_part = 1 * (1.2 + 1) / (1 + 1.2 * 1)
        assert retriever.score("apple")[0] == pytest.approx(idf * tf_part, rel=1e-5)

    def test_rare_term_outweighs_common_term(self):
        retriever = BM25Retriever(
            make_chunks("common word here", "common word zebra", "common thing"),
            top_k=3,
        )
        results = retriever.retrieve("common zebra")
        assert results[0].metadata["doc_id"] == "d1"

    def test_length_normalization_prefers_shorter_doc(self):
        chunks = make_chunks("python", "python " + "filler " * 20)
        top = BM25Retriever(chunks, top_k=2).retrieve("python")
        assert top[0].metadata["doc_id"] == "d0"

        # With b=0 length is ignored and both docs score the same
        scores = BM25Retriever(chunks, b=0.0).score("python")
        assert scores[0] == pytest.approx(scores[1])

    def test_term_frequency_saturates(self):
        retriever = BM25Retriever(make_chunks("x", "x x", "x x x x x x x x x x", "y"), b=0.0)
        s1, s2, s10, _ = retriever.score("x")
        assert s1 < s2 < s10
        # Going from 1 to 2 occurrences adds more than going from 2 to 10
        assert s2 - s1 > (s10 - s2) / 8
        assert s10 < s1 * (1.2 + 1)

    def test_no_overlap_returns_nothing(self):
        retriever = BM25Retriever(make_chunks("apple pie"))
        assert retriever.retrieve("quantum physics") == []

    def test_top_k_and_ordering(self):
        retriever = BM25Retriever(
            make_chunks("cat", "cat cat dog", "dog", "cat dog bird"), top_k=2
        )
        results = retriever.retrieve_with_scores("cat")
        assert len(results) == 2
        assert results[0][1] >= results[1][1]

    def test_add_updates_statistics(self):
        retriever = BM25Retriever(make_chunks("apple"))
        before = retriever.score("apple")[0]
        retriever.add(make_chunks("apple", "banana", "cherry"))
        # apple is now in 2 of 4 docs instead of 1 of 1, so its idf changes
        assert retriever.score("apple")[0] != pytest.approx(before)
        assert len(retriever) == 4

    def test_save_load_roundtrip(self, tmp_path):
        retriever = BM25Retriever(make_chunks("apple pie", "banana split"), top_k=1, k1=1.5)
        retriever.save(str(tmp_path / "bm25"))
        loaded = BM25Retriever.load(str(tmp_path / "bm25"))
        assert loaded.k1 == 1.5
        assert loaded.retrieve("banana")[0].metadata["doc_id"] == "d1"
