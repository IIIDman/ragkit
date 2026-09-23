"""Tests for rank fusion and the hybrid retriever."""

import pytest

from ragkit import Chunk, HybridRetriever
from ragkit.retrievers import min_max_fusion, reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_hand_computed_scores(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert fused["c"] == pytest.approx(1 / 62)

    def test_agreement_beats_single_top_rank(self):
        # "b" is 2nd in both lists, "a" is 1st in one list only
        fused = reciprocal_rank_fusion([["a", "b"], ["x", "b"]])
        assert fused["b"] > fused["a"]

    def test_weights_scale_contributions(self):
        fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0])
        assert fused["a"] == pytest.approx(2 * fused["b"])


class TestMinMaxFusion:
    def test_scores_are_normalized_per_list(self):
        # Raw scales differ wildly (BM25 ~ 10s, cosine ~ 0.x) but normalize to [0, 1]
        fused = min_max_fusion([[("a", 30.0), ("b", 10.0)], [("b", 0.9), ("a", 0.5)]])
        assert fused["a"] == pytest.approx(1.0 + 0.0)
        assert fused["b"] == pytest.approx(0.0 + 1.0)

    def test_missing_doc_counts_as_zero(self):
        fused = min_max_fusion([[("a", 5.0), ("b", 1.0)], [("c", 0.7)]])
        assert fused["b"] == pytest.approx(0.0)
        assert fused["c"] == pytest.approx(1.0)


class FixedRetriever:
    """Returns a fixed ranked list of (chunk, score)."""

    def __init__(self, items):
        self.items = items

    def retrieve_with_scores(self, query):
        return self.items


def chunk(text):
    return Chunk(content=text, metadata={"doc_id": text})


class TestHybridRetriever:
    def test_fuses_and_truncates(self):
        dense = FixedRetriever([(chunk("a"), 0.9), (chunk("b"), 0.8), (chunk("c"), 0.1)])
        sparse = FixedRetriever([(chunk("b"), 12.0), (chunk("d"), 3.0)])
        hybrid = HybridRetriever([dense, sparse], top_k=2)
        results = hybrid.retrieve("q")
        assert [c.content for c in results] == ["b", "a"]

    def test_matches_separate_copies_by_content(self):
        # Different Chunk objects with the same text must be treated as one doc
        dense = FixedRetriever([(chunk("same"), 0.9)])
        sparse = FixedRetriever([(chunk("same"), 5.0)])
        results = HybridRetriever([dense, sparse], top_k=5, fusion="rrf").retrieve_with_scores("q")
        assert len(results) == 1
        assert results[0][1] == pytest.approx(2 / 61)

    def test_custom_key(self):
        a1 = Chunk(content="text v1", metadata={"doc_id": "x"})
        a2 = Chunk(content="text v2", metadata={"doc_id": "x"})
        hybrid = HybridRetriever(
            [FixedRetriever([(a1, 1.0)]), FixedRetriever([(a2, 1.0)])],
            key=lambda c: c.metadata["doc_id"],
        )
        assert len(hybrid.retrieve("q")) == 1

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            HybridRetriever([FixedRetriever([])], fusion="max")
        with pytest.raises(ValueError):
            HybridRetriever([FixedRetriever([]), FixedRetriever([])], weights=[1.0])
