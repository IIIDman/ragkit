"""Tests for the reranking stage."""

import pytest

from ragkit import Chunk, CrossEncoderReranker, RerankRetriever


class ListRetriever:
    """First stage returning chunks with descending scores 1.0, 0.9, ..."""

    def __init__(self, texts):
        self.chunks = [Chunk(content=t) for t in texts]

    def retrieve_with_scores(self, query):
        return [(c, 1.0 - 0.1 * i) for i, c in enumerate(self.chunks)]


class LengthReranker:
    """Toy reranker: longer chunk = more relevant."""

    def score(self, query, chunks):
        return [float(len(c.content)) for c in chunks]


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the text."""

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return [sum(w in text for w in query.split()) for query, text in pairs]


class TestRerankRetriever:
    def test_reorders_and_truncates(self):
        retriever = RerankRetriever(ListRetriever(["a", "aaa", "aa"]), LengthReranker(), top_k=2)
        assert [c.content for c in retriever.retrieve("q")] == ["aaa", "aa"]

    def test_returns_reranker_scores(self):
        retriever = RerankRetriever(ListRetriever(["a", "aaa"]), LengthReranker(), top_k=5)
        assert [s for _, s in retriever.retrieve_with_scores("q")] == [3.0, 1.0]

    def test_alpha_blends_both_stages(self):
        # First stage prefers "a" (score 1.0), reranker prefers "aaa"
        base = ListRetriever(["a", "aa", "aaa"])
        assert RerankRetriever(base, LengthReranker(), top_k=1).retrieve("q")[0].content == "aaa"
        assert RerankRetriever(base, LengthReranker(), top_k=1, alpha=0.0).retrieve("q")[0].content == "a"
        # alpha=0.5: a = 0.5*0 + 0.5*1 = 0.5, aa = 0.5*0.5 + 0.5*0.5 = 0.5, aaa = 0.5*1 + 0 = 0.5
        blended = RerankRetriever(base, LengthReranker(), top_k=3, alpha=0.5)
        assert all(s == pytest.approx(0.5) for _, s in blended.retrieve_with_scores("q"))

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            RerankRetriever(ListRetriever([]), LengthReranker(), alpha=1.5)

    def test_empty_candidates(self):
        retriever = RerankRetriever(ListRetriever([]), CrossEncoderReranker(), top_k=5)
        assert retriever.retrieve("q") == []


class TestCrossEncoderReranker:
    def test_sorts_by_model_score(self):
        reranker = CrossEncoderReranker()
        reranker._model = FakeCrossEncoder()
        chunks = [Chunk(content="nothing here"), Chunk(content="cats and dogs"), Chunk(content="cats")]
        ranked = reranker.rerank("cats dogs", chunks)
        assert [c.content for c, _ in ranked] == ["cats and dogs", "cats", "nothing here"]
        assert [s for _, s in ranked] == [2.0, 1.0, 0.0]

    def test_model_loads_lazily(self):
        reranker = CrossEncoderReranker(model_name="does-not-exist")
        assert reranker._model is None
