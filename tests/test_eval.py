"""Tests for retrieval metrics and the evaluation loop."""

import math

import numpy as np
import pytest

from ragkit import Chunk, SimilarityRetriever, SimpleStore
from ragkit.eval import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


class TestMetrics:
    def test_recall_counts_hits_over_all_relevant(self):
        relevant = {"a": 1, "b": 1, "c": 1}
        assert recall_at_k(["a", "x", "b"], relevant, 3) == pytest.approx(2 / 3)
        assert recall_at_k(["a", "x", "b"], relevant, 1) == pytest.approx(1 / 3)

    def test_recall_ignores_zero_grades(self):
        assert recall_at_k(["a"], {"a": 1, "b": 0}, 10) == 1.0

    def test_mrr_uses_first_relevant_rank(self):
        relevant = {"b": 1, "c": 1}
        assert mrr_at_k(["a", "b", "c"], relevant, 10) == 0.5
        assert mrr_at_k(["a", "b", "c"], relevant, 1) == 0.0

    def test_ndcg_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b": 1}, 10) == pytest.approx(1.0)

    def test_ndcg_hand_computed(self):
        # One relevant doc at rank 2: DCG = 1/log2(3), ideal DCG = 1/log2(2) = 1
        assert ndcg_at_k(["x", "a"], {"a": 1}, 10) == pytest.approx(1 / math.log2(3))

    def test_ndcg_graded_relevance(self):
        # Grade 2 doc ranked below grade 1 doc scores less than the ideal order
        relevant = {"a": 2, "b": 1}
        swapped = ndcg_at_k(["b", "a"], relevant, 10)
        expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
        assert swapped == pytest.approx(expected)


class KeywordEmbeddings:
    """Deterministic toy embedder: one dimension per vocabulary word."""

    VOCAB = ["cat", "dog", "fish", "bird"]

    def embed(self, texts):
        vectors = np.array(
            [[float(word in text.lower()) for word in self.VOCAB] for text in texts]
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def embed_query(self, text):
        return self.embed([text])[0]

    def embed_documents(self, texts):
        return self.embed(texts)


class TestEvaluate:
    def test_end_to_end_search_and_scoring(self):
        store = SimpleStore(embedding_model=KeywordEmbeddings())
        store.add([
            Chunk(content="All about cats", metadata={"doc_id": "d1"}),
            Chunk(content="Dogs and a dog park", metadata={"doc_id": "d2"}),
            Chunk(content="Fish tanks", metadata={"doc_id": "d3"}),
        ])
        retriever = SimilarityRetriever(store, top_k=3)

        assert retriever.retrieve("fish food")[0].metadata["doc_id"] == "d3"

        result = evaluate(
            retriever,
            queries={"q1": "cat toys", "q2": "dog walking"},
            qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
            k_values=(1, 3),
        )
        assert result.metrics["MRR@10"] == 1.0
        assert result.metrics["nDCG@1"] == 1.0
        assert result.metrics["Recall@3"] == 1.0
        assert result.num_queries == 2

    def test_chunks_of_same_doc_count_once(self):
        class FixedRetriever:
            def retrieve(self, query):
                return [
                    Chunk(content="", metadata={"doc_id": "x"}),
                    Chunk(content="", metadata={"doc_id": "x"}),
                    Chunk(content="", metadata={"doc_id": "a"}),
                ]

        result = evaluate(FixedRetriever(), {"q": "..."}, {"q": {"a": 1}}, k_values=(2,))
        # "a" is the second unique doc, so it is inside the top 2
        assert result.metrics["Recall@2"] == 1.0
