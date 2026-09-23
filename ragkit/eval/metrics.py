"""Retrieval metrics.

Each function scores a single query: a ranked list of retrieved doc ids
against a dict of relevance labels {doc_id: grade}. Grades > 0 count as
relevant. Average over queries to get the dataset score.
"""

import math
from typing import Dict, List


def recall_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """
    Fraction of relevant docs that appear in the top k.

    Args:
        retrieved: Ranked doc ids, best first
        relevant: Relevance labels for the query
        k: Cutoff

    Returns:
        Recall in [0, 1]
    """
    positives = {doc_id for doc_id, grade in relevant.items() if grade > 0}
    if not positives:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in positives)
    return hits / len(positives)


def mrr_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """
    Reciprocal rank of the first relevant doc in the top k (0 if none).

    Args:
        retrieved: Ranked doc ids, best first
        relevant: Relevance labels for the query
        k: Cutoff

    Returns:
        Reciprocal rank in [0, 1]
    """
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int) -> float:
    """
    Normalized discounted cumulative gain at k.

    Gain is the relevance grade, discounted by log2(rank + 1), then divided
    by the best possible score for this query (ideal ordering). Same
    definition as pytrec_eval, which the BEIR benchmark uses.

    Args:
        retrieved: Ranked doc ids, best first
        relevant: Relevance labels for the query
        k: Cutoff

    Returns:
        nDCG in [0, 1]
    """
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
    )
    ideal_grades = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
