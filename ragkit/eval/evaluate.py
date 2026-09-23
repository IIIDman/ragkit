"""Run a retriever over a labelled query set and aggregate metrics."""

import time
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


@dataclass
class EvalResult:
    """Mean metrics over all queries plus per-query latency percentiles."""

    name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    num_queries: int = 0

    def as_row(self) -> Dict[str, float]:
        return {**self.metrics, "p50 ms": self.latency_p50_ms, "p95 ms": self.latency_p95_ms}


def evaluate(
    retriever,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    k_values: Sequence[int] = (10, 100),
    id_key: str = "doc_id",
    name: str = "retriever",
) -> EvalResult:
    """
    Evaluate a retriever on labelled queries.

    The retriever must have retrieve(query) -> List[Chunk] and return at
    least max(k_values) chunks for the deeper metrics to be meaningful.

    Args:
        retriever: Object with a retrieve(query) method
        queries: {query_id: query_text}
        qrels: {query_id: {doc_id: grade}}
        k_values: Cutoffs for recall and nDCG (MRR is always @10)
        id_key: Chunk metadata key holding the doc id
        name: Label for the result row

    Returns:
        EvalResult
    """
    sums = {}
    latencies = []

    for query_id, query in queries.items():
        start = time.perf_counter()
        chunks = retriever.retrieve(query)
        latencies.append((time.perf_counter() - start) * 1000)

        # A doc split into several chunks should count once, at its best rank
        retrieved = list(dict.fromkeys(c.metadata[id_key] for c in chunks))
        relevant = qrels.get(query_id, {})

        scores = {"MRR@10": mrr_at_k(retrieved, relevant, 10)}
        for k in k_values:
            scores[f"nDCG@{k}"] = ndcg_at_k(retrieved, relevant, k)
            scores[f"Recall@{k}"] = recall_at_k(retrieved, relevant, k)
        for key, value in scores.items():
            sums[key] = sums.get(key, 0.0) + value

    n = len(queries)
    order = sorted(sums, key=lambda key: (key.split("@")[0], int(key.split("@")[1])))
    return EvalResult(
        name=name,
        metrics={key: sums[key] / n for key in order},
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        num_queries=n,
    )


def format_table(results: Sequence[EvalResult]) -> str:
    """Render results as a Markdown table."""
    columns = list(results[0].as_row())
    lines = [
        "| Retriever | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for r in results:
        row = r.as_row()
        cells = [f"{row[c]:.1f}" if "ms" in c else f"{row[c]:.3f}" for c in columns]
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
