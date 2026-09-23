"""
Benchmark RAGKit retrievers on BEIR SciFact.

SciFact: 5,183 paper abstracts, 300 test claims, each labelled with the
abstracts that support or refute it. The corpus downloads on first run
(~3MB) to ~/.cache/ragkit/beir.

Usage:
    python examples/eval_scifact.py
    python examples/eval_scifact.py --dataset nfcorpus
"""

import argparse

from ragkit import FAISSStore, SentenceTransformerEmbeddings, SimilarityRetriever
from ragkit.eval import evaluate, format_table, load_beir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--depth", type=int, default=100, help="Chunks retrieved per query")
    args = parser.parse_args()

    data = load_beir(args.dataset)
    print(f"{args.dataset}: {len(data.corpus)} docs, {len(data.queries)} queries")

    chunks = data.to_chunks()
    embeddings = SentenceTransformerEmbeddings(model_name=args.embedding_model)
    store = FAISSStore(embedding_model=embeddings, dimension=embeddings.dimension)
    print("Embedding corpus...")
    store.add(chunks)

    results = [
        evaluate(
            SimilarityRetriever(store, top_k=args.depth),
            data.queries,
            data.qrels,
            name=f"dense ({args.embedding_model})",
        ),
    ]

    print()
    print(format_table(results))


if __name__ == "__main__":
    main()
