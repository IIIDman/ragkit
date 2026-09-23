"""
Compare PDF parsing setups on questions whose answers sit in body text or tables.

Two two-column papers from arXiv (DPR, EMNLP 2020; ColBERT, SIGIR 2020),
17 questions written from the rendered pages: 9 answered in body text, 8
by a table cell. Retrieval is ragkit's default hybrid search, top 5.

- text hit: the answer string is in one of the top 5 chunks
- table hit: the value AND its row label are in the same top-5 chunk,
  i.e. the number arrives with enough context to be read correctly

Usage:
    pip install docling
    python examples/eval_pdf.py
"""

import html
import re
import time
import urllib.request
from pathlib import Path

from ragkit import (
    BM25Retriever,
    Chunk,
    DoclingLoader,
    FAISSStore,
    HybridRetriever,
    PDFLoader,
    RecursiveCharacterSplitter,
    SentenceTransformerEmbeddings,
    SimilarityRetriever,
)

CACHE = Path.home() / ".cache" / "ragkit" / "pdfs"

PAPERS = {
    "dpr": "https://arxiv.org/pdf/2004.04906v3",
    "colbert": "https://arxiv.org/pdf/2004.12832v2",
}

QUESTIONS = [
    # DPR, body text
    {"paper": "dpr", "q": "How many passages does the Wikipedia corpus contain after splitting articles into 100-word blocks?", "a": "21,015,324"},
    {"paper": "dpr", "q": "How many questions per second can DPR process with a FAISS index?", "a": "995.0"},
    {"paper": "dpr", "q": "How long does it take to compute dense embeddings for 21 million passages?", "a": "8.8 hours"},
    {"paper": "dpr", "q": "What top-20 accuracy does DPR trained only on Natural Questions get on WebQuestions and TREC?", "a": "69.9/86.3"},
    {"paper": "dpr", "q": "What batch size was used to train the main DPR model?", "a": "batch size of 128"},
    # DPR, tables
    {"paper": "dpr", "q": "How many test questions does the TriviaQA dataset have?", "a": "11,313", "row": "TriviaQA"},
    {"paper": "dpr", "q": "What is the top-100 retrieval accuracy of BM25 on CuratedTREC?", "a": "84.1", "row": "BM25"},
    {"paper": "dpr", "q": "What top-100 accuracy do gold negatives with 127 in-batch negatives reach on Natural Questions dev?", "a": "83.1", "row": "Gold"},
    # ColBERT, body text
    {"paper": "colbert", "q": "How many passages are in the MS MARCO ranking collection?", "a": "8.8M passages"},
    {"paper": "colbert", "q": "Which GPU was used to measure the latency of neural re-ranking models?", "a": "Tesla V100"},
    {"paper": "colbert", "q": "How many queries are in the TREC CAR 2017 test set?", "a": "2,254"},
    {"paper": "colbert", "q": "What batch size was used to fine-tune ColBERT?", "a": "batch size 32"},
    # ColBERT, tables
    {"paper": "colbert", "q": "What is the re-ranking latency in milliseconds of BERT-large on MS MARCO?", "a": "32,900", "row": "BERT"},
    {"paper": "colbert", "q": "What Recall@1000 does end-to-end ColBERT reach on MS MARCO?", "a": "96.8", "row": "end-to-end"},
    {"paper": "colbert", "q": "What is the Recall@50 of docTTTTTquery on MS MARCO?", "a": "75.6", "row": "docTTTTTquery"},
    {"paper": "colbert", "q": "What MAP does BM25 + ColBERT reach on TREC CAR?", "a": "31.3", "row": "ColBERT"},
    {"paper": "colbert", "q": "What MRR@10 does re-ranking with cosine similarity and 24-dimensional embeddings get?", "a": "33.9", "row": "Cosine"},
]


def normalize(text):
    text = html.unescape(text).replace("\u00ad", "")
    return re.sub(r"\s+", " ", text).lower()


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in PAPERS.items():
        paths[name] = CACHE / f"{name}.pdf"
        if not paths[name].exists():
            urllib.request.urlretrieve(url, paths[name])
    return paths


def evaluate(name, chunks, embeddings, parse_secs, k=5):
    store = FAISSStore(embeddings, embeddings.dimension)
    store.add(chunks)
    retriever = HybridRetriever(
        [SimilarityRetriever(store, top_k=50), BM25Retriever(chunks, top_k=50)], top_k=k
    )
    hits = {"text": 0, "table": 0}
    misses = []
    for q in QUESTIONS:
        answer = normalize(q["a"])
        top = [
            normalize(c.content)
            for c in retriever.retrieve(q["q"])
            if c.metadata["paper"] == q["paper"]
        ]
        if "row" in q:
            ok = any(answer in t and normalize(q["row"]) in t for t in top)
        else:
            ok = any(answer in t for t in top)
        hits["table" if "row" in q else "text"] += ok
        if not ok:
            misses.append(q["a"])
    n_text = sum("row" not in q for q in QUESTIONS)
    n_table = len(QUESTIONS) - n_text
    print(
        f"| {name} | {hits['text']}/{n_text} | {hits['table']}/{n_table} "
        f"| {len(chunks)} | {parse_secs:.1f} s | {', '.join(misses) or '-'} |",
        flush=True,
    )


def tag(chunks, paper):
    for c in chunks:
        c.metadata["paper"] = paper
    return chunks


def main():
    paths = download()
    embeddings = SentenceTransformerEmbeddings()
    splitter = RecursiveCharacterSplitter(chunk_size=512, chunk_overlap=50)

    print("| Setup | Text hits | Table hits (with row) | Chunks | Parse time | Missed |")
    print("|---|---|---|---|---|---|")

    for label, clean in (("pypdf, raw", False), ("pypdf, cleaned (default)", True)):
        start = time.perf_counter()
        chunks = []
        for paper, path in paths.items():
            chunks += tag(splitter.split(PDFLoader(clean=clean).load(str(path))), paper)
        evaluate(label + " + 512-char chunks", chunks, embeddings, time.perf_counter() - start)

    markdown = DoclingLoader(chunking=False)
    structured = DoclingLoader()
    structured._converter = markdown.converter  # share the loaded layout models

    start = time.perf_counter()
    chunks = []
    for paper, path in paths.items():
        chunks += tag(splitter.split(markdown.load(str(path))), paper)
    evaluate("docling Markdown + 512-char chunks", chunks, embeddings, time.perf_counter() - start)

    start = time.perf_counter()
    chunks = []
    for paper, path in paths.items():
        docs = structured.load(str(path))
        chunks += tag([Chunk(content=d.content, metadata=dict(d.metadata)) for d in docs], paper)
    evaluate("docling structure-aware chunks (default)", chunks, embeddings, time.perf_counter() - start)


if __name__ == "__main__":
    main()
