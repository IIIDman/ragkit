"""Loaders for BEIR retrieval benchmarks."""

import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..core import Chunk

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEFAULT_CACHE = Path.home() / ".cache" / "ragkit" / "beir"


@dataclass
class BeirDataset:
    """A BEIR dataset: corpus, queries and relevance labels (qrels)."""

    name: str
    corpus: Dict[str, Dict[str, str]]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]

    def to_chunks(self) -> List[Chunk]:
        """
        One chunk per corpus doc, with the doc id in metadata["doc_id"].

        Title and text are joined, as in the BEIR reference setup.
        """
        chunks = []
        for doc_id, doc in self.corpus.items():
            content = f"{doc['title']}\n{doc['text']}" if doc["title"] else doc["text"]
            chunks.append(Chunk(content=content, metadata={"doc_id": doc_id, "source": self.name}))
        return chunks


def load_beir(name: str, split: str = "test", cache_dir: Optional[str] = None) -> BeirDataset:
    """
    Download (once) and load a BEIR dataset.

    Only queries that have relevance labels in the split are kept.

    Args:
        name: Dataset name, e.g. "scifact", "nfcorpus", "fiqa"
        split: Qrels split, usually "test"
        cache_dir: Where to store downloads (default ~/.cache/ragkit/beir)

    Returns:
        BeirDataset
    """
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    data_dir = root / name
    if not data_dir.exists():
        _download(name, root)

    corpus = {}
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = {"title": row.get("title", ""), "text": row["text"]}

    qrels: Dict[str, Dict[str, int]] = {}
    with open(data_dir / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for query_id, doc_id, score in reader:
            qrels.setdefault(query_id, {})[doc_id] = int(score)

    queries = {}
    with open(data_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]

    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def _download(name: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / f"{name}.zip"
    print(f"Downloading BEIR {name}...")
    urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink()
