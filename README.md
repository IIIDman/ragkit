# RAGKit

A lightweight RAG framework for building document Q&A applications locally.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAGKit lets you build "chat with your documents" apps without needing cloud services or complex infrastructure. It's designed to run on regular hardware and get you started quickly.

## Features

- Simple API - basic usage is around 5 lines of code
- Runs locally, no API keys needed for the default setup
- Supports PDF, text, and Markdown files
- Can use local models (HuggingFace, Ollama) or cloud APIs (OpenAI)
- Hybrid retrieval: dense vectors (FAISS) + BM25 keyword search, fused per query
- Optional cross-encoder reranking
- Prompt injection defenses: a classifier that screens chunks, and randomly named delimiters around sources in the prompt
- Optional layout-aware parsing with [Docling](https://github.com/docling-project/docling): tables, multi-column pages, DOCX/PPTX/XLSX/HTML, OCR for scans
- Evaluation harness on BEIR benchmarks, so retrieval changes come with numbers
- High-level API for quick prototyping, lower-level components if you need more control

## Installation

```bash
pip install "git+https://github.com/IIIDman/ragkit"

# or with PDF support
pip install "ragkit[pdf] @ git+https://github.com/IIIDman/ragkit"

# or everything
pip install "ragkit[all] @ git+https://github.com/IIIDman/ragkit"
```

## Basic Usage

```python
from ragkit import RAGKit

# Initialize - this will download the embedding model on first run
rag = RAGKit()

# Add documents
rag.add_document("research_paper.pdf")
rag.add_document("notes.txt")

# Ask questions
answer = rag.query("What are the main findings?")
print(answer.text)
print(answer.sources)
```

RAGKit handles chunking, embedding, retrieval, and generation automatically.

## How it works

RAGKit implements Retrieval-Augmented Generation:

```
                 ┌─> dense search (FAISS) ─┐
Query ───────────┤                         ├─> fuse ─> [rerank] ─> LLM generates answer
                 └─> BM25 keyword search ──┘
```

The basic flow:
1. Documents get split into chunks, optionally screened for prompt injection, then embedded and added to a keyword index
2. A question is searched both ways: by meaning (embeddings) and by exact terms (BM25)
3. The two ranked lists are merged into one (min-max score fusion by default, RRF as an option)
4. Optionally a cross-encoder reranks the top candidates
5. The best chunks go to the LLM, each wrapped in delimiters it is told not to take instructions from

## Configuration

### LLM backends

```python
# Local HuggingFace model (default)
rag = RAGKit(llm_backend="huggingface")

# Ollama - needs Ollama running locally
rag = RAGKit(llm_backend="ollama", llm="llama3.2")

# OpenAI - needs OPENAI_API_KEY environment variable
rag = RAGKit(llm_backend="openai", llm="gpt-4")
```

### Other settings

```python
rag = RAGKit(
    embedding_model="all-mpnet-base-v2",  # different embedding model
    chunk_size=1000,
    chunk_overlap=100,
    top_k=5,  # number of chunks to retrieve
)
```

### Adding documents

```python
# single file
rag.add_document("paper.pdf")

# multiple
rag.add_documents(["doc1.pdf", "doc2.txt", "notes.md"])

# whole directory
rag.add_directory("./documents/", glob="**/*.pdf")

# or just raw text
rag.add_text("Some important information...", metadata={"source": "manual entry"})
```

### Retrieval

```python
rag = RAGKit(retrieval="hybrid")  # default: dense + BM25
rag = RAGKit(retrieval="dense")   # vectors only (v0.1 behaviour)
rag = RAGKit(retrieval="bm25")    # keywords only, no GPU needed at query time

# rerank the top 20 candidates with a cross-encoder
rag = RAGKit(reranker="BAAI/bge-reranker-v2-m3", rerank_depth=20)
```

Rerankers are off by default. They help a lot on web-style questions, but the popular MS MARCO models made results worse on scientific text in our tests (see below), so measure on your own data first.

### Prompt injection guard

```python
rag = RAGKit(guard=True)
rag.add_directory("./downloaded_pages/")
print(rag.flagged_chunks)  # chunks that were skipped, with metadata["injection_score"]
```

Retrieved documents are untrusted input. A page can contain text like "assistant, ignore your instructions and...", aimed at the LLM that reads it. With `guard=True`, each chunk is scored by a classifier when it is added and flagged chunks are not indexed. Independently, the QA prompt wraps every source in tags with a random name per request (`<source-9f2c1a7b>`) and tells the model the contents are data. A document can't close a tag whose name it can't predict.

The classifier is a speed bump, not a wall, see the numbers below. If your LLM can call tools, gate those actions in code.

### Parsing PDFs and office files

```python
rag = RAGKit()                  # pypdf: fast, fine for most born-digital PDFs
rag = RAGKit(parser="docling")  # pip install docling
rag.add_document("report.docx")
```

With `parser="docling"`, files go through Docling's layout models and are chunked along the document structure: each chunk carries its section headings, and table rows are written out as "row, column = value", so a number doesn't lose its labels. It also handles DOCX, PPTX, XLSX and HTML. OCR is off by default (it made parsing about 10x slower); for scanned PDFs use `DoclingLoader(ocr=True)` directly.

### Saving and loading

```python
rag.save("my_index")

# later
rag = RAGKit.load("my_index")
```

## Retrieval quality

Measured with the included harness on [BEIR](https://github.com/beir-cellar/beir) SciFact (5,183 abstracts, 300 test claims). Embedding model all-MiniLM-L6-v2, CPU/MPS laptop.

| Retriever | nDCG@10 | Recall@100 | Latency p50 |
|---|---|---|---|
| dense (v0.1) | 0.645 | 0.925 | 3.6 ms |
| BM25 | 0.687 | 0.928 | 0.2 ms |
| hybrid, RRF | 0.715 | 0.955 | 3.8 ms |
| **hybrid, min-max fusion (default)** | **0.729** | **0.955** | 3.8 ms |
| hybrid + rerank top 20, ms-marco-MiniLM-L6-v2 | 0.701 | - | 86 ms |
| hybrid + rerank top 20, bge-reranker-v2-m3 | 0.742 | - | ~2 s |

- The dense and BM25 baselines reproduce published numbers (MTEB 0.645, BEIR paper 0.665 without stemming), which is how we know the harness is right.
- Hybrid beats either retriever alone. It also held on NFCorpus (0.344 vs 0.316 dense, 0.323 BM25). Min-max fusion beating RRF matches [Bruch et al. 2023](https://arxiv.org/abs/2210.11934).
- The small MS MARCO reranker hurts here: it was trained on web search queries, not scientific claims. The BEIR paper reports the same effect. A much larger reranker gains +0.013 at a large latency cost.
- Fusion weights are untuned. Rerank blending was tuned on the SciFact train split, and the gain didn't carry over to test.

Reproduce:

```bash
python examples/eval_scifact.py
python examples/eval_scifact.py --dataset nfcorpus
```

### PDF parsing

`python examples/eval_pdf.py`: two two-column arXiv papers (DPR, ColBERT), 17 questions written from the rendered pages, hybrid search top 5. A table answer only counts if the value and its row label come back in the same chunk.

| Setup | Text answers | Table answers | Parse time, 23 pages |
|---|---|---|---|
| pypdf + 512-char chunks (default) | 8/9 | 6/8 | 0.4 s |
| Docling Markdown + 512-char chunks | 9/9 | 2/8 | 8 s |
| Docling structure-aware chunks (`parser="docling"`) | 9/9 | 6/8 | 8 s |

On clean LaTeX papers pypdf holds up: it keeps table rows on one line, so a row fits in a chunk. Its text has flaws the answer-matching barely sees: ligature characters ("unﬁltered", now expanded by `PDFLoader`), lost spaces ("size of128"), footnotes spliced into sentences. Docling's text is clean, but the same Docling output scores 2/8 or 6/8 on tables depending only on how it is chunked. Chunking mattered more than parsing. 17 questions is a small test, so read this as "no clear winner on this kind of PDF", and expect Docling to matter more on scans, slides and office documents.

### Injection guard

`python examples/eval_injection.py`, default classifier (protectai/deberta-v3-base-prompt-injection-v2), threshold 0.5:

| Set | Rate |
|---|---|
| Clean SciFact abstracts | 0.2% false positives |
| [NotInject](https://huggingface.co/datasets/leolee99/NotInject), benign text with words like "ignore" | 43% false positives |
| Injection on its own ([deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)) | 44% detected |
| Same injection inside an abstract | 34% detected |
| Injection placed after 3 abstracts of text | 43% detected (0% without windowing) |
| Injection phrased as ordinary prose | 20% detected (n=10) |

The classifier is trained on standalone prompts, so a payload inside a long document gets diluted, and anything past its 512-token limit is invisible. Scoring short overlapping windows and taking the maximum fixes the second problem and improves the first (10% to 34%). It still misses most attacks that avoid the usual wording, and it over-flags harmless text about "ignoring" things. Treat it as one layer.

## Advanced usage

If you need more control, you can use the components directly:

```python
from ragkit import (
    PDFLoader,
    RecursiveCharacterSplitter,
    SentenceTransformerEmbeddings,
    FAISSStore,
    SimilarityRetriever,
    HuggingFaceLLM,
    QAChain,
)

# Load and split
loader = PDFLoader()
documents = loader.load("paper.pdf")

splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split(documents)

# Embed and store
embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
vectorstore = FAISSStore(embeddings)
vectorstore.add(chunks)

# Set up retrieval and generation
retriever = SimilarityRetriever(vectorstore, top_k=5)
llm = HuggingFaceLLM(model_name="HuggingFaceTB/SmolLM-360M-Instruct")

chain = QAChain(retriever=retriever, llm=llm)
answer = chain.run("What methodology did they use?")
```

Hybrid retrieval, reranking and screening compose the same way:

```python
from ragkit import (
    BM25Retriever,
    CrossEncoderReranker,
    GuardedRetriever,
    HybridRetriever,
    InjectionGuard,
    RerankRetriever,
)

dense = SimilarityRetriever(vectorstore, top_k=50)
bm25 = BM25Retriever(chunks, top_k=50)

hybrid = HybridRetriever([dense, bm25], top_k=20)            # fusion="rrf" also available
reranked = RerankRetriever(hybrid, CrossEncoderReranker(), top_k=5)
retriever = GuardedRetriever(reranked, InjectionGuard())     # screen at query time

chain = QAChain(retriever=retriever, llm=llm)
```

Evaluate any retriever on a BEIR dataset:

```python
from ragkit.eval import load_beir, evaluate, format_table

data = load_beir("scifact")
chunks = data.to_chunks()
result = evaluate(BM25Retriever(chunks, top_k=100), data.queries, data.qrels, name="bm25")
print(format_table([result]))
```

## Supported formats

| Format | Extension | Loader |
|--------|-----------|--------|
| PDF | .pdf | PDFLoader |
| Plain text | .txt | TextLoader |
| Markdown | .md | MarkdownLoader |
| PDF, Word, PowerPoint, Excel, HTML | .pdf .docx .pptx .xlsx .html | DoclingLoader (`parser="docling"`) |

## Project structure

```
ragkit/
├── loaders/          # document loading
├── splitters/        # text chunking
├── embeddings/       # vector embeddings
├── vectorstores/     # FAISS and simple numpy store
├── retrievers/       # dense, MMR, BM25, hybrid fusion, rerank and guard wrappers
├── rerankers/        # cross-encoder reranker
├── guards/           # prompt injection classifier
├── llms/             # HuggingFace, Ollama, OpenAI backends
├── chains/           # QA and conversational chains
└── eval/             # BEIR loader, retrieval metrics, evaluation loop
```

## Comparison with other frameworks

RAGKit is smaller and simpler than LangChain or LlamaIndex. It has fewer features and less flexibility, but it's easier to get started with and has fewer dependencies. 

If you need production features, extensive integrations, or enterprise support, those frameworks are probably better choices. RAGKit is more suited for prototyping, learning, or simple internal tools where you don't want to deal with a lot of complexity.

## Use cases

- Asking questions about PDFs (research papers, reports, etc)
- Searching through documentation
- Building a Q&A system over personal notes
- Understanding unfamiliar codebases

## Requirements

- Python 3.9 or higher
- Around 8GB RAM for the default models
- About 2GB disk space for model downloads

## License

MIT - see LICENSE file.
