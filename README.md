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
- Uses FAISS for vector search
- High-level API for quick prototyping, lower-level components if you need more control

## Installation

```bash
pip install ragkit

# or with PDF support
pip install ragkit[pdf]

# or everything
pip install ragkit[all]
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
Query -> Embed -> Search vector store -> Get relevant chunks -> LLM generates answer
```

The basic flow:
1. Documents get split into chunks and embedded as vectors
2. When you ask a question, it's embedded and compared against stored vectors  
3. Most similar chunks are retrieved and passed to an LLM
4. The LLM generates an answer based on the retrieved context

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

### Saving and loading

```python
rag.save("my_index")

# later
rag = RAGKit.load("my_index")
```

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

## Supported formats

| Format | Extension | Loader |
|--------|-----------|--------|
| PDF | .pdf | PDFLoader |
| Plain text | .txt | TextLoader |
| Markdown | .md | MarkdownLoader |

## Project structure

```
ragkit/
├── loaders/          # document loading
├── splitters/        # text chunking
├── embeddings/       # vector embeddings
├── vectorstores/     # FAISS and simple numpy store
├── retrievers/       # similarity search, MMR
├── llms/             # HuggingFace, Ollama, OpenAI backends
└── chains/           # QA and conversational chains
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
