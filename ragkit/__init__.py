"""
RAGKit - Simple RAG Framework

A lightweight framework for building RAG applications that runs locally.

Basic usage:
    from ragkit import RAGKit
    
    rag = RAGKit()
    rag.add_document("paper.pdf")
    answer = rag.query("What is the main finding?")
    print(answer)

You can also use the component classes directly for more control -
see the README for examples.
"""

__version__ = "0.1.0"
__author__ = "Dmitriy Tsarev"
__license__ = "MIT"

# High-level API
from .ragkit import RAGKit

# Core data structures
from .core import Document, Chunk, Answer

# Loaders
from .loaders import (
    TextLoader,
    PDFLoader,
    PDFPlumberLoader,
    MarkdownLoader,
    DirectoryLoader,
)

# Splitters
from .splitters import (
    RecursiveCharacterSplitter,
    CharacterSplitter,
    TokenSplitter,
)

# Embeddings
from .embeddings import (
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings,
)

# Vector stores
from .vectorstores import (
    FAISSStore,
    SimpleStore,
)

# Retrievers
from .retrievers import (
    SimilarityRetriever,
    MMRRetriever,
)

# LLMs
from .llms import (
    HuggingFaceLLM,
    OllamaLLM,
    OpenAILLM,
)

# Chains
from .chains import (
    QAChain,
    ConversationalChain,
)

__all__ = [
    # Version
    "__version__",
    
    # High-level
    "RAGKit",
    
    # Core
    "Document",
    "Chunk",
    "Answer",
    
    # Loaders
    "TextLoader",
    "PDFLoader",
    "PDFPlumberLoader",
    "MarkdownLoader",
    "DirectoryLoader",
    
    # Splitters
    "RecursiveCharacterSplitter",
    "CharacterSplitter",
    "TokenSplitter",
    
    # Embeddings
    "SentenceTransformerEmbeddings",
    "HuggingFaceEmbeddings",
    
    # Vector stores
    "FAISSStore",
    "SimpleStore",
    
    # Retrievers
    "SimilarityRetriever",
    "MMRRetriever",
    
    # LLMs
    "HuggingFaceLLM",
    "OllamaLLM",
    "OpenAILLM",
    
    # Chains
    "QAChain",
    "ConversationalChain",
]
