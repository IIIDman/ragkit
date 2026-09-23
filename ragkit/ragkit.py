"""
RAGKit - main module

This is the high-level interface. For most use cases you can just use the
RAGKit class directly.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union
import pickle

from .core import Document, Chunk, Answer
from .loaders import TextLoader, PDFLoader, MarkdownLoader, DirectoryLoader
from .splitters import RecursiveCharacterSplitter
from .embeddings import SentenceTransformerEmbeddings
from .vectorstores import FAISSStore, SimpleStore
from .retrievers import BM25Retriever, HybridRetriever, RerankRetriever, SimilarityRetriever
from .rerankers import CrossEncoderReranker
from .guards import InjectionGuard
from .llms import HuggingFaceLLM, OllamaLLM
from .chains import QAChain

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = ("hybrid", "dense", "bm25")


class _BM25View:
    """One shared BM25 index, queried with its own number of results."""
    
    def __init__(self, bm25: BM25Retriever, top_k: int):
        self.bm25 = bm25
        self.top_k = top_k
    
    def retrieve_with_scores(self, query: str):
        return self.bm25.retrieve_with_scores(query, top_k=self.top_k)
    
    def retrieve(self, query: str):
        return self.bm25.retrieve(query, top_k=self.top_k)


class RAGKit:
    """
    Main interface for RAG applications.
    
    Handles document loading, chunking, embedding, and querying.
    Uses reasonable defaults so you can get started without much config.
    
    Retrieval is hybrid (dense + BM25) by default, which scored 0.729
    nDCG@10 on BEIR SciFact vs 0.645 for dense alone. A cross-encoder
    reranker and a prompt injection guard are opt-in.

    Example:
        rag = RAGKit()
        rag.add_document("report.pdf")
        answer = rag.query("What are the key findings?")
        print(answer.text)
    """
    
    # Map file extensions to loaders
    LOADER_MAP = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
    }
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm: Optional[str] = None,
        llm_backend: str = "huggingface",  # "huggingface", "ollama", "openai"
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 4,
        use_faiss: bool = True,
        device: Optional[str] = None,
        retrieval: str = "hybrid",
        fetch_k: int = 50,
        reranker: Optional[str] = None,
        rerank_depth: int = 20,
        rerank_alpha: float = 1.0,
        guard: bool = False,
        guard_threshold: float = 0.5,
        spotlight: bool = True,
    ):
        """
        Initialize RAGKit.
        
        Args:
            embedding_model: Sentence transformer model name
            llm: LLM model name (None for default based on backend)
            llm_backend: "huggingface", "ollama", or "openai"
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            use_faiss: Use FAISS for vector store (faster for large datasets)
            device: Device for models ("cpu", "cuda", "mps", or None for auto)
            retrieval: "hybrid" (dense + BM25), "dense" or "bm25"
            fetch_k: Candidates each hybrid sub-retriever returns before fusion
            reranker: Cross-encoder model name to rerank results, None to skip.
                Measure on your data first: rerankers trained on web search can
                hurt in other domains
            rerank_depth: Candidates sent to the reranker
            rerank_alpha: Reranker weight when blending with first-stage scores
            guard: Scan chunks for prompt injection when they are added and
                skip flagged ones (see flagged_chunks)
            guard_threshold: Injection probability at which a chunk is skipped
            spotlight: Wrap sources in randomly named tags in the prompt and
                tell the LLM not to follow instructions inside them
        """
        if retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval: {retrieval}. Use one of {RETRIEVAL_MODES}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.device = device
        self.use_faiss = use_faiss
        self.retrieval = retrieval
        self.fetch_k = fetch_k
        self.rerank_depth = rerank_depth
        self.rerank_alpha = rerank_alpha
        self.spotlight = spotlight
        
        # Initialize components
        self._embeddings = SentenceTransformerEmbeddings(
            model_name=embedding_model,
            device=device
        )
        
        self._splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vector store
        if use_faiss:
            self._vectorstore = FAISSStore(
                embedding_model=self._embeddings,
                dimension=self._embeddings.dimension
            )
        else:
            self._vectorstore = SimpleStore(embedding_model=self._embeddings)
        
        # Keyword index, kept in sync with the vector store
        self._bm25 = BM25Retriever() if retrieval in ("hybrid", "bm25") else None
        
        # Optional stages
        self._reranker = (
            CrossEncoderReranker(model_name=reranker, device=device) if reranker else None
        )
        self._guard = (
            InjectionGuard(threshold=guard_threshold, device=device) if guard else None
        )
        self.flagged_chunks: List[Chunk] = []
        
        # LLM
        self._llm = self._init_llm(llm, llm_backend)
        
        # Retriever and chain
        self._build_chain()
        
        # Track documents
        self._documents: List[Document] = []
    
    def _make_retriever(self, top_k: int):
        """Assemble the retrieval pipeline for a given number of results."""
        first_k = max(top_k, self.rerank_depth) if self._reranker else top_k
        
        if self.retrieval == "dense":
            first_stage = SimilarityRetriever(self._vectorstore, top_k=first_k)
        elif self.retrieval == "bm25":
            first_stage = _BM25View(self._bm25, first_k)
        else:
            depth = max(self.fetch_k, first_k)
            first_stage = HybridRetriever(
                [
                    SimilarityRetriever(self._vectorstore, top_k=depth),
                    _BM25View(self._bm25, depth),
                ],
                top_k=first_k,
            )
        
        if self._reranker:
            return RerankRetriever(
                first_stage, self._reranker, top_k=top_k, alpha=self.rerank_alpha
            )
        return first_stage
    
    def _build_chain(self) -> None:
        self._retriever = self._make_retriever(self.top_k)
        self._chain = QAChain(
            retriever=self._retriever,
            llm=self._llm,
            spotlight=self.spotlight,
        )
    
    def _index(self, chunks: List[Chunk]) -> int:
        """Screen (if guarded) and add chunks to every index. Returns chunks added."""
        if self._guard and chunks:
            chunks, flagged = self._guard.filter(chunks)
            for chunk in flagged:
                logger.warning(
                    "Skipped chunk from %s (injection score %.2f)",
                    chunk.metadata.get("source", "unknown"),
                    chunk.metadata["injection_score"],
                )
            self.flagged_chunks.extend(flagged)
        
        self._vectorstore.add(chunks)
        if self._bm25 is not None:
            self._bm25.add(chunks)
        return len(chunks)
    
    def _init_llm(self, model_name: Optional[str], backend: str):
        """Initialize LLM based on backend."""
        if backend == "huggingface":
            return HuggingFaceLLM(
                model_name=model_name or "HuggingFaceTB/SmolLM-135M-Instruct",
                device=self.device
            )
        elif backend == "ollama":
            return OllamaLLM(model_name=model_name or "llama3.2")
        elif backend == "openai":
            from .llms import OpenAILLM
            return OpenAILLM(model_name=model_name or "gpt-3.5-turbo")
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")
    
    def add_document(self, file_path: str) -> int:
        """
        Add a document to the knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Number of chunks added
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Select loader based on extension
        suffix = path.suffix.lower()
        if suffix not in self.LOADER_MAP:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {list(self.LOADER_MAP.keys())}"
            )
        
        # Load document
        loader = self.LOADER_MAP[suffix]()
        documents = loader.load(str(path))
        
        # Split into chunks and index
        chunks = self._splitter.split(documents)
        added = self._index(chunks)
        
        # Track documents
        self._documents.extend(documents)
        
        return added
    
    def add_documents(self, file_paths: List[str]) -> int:
        """
        Add multiple documents.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Total number of chunks added
        """
        total_chunks = 0
        for path in file_paths:
            total_chunks += self.add_document(path)
        return total_chunks
    
    def add_directory(
        self,
        directory_path: str,
        glob: str = "**/*",
        recursive: bool = True
    ) -> int:
        """
        Add all documents from a directory.
        
        Args:
            directory_path: Path to directory
            glob: Glob pattern for file matching
            recursive: Whether to search subdirectories
            
        Returns:
            Number of chunks added
        """
        loader = DirectoryLoader(glob_pattern=glob, recursive=recursive)
        documents = loader.load(directory_path)
        
        # Split into chunks and index
        chunks = self._splitter.split(documents)
        added = self._index(chunks)
        
        # Track documents
        self._documents.extend(documents)
        
        return added
    
    def add_text(self, text: str, metadata: Optional[dict] = None) -> int:
        """
        Add raw text to the knowledge base.
        
        Args:
            text: Text content
            metadata: Optional metadata dict
            
        Returns:
            Number of chunks added
        """
        document = Document(content=text, metadata=metadata or {})
        chunks = self._splitter.split([document])
        added = self._index(chunks)
        self._documents.append(document)
        return added
    
    def query(self, question: str) -> Answer:
        """
        Query the knowledge base.
        
        Args:
            question: Question to answer
            
        Returns:
            Answer object with text, sources, and chunks
        """
        return self._chain.run(question)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """
        Search for relevant chunks without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of results (default: self.top_k)
            
        Returns:
            List of relevant chunks
        """
        if top_k is None or top_k == self.top_k:
            return self._retriever.retrieve(query)
        return self._make_retriever(top_k).retrieve(query)
    
    def save(self, path: str) -> None:
        """
        Save the RAGKit index to disk.
        
        Args:
            path: Directory path to save to
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        self._vectorstore.save(str(save_path / "vectorstore"))
        
        # Save config
        config = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "embedding_model": self._embeddings.model_name,
            "use_faiss": self.use_faiss,
            "retrieval": self.retrieval,
            "fetch_k": self.fetch_k,
            "reranker": self._reranker.model_name if self._reranker else None,
            "rerank_depth": self.rerank_depth,
            "rerank_alpha": self.rerank_alpha,
            "spotlight": self.spotlight,
            "num_documents": len(self._documents),
            "num_chunks": len(self._vectorstore),
        }
        
        with open(save_path / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        
        print(f"Saved RAGKit index to {path}")
    
    @classmethod
    def load(
        cls,
        path: str,
        llm: Optional[str] = None,
        llm_backend: str = "huggingface",
        device: Optional[str] = None,
        **overrides,
    ) -> "RAGKit":
        """
        Load a RAGKit index from disk.
        
        Retrieval settings are restored from the saved config. The BM25
        index is rebuilt from the stored chunks.
        
        Args:
            path: Directory path to load from
            llm: LLM model name
            llm_backend: LLM backend
            device: Device for models
            **overrides: Any RAGKit setting to change, e.g. reranker=...
            
        Returns:
            Loaded RAGKit instance
        """
        load_path = Path(path)
        
        # Load config
        with open(load_path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        # Create instance (indexes saved by v0.1 have no retrieval keys:
        # they load as dense, which is what they were built for)
        settings = {
            "embedding_model": config["embedding_model"],
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "top_k": config["top_k"],
            "use_faiss": config.get("use_faiss", True),
            "retrieval": config.get("retrieval", "dense"),
            "fetch_k": config.get("fetch_k", 50),
            "reranker": config.get("reranker"),
            "rerank_depth": config.get("rerank_depth", 20),
            "rerank_alpha": config.get("rerank_alpha", 1.0),
            "spotlight": config.get("spotlight", True),
        }
        settings.update(overrides)
        rag = cls(llm=llm, llm_backend=llm_backend, device=device, **settings)
        
        # Load vector store and rebuild the keyword index from its chunks
        rag._vectorstore = type(rag._vectorstore).load(
            str(load_path / "vectorstore"),
            embedding_model=rag._embeddings
        )
        if rag._bm25 is not None:
            rag._bm25.add(rag._vectorstore._chunks)
        
        rag._build_chain()
        
        print(f"Loaded RAGKit index from {path} ({config['num_chunks']} chunks)")
        
        return rag
    
    @property
    def num_chunks(self) -> int:
        """Number of chunks in the vector store."""
        return len(self._vectorstore)
    
    @property
    def num_documents(self) -> int:
        """Number of documents added."""
        return len(self._documents)
    
    def __repr__(self) -> str:
        return (
            f"RAGKit(documents={self.num_documents}, "
            f"chunks={self.num_chunks}, "
            f"retrieval={self.retrieval}, "
            f"top_k={self.top_k})"
        )
