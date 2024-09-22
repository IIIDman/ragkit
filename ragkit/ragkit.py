"""
RAGKit - main module

This is the high-level interface. For most use cases you can just use the
RAGKit class directly.
"""

from pathlib import Path
from typing import Optional, List, Union
import pickle

from .core import Document, Chunk, Answer
from .loaders import TextLoader, PDFLoader, MarkdownLoader, DirectoryLoader
from .splitters import RecursiveCharacterSplitter
from .embeddings import SentenceTransformerEmbeddings
from .vectorstores import FAISSStore, SimpleStore
from .retrievers import SimilarityRetriever
from .llms import HuggingFaceLLM, OllamaLLM
from .chains import QAChain


class RAGKit:
    """
    Main interface for RAG applications.
    
    Handles document loading, chunking, embedding, and querying.
    Uses reasonable defaults so you can get started without much config.
    
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
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.device = device
        
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
        
        # LLM
        self._llm = self._init_llm(llm, llm_backend)
        
        # Retriever and chain
        self._retriever = SimilarityRetriever(
            vectorstore=self._vectorstore,
            top_k=top_k
        )
        
        self._chain = QAChain(
            retriever=self._retriever,
            llm=self._llm
        )
        
        # Track documents
        self._documents: List[Document] = []
    
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
        
        # Split into chunks
        chunks = self._splitter.split(documents)
        
        # Add to vector store
        self._vectorstore.add(chunks)
        
        # Track documents
        self._documents.extend(documents)
        
        return len(chunks)
    
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
        
        # Split into chunks
        chunks = self._splitter.split(documents)
        
        # Add to vector store
        self._vectorstore.add(chunks)
        
        # Track documents
        self._documents.extend(documents)
        
        return len(chunks)
    
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
        self._vectorstore.add(chunks)
        self._documents.append(document)
        return len(chunks)
    
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
        k = top_k or self.top_k
        results = self._vectorstore.search(query, top_k=k)
        return [chunk for chunk, score in results]
    
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
    ) -> "RAGKit":
        """
        Load a RAGKit index from disk.
        
        Args:
            path: Directory path to load from
            llm: LLM model name
            llm_backend: LLM backend
            device: Device for models
            
        Returns:
            Loaded RAGKit instance
        """
        load_path = Path(path)
        
        # Load config
        with open(load_path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        # Create instance
        rag = cls(
            embedding_model=config["embedding_model"],
            llm=llm,
            llm_backend=llm_backend,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            top_k=config["top_k"],
            device=device,
        )
        
        # Load vector store
        rag._vectorstore = type(rag._vectorstore).load(
            str(load_path / "vectorstore"),
            embedding_model=rag._embeddings
        )
        
        # Update retriever
        rag._retriever = SimilarityRetriever(
            vectorstore=rag._vectorstore,
            top_k=rag.top_k
        )
        
        rag._chain = QAChain(
            retriever=rag._retriever,
            llm=rag._llm
        )
        
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
            f"top_k={self.top_k})"
        )
