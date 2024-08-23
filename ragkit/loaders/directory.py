"""Directory loader for batch loading multiple files."""

from pathlib import Path
from typing import List, Optional
from ..core import Document
from .text import TextLoader
from .pdf import PDFLoader
from .markdown import MarkdownLoader


class DirectoryLoader:
    """Load all documents from a directory."""
    
    # Map file extensions to loaders
    LOADER_MAP = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
    }
    
    def __init__(
        self,
        glob_pattern: str = "**/*",
        recursive: bool = True,
        silent_errors: bool = True
    ):
        self.glob_pattern = glob_pattern
        self.recursive = recursive
        self.silent_errors = silent_errors
    
    def load(self, directory_path: str) -> List[Document]:
        """Load all supported files from a directory."""
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        documents = []
        
        # Find all files matching the glob pattern
        if self.recursive:
            files = list(path.glob(self.glob_pattern))
        else:
            files = list(path.glob(self.glob_pattern.replace("**/", "")))
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            suffix = file_path.suffix.lower()
            
            if suffix in self.LOADER_MAP:
                try:
                    loader = self.LOADER_MAP[suffix]()
                    docs = loader.load(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    if not self.silent_errors:
                        raise
                    print(f"Warning: Could not load {file_path}: {e}")
        
        return documents
