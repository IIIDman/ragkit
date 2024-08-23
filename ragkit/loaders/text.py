"""Text file loader."""

from pathlib import Path
from typing import List
from ..core import Document


class TextLoader:
    """Load plain text files."""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Document]:
        """Load a text file and return a list with one Document."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = path.read_text(encoding=self.encoding)
        
        return [Document(
            content=content,
            metadata={
                "source": str(path.absolute()),
                "filename": path.name,
                "filetype": "text"
            }
        )]
