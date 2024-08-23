"""Markdown file loader."""

from pathlib import Path
from typing import List
from ..core import Document


class MarkdownLoader:
    """Load Markdown files."""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Document]:
        """Load a markdown file and return a list with one Document."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = path.read_text(encoding=self.encoding)
        
        # Extract frontmatter if present
        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "filetype": "markdown"
        }
        
        # Simple frontmatter extraction (YAML between --- markers)
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                content = parts[2].strip()
                metadata["frontmatter"] = frontmatter
        
        return [Document(content=content, metadata=metadata)]
