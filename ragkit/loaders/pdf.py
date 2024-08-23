"""PDF file loader."""

from pathlib import Path
from typing import List
from ..core import Document


class PDFLoader:
    """Load PDF files using pypdf."""
    
    def __init__(self, extract_images: bool = False):
        self.extract_images = extract_images
    
    def load(self, file_path: str) -> List[Document]:
        """Load a PDF file and return a list of Documents (one per page)."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install it with: pip install pypdf"
            )
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        reader = PdfReader(str(path))
        documents = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                documents.append(Document(
                    content=text,
                    metadata={
                        "source": str(path.absolute()),
                        "filename": path.name,
                        "filetype": "pdf",
                        "page": page_num,
                        "total_pages": len(reader.pages)
                    }
                ))
        
        return documents


class PDFPlumberLoader:
    """Load PDF files using pdfplumber (better for tables)."""
    
    def load(self, file_path: str) -> List[Document]:
        """Load a PDF file using pdfplumber."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required. "
                "Install it with: pip install pdfplumber"
            )
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        documents = []
        
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        content=text,
                        metadata={
                            "source": str(path.absolute()),
                            "filename": path.name,
                            "filetype": "pdf",
                            "page": page_num,
                            "total_pages": len(pdf.pages)
                        }
                    ))
        
        return documents
