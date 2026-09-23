"""PDF file loader."""

import re
from pathlib import Path
from typing import List
from ..core import Document

# Typographic ligatures that PDF text extraction returns as single characters.
# "unﬁltered" would never match a search for "unfiltered".
_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
})
# A word split across lines: "perfor-\nmance"
_LINE_BREAK_HYPHEN = re.compile(r"([a-z])-\n([a-z])")


def clean_pdf_text(text: str) -> str:
    """Expand ligatures and rejoin words hyphenated at line breaks."""
    text = text.translate(_LIGATURES)
    return _LINE_BREAK_HYPHEN.sub(r"\1\2", text)


class PDFLoader:
    """Load PDF files using pypdf."""
    
    def __init__(self, extract_images: bool = False, clean: bool = True):
        """
        Args:
            extract_images: Unused, kept for compatibility
            clean: Expand ligatures and rejoin hyphenated line breaks
        """
        self.extract_images = extract_images
        self.clean = clean
    
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
            if self.clean:
                text = clean_pdf_text(text)
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
