"""Layout-aware document loader using Docling."""

import html
from pathlib import Path
from typing import List, Optional

from ..core import Document


class DoclingLoader:
    """
    Load PDF, DOCX, PPTX, XLSX and HTML files with Docling.

    Docling runs layout models over each page, so it keeps reading order in
    multi-column papers and recovers tables as tables. By default it also
    chunks the document itself (Docling's HybridChunker): chunks follow the
    document structure, each is prefixed with its section headings, and table
    rows are written out as "row, column = value" so a number keeps its labels.

    Returned documents have metadata["pre_chunked"] = True, which tells
    RAGKit to index them as they are instead of splitting them again.

    Requires: pip install docling
    """

    SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm")

    def __init__(
        self,
        chunking: bool = True,
        max_tokens: int = 256,
        tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ocr: bool = False,
    ):
        """
        Initialize loader. Models load on first use.

        Args:
            chunking: Return structure-aware chunks. False returns one
                Markdown document per file, for use with any ragkit splitter
            max_tokens: Chunk size limit, counted with tokenizer_model
            tokenizer_model: Tokenizer of the embedding model the chunks are for
            ocr: Run OCR on page images. Needed for scanned PDFs, slow
                (about 10x) and unnecessary for born-digital files
        """
        self.chunking = chunking
        self.max_tokens = max_tokens
        self.tokenizer_model = tokenizer_model
        self.ocr = ocr
        self._converter = None
        self._chunker = None

    @property
    def converter(self):
        """Lazy build the Docling converter."""
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
            except ImportError:
                raise ImportError(
                    "docling is required. Install it with: pip install docling"
                )
            options = PdfPipelineOptions(do_ocr=self.ocr)
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    @property
    def chunker(self):
        """Lazy build the structure-aware chunker."""
        if self._chunker is None:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from transformers import AutoTokenizer

            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_model),
                max_tokens=self.max_tokens,
            )
            self._chunker = HybridChunker(tokenizer=tokenizer)
        return self._chunker

    def load(self, file_path: str) -> List[Document]:
        """Convert a file and return its chunks (or one Markdown document)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = self.converter.convert(str(path)).document
        base = {
            "source": str(path.absolute()),
            "filename": path.name,
            "filetype": path.suffix.lower().lstrip("."),
            "loader": "docling",
        }

        if not self.chunking:
            # Docling escapes &, < and > in Markdown output
            return [Document(content=html.unescape(doc.export_to_markdown()), metadata=base)]

        documents = []
        for chunk in self.chunker.chunk(doc):
            metadata = {**base, "pre_chunked": True}
            if chunk.meta.headings:
                metadata["headings"] = list(chunk.meta.headings)
            page = self._first_page(chunk)
            if page is not None:
                metadata["page"] = page
            text = html.unescape(self.chunker.contextualize(chunk))
            documents.append(Document(content=text, metadata=metadata))
        return documents

    @staticmethod
    def _first_page(chunk) -> Optional[int]:
        for item in chunk.meta.doc_items:
            for prov in getattr(item, "prov", None) or []:
                return prov.page_no
        return None
