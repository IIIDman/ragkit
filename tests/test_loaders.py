"""Tests for PDF cleanup and the Docling loader (with a fake Docling)."""

from types import SimpleNamespace

from ragkit import DoclingLoader
from ragkit.loaders.pdf import clean_pdf_text


def test_clean_pdf_text_expands_ligatures():
    assert clean_pdf_text("unﬁltered ﬂow eﬀect") == "unfiltered flow effect"


def test_clean_pdf_text_rejoins_hyphenated_words_only():
    assert clean_pdf_text("perfor-\nmance") == "performance"
    # A hyphen before a digit or capital is kept: "Top-20", "BM25-Based"
    assert clean_pdf_text("Top-\n20") == "Top-\n20"
    assert clean_pdf_text("well-known") == "well-known"


class FakeChunker:
    def chunk(self, doc):
        prov = SimpleNamespace(page_no=3)
        item = SimpleNamespace(prov=[prov])
        meta = SimpleNamespace(headings=["4 Experiments"], doc_items=[item])
        return [SimpleNamespace(text="BM25, Top-20 = 59.1", meta=meta)]

    def contextualize(self, chunk):
        return "4 Experiments\nBM25, Top-20 = 59.1 &amp; more"


def make_loader(**kwargs):
    loader = DoclingLoader(**kwargs)
    document = SimpleNamespace(export_to_markdown=lambda: "# Title\n\nA &amp; B")
    loader._converter = SimpleNamespace(convert=lambda path: SimpleNamespace(document=document))
    loader._chunker = FakeChunker()
    return loader


def test_docling_chunks_carry_structure(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF")
    docs = make_loader().load(str(path))
    assert len(docs) == 1
    assert docs[0].content == "4 Experiments\nBM25, Top-20 = 59.1 & more"
    assert docs[0].metadata["pre_chunked"] is True
    assert docs[0].metadata["headings"] == ["4 Experiments"]
    assert docs[0].metadata["page"] == 3
    assert docs[0].metadata["filetype"] == "pdf"


def test_docling_markdown_mode(tmp_path):
    path = tmp_path / "report.docx"
    path.write_bytes(b"PK")
    docs = make_loader(chunking=False).load(str(path))
    assert docs[0].content == "# Title\n\nA & B"
    assert "pre_chunked" not in docs[0].metadata
