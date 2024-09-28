"""Basic tests for RAGKit."""

import pytest
from ragkit import Document, Chunk, Answer
from ragkit import RecursiveCharacterSplitter


class TestCoreDataStructures:
    """Test core data structures."""
    
    def test_document_creation(self):
        doc = Document(content="Hello world", metadata={"source": "test"})
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test"
    
    def test_document_default_metadata(self):
        doc = Document(content="Hello")
        assert doc.metadata == {}
    
    def test_chunk_creation(self):
        chunk = Chunk(content="Test chunk", metadata={"id": 1})
        assert chunk.content == "Test chunk"
        assert chunk.embedding is None
    
    def test_answer_creation(self):
        answer = Answer(text="The answer is 42", sources=["doc1.pdf"])
        assert answer.text == "The answer is 42"
        assert "doc1.pdf" in answer.sources
    
    def test_answer_str(self):
        answer = Answer(text="Test answer")
        assert str(answer) == "Test answer"


class TestSplitter:
    """Test text splitters."""
    
    def test_recursive_splitter(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=10)
        
        doc = Document(
            content="This is a test. " * 20,
            metadata={"source": "test"}
        )
        
        chunks = splitter.split([doc])
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all("source" in c.metadata for c in chunks)
    
    def test_splitter_preserves_metadata(self):
        splitter = RecursiveCharacterSplitter(chunk_size=50)
        
        doc = Document(
            content="Short text",
            metadata={"author": "Test Author"}
        )
        
        chunks = splitter.split([doc])
        
        assert chunks[0].metadata["author"] == "Test Author"


class TestLoaders:
    """Test document loaders."""
    
    def test_text_loader_import(self):
        from ragkit import TextLoader
        loader = TextLoader()
        assert loader.encoding == "utf-8"
    
    def test_pdf_loader_import(self):
        from ragkit import PDFLoader
        loader = PDFLoader()
        assert hasattr(loader, "load")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
