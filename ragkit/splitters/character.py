"""Character-based text splitter."""

from typing import List
from ..core import Document, Chunk


class RecursiveCharacterSplitter:
    """
    Split text recursively by different separators.
    
    Tries to split by paragraphs first, then sentences, then words,
    then characters, keeping chunks close to the target size.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into chunks."""
        all_chunks = []
        
        for doc in documents:
            chunks = self._split_text(doc.content)
            
            for i, chunk_text in enumerate(chunks):
                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                all_chunks.append(chunk)
        
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive separators."""
        return self._split_recursive(text, self.separators)
    
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text with fallback separators."""
        if not text:
            return []
        
        # If text is small enough, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator
        for i, sep in enumerate(separators):
            if sep == "":
                # Last resort: split by characters
                return self._split_by_chars(text)
            
            if sep in text:
                splits = text.split(sep)
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    # Add separator back (except for last split)
                    piece = split + sep if sep else split
                    
                    if len(current_chunk) + len(piece) <= self.chunk_size:
                        current_chunk += piece
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # If this piece is too large, split it further
                        if len(piece) > self.chunk_size:
                            remaining_seps = separators[i + 1:]
                            sub_chunks = self._split_recursive(piece, remaining_seps)
                            chunks.extend(sub_chunks[:-1])
                            current_chunk = sub_chunks[-1] if sub_chunks else ""
                        else:
                            current_chunk = piece
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Add overlap between chunks
                if self.chunk_overlap > 0:
                    chunks = self._add_overlap(chunks)
                
                return chunks
        
        return [text]
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Split text by character count."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # Prepend overlap to current chunk
            overlapped.append(overlap_text + " " + curr_chunk)
        
        return overlapped


class CharacterSplitter:
    """Simple character-based splitter (no recursive logic)."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into fixed-size chunks."""
        all_chunks = []
        
        for doc in documents:
            text = doc.content
            step = self.chunk_size - self.chunk_overlap
            
            for i, start in enumerate(range(0, len(text), step)):
                chunk_text = text[start:start + self.chunk_size]
                if chunk_text.strip():
                    chunk = Chunk(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": i,
                            "start_char": start
                        }
                    )
                    all_chunks.append(chunk)
        
        return all_chunks
