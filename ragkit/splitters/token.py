"""Token-based text splitter."""

from typing import List, Optional
from ..core import Document, Chunk


class TokenSplitter:
    """
    Split text by token count using a tokenizer.
    
    More precise for LLM context windows than character splitting.
    """
    
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 20,
        tokenizer: Optional[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = self._load_tokenizer(tokenizer)
    
    def _load_tokenizer(self, tokenizer_name: Optional[str]):
        """Load tokenizer (defaults to simple whitespace if no transformers)."""
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(tokenizer_name)
            except ImportError:
                print("Warning: transformers not installed, using simple tokenizer")
        
        # Simple fallback tokenizer
        return None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        # Simple whitespace tokenization as fallback
        return text.split()
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        if self.tokenizer:
            return self.tokenizer.convert_tokens_to_string(tokens)
        return " ".join(tokens)
    
    def split(self, documents: List[Document]) -> List[Chunk]:
        """Split documents into token-based chunks."""
        all_chunks = []
        
        for doc in documents:
            tokens = self._tokenize(doc.content)
            step = self.chunk_size - self.chunk_overlap
            chunk_id = 0
            
            for start in range(0, len(tokens), step):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self._detokenize(chunk_tokens)
                
                if chunk_text.strip():
                    chunk = Chunk(
                        content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": chunk_id,
                            "token_start": start,
                            "token_end": end
                        }
                    )
                    all_chunks.append(chunk)
                    chunk_id += 1
        
        return all_chunks
