"""Sentence Transformer embeddings."""

from typing import List, Optional
import numpy as np


class SentenceTransformerEmbeddings:
    """
    Generate embeddings using Sentence Transformers.
    
    Default model: all-MiniLM-L6-v2 (fast, 384 dimensions)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._device = device
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with: pip install sentence-transformers"
                )
            
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100
        )
        
        return np.array(embeddings)
    
    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed([text])[0]
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for documents."""
        return self.embed(texts)


class HuggingFaceEmbeddings:
    """
    Generate embeddings using any HuggingFace model.
    
    More flexible than SentenceTransformer wrapper but requires
    more setup.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        
        if self.device:
            self._model = self._model.to(self.device)
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using mean pooling."""
        import torch
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        if self.device:
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Mean pooling
            attention_mask = encoded["attention_mask"]
            embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        
        # Normalize
        mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
        
        return mean_embeddings.cpu().numpy()
    
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.embed(texts)
