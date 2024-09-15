"""HuggingFace LLM backend."""

from typing import Optional


class HuggingFaceLLM:
    """
    LLM using HuggingFace transformers.
    
    Default: SmolLM-135M (very small, runs on any hardware)
    For better quality: SmolLM-360M, Phi-2, Mistral-7B
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device is None else None,
            low_cpu_mem_usage=True,
        )
        
        if self.device and self.device != "auto":
            self._model = self._model.to(self.device)
        
        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
    
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
    
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


class OllamaLLM:
    """
    LLM using Ollama (local model server).
    
    Requires Ollama to be installed and running.
    https://ollama.ai
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
    
    def generate(self, prompt: str) -> str:
        """Generate text using Ollama API."""
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                }
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")
        
        return response.json()["response"].strip()


class OpenAILLM:
    """
    LLM using OpenAI API.
    
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai is required. Install with: pip install openai")
            
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content.strip()
