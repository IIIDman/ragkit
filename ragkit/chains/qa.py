"""Question-answering chains."""

from typing import Optional, List
from ..core import Answer, Chunk


DEFAULT_PROMPT_TEMPLATE = """Use the following context to answer the question. 
If you cannot answer based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


class QAChain:
    """
    Question-answering chain that combines retrieval and generation.
    """
    
    def __init__(
        self,
        retriever,
        llm,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        """
        Initialize QA chain.
        
        Args:
            retriever: Retriever with retrieve method
            llm: LLM with generate method
            prompt_template: Template with {context} and {question} placeholders
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
    
    def run(self, question: str) -> Answer:
        """
        Answer a question using retrieval-augmented generation.
        
        Args:
            question: Question to answer
            
        Returns:
            Answer object with text, sources, and chunks
        """
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(question)
        
        if not chunks:
            return Answer(
                text="I couldn't find any relevant information to answer your question.",
                sources=[],
                chunks=[]
            )
        
        # Build context from chunks
        context = self._build_context(chunks)
        
        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # Generate answer
        answer_text = self.llm.generate(prompt)
        
        # Extract sources
        sources = self._extract_sources(chunks)
        
        return Answer(
            text=answer_text,
            sources=sources,
            chunks=chunks
        )
    
    def _build_context(self, chunks: List[Chunk]) -> str:
        """Build context string from chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "")
            
            header = f"[Source {i}: {source}"
            if page:
                header += f", Page {page}"
            header += "]"
            
            context_parts.append(f"{header}\n{chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Chunk]) -> List[str]:
        """Extract unique sources from chunks."""
        sources = []
        seen = set()
        
        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown")
            if source not in seen:
                sources.append(source)
                seen.add(source)
        
        return sources


class ConversationalChain:
    """
    Conversational QA chain with memory.
    
    Maintains conversation history for multi-turn interactions.
    """
    
    def __init__(
        self,
        retriever,
        llm,
        max_history: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_history = max_history
        self.history: List[tuple] = []  # List of (question, answer) tuples
    
    def run(self, question: str) -> Answer:
        """Answer a question with conversation context."""
        # Build conversation history
        history_text = ""
        if self.history:
            history_parts = []
            for q, a in self.history[-self.max_history:]:
                history_parts.append(f"Human: {q}\nAssistant: {a}")
            history_text = "\n\n".join(history_parts)
        
        # Retrieve chunks
        chunks = self.retriever.retrieve(question)
        
        # Build context
        context = "\n\n".join([chunk.content for chunk in chunks])
        
        # Build prompt with history
        prompt = f"""Use the following context and conversation history to answer the question.

Context:
{context}

Previous conversation:
{history_text}

Current question: {question}

Answer:"""
        
        # Generate answer
        answer_text = self.llm.generate(prompt)
        
        # Update history
        self.history.append((question, answer_text))
        
        return Answer(
            text=answer_text,
            sources=[c.metadata.get("source", "") for c in chunks],
            chunks=chunks
        )
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
