"""
RAGKit Quickstart Example

This example shows the simplest way to use RAGKit.
"""

from ragkit import RAGKit

# Initialize RAGKit with defaults
# First run will download embedding model (~90MB)
print("Initializing RAGKit...")
rag = RAGKit()

# Add a document (create a sample text file for testing)
sample_text = """
Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed. There are three main types:

1. Supervised Learning: The algorithm learns from labeled training data.
   Examples include classification and regression tasks.

2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
   Examples include clustering and dimensionality reduction.

3. Reinforcement Learning: The algorithm learns through trial and error,
   receiving rewards or penalties for actions.

Deep learning is a subset of machine learning that uses neural networks
with many layers. It has been particularly successful in image recognition,
natural language processing, and game playing.
"""

# Add text directly
print("Adding sample text...")
rag.add_text(sample_text, metadata={"source": "ml_basics.txt"})

# Query the knowledge base
print("\n" + "="*50)
print("Query: What are the types of machine learning?")
print("="*50)

answer = rag.query("What are the types of machine learning?")
print(f"\nAnswer: {answer.text}")
print(f"\nSources: {answer.sources}")

# Another query
print("\n" + "="*50)
print("Query: What is deep learning?")
print("="*50)

answer = rag.query("What is deep learning?")
print(f"\nAnswer: {answer.text}")

print("\nDone.")
