"""
Chat with PDF Example

This example shows how to use RAGKit to chat with PDF documents.

Usage:
    python chat_with_pdf.py path/to/your/document.pdf
"""

import sys
from ragkit import RAGKit


def main():
    if len(sys.argv) < 2:
        print("Usage: python chat_with_pdf.py <pdf_path>")
        print("\nExample: python chat_with_pdf.py research_paper.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print(f"Loading PDF: {pdf_path}")
    print("This may take a moment on first run (downloading models)...\n")
    
    # Initialize RAGKit
    rag = RAGKit()
    
    # Add the PDF
    try:
        num_chunks = rag.add_document(pdf_path)
        print(f"Loaded PDF into {num_chunks} chunks\n")
    except FileNotFoundError:
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    except ImportError:
        print("Error: pypdf is required for PDF support.")
        print("Install it with: pip install pypdf")
        sys.exit(1)
    
    # Interactive chat loop
    print("="*50)
    print("Chat with your PDF. Type 'quit' to exit.")
    print("="*50)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nThinking...")
            answer = rag.query(question)
            
            print(f"\nAnswer: {answer.text}")
            
            if answer.sources:
                print(f"\nSources: {', '.join(answer.sources)}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
