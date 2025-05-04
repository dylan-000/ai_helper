#!/usr/bin/env python3
"""
Script to create and save a ChromaDB vector store for the Terraria guide data
"""

import os
import sys
import traceback
import shutil
import argparse
from typing import Dict, Any

# Add parent directory to path to allow importing from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model.model_utils.vector_store import (
    load_documents,
    chunk_documents,
    create_vector_store
)

def cleanup_vector_store(persist_dir: str, force: bool = False) -> bool:
    """
    Clean up existing vector store directory
    
    Args:
        persist_dir: Path to the vector store directory
        force: If True, delete without prompt
    
    Returns:
        True if cleaned up or not needed, False if aborted
    """
    if os.path.exists(persist_dir):
        if not force:
            response = input(f"Vector store already exists at {persist_dir}. Delete and recreate? (y/n): ")
            if response.lower() != 'y':
                print("Aborted. Use --force to bypass this prompt.")
                return False
        
        print(f"Removing existing vector store at {persist_dir}")
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"Error removing directory: {e}")
            print("Please manually delete the directory and try again.")
            return False
    
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create a ChromaDB vector store for Terraria guide data")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing vector store")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=4000, help="Batch size for adding to ChromaDB")
    return parser.parse_args()

def main():
    """Main function to create and save the vector store"""
    
    # Parse command line arguments
    args = parse_args()
    
    # Configuration
    data_dir = "model/data"
    persist_dir = "model/data/chroma_db"
    collection_name = "terraria_guide"
    embedding_model = "nomic-embed-text"  # Use nomic-embed-text for embeddings
    
    # Clean up existing vector store if needed
    if not cleanup_vector_store(persist_dir, args.force):
        return
    
    try:
        print(f"Loading documents from {data_dir} (processing all files, including large ones)")
        documents = load_documents(data_dir, process_large_files=True)
        
        if not documents:
            print(f"Error: No documents found in {data_dir}")
            return
        
        print(f"Chunking {len(documents)} documents (size={args.chunk_size}, overlap={args.chunk_overlap})")
        chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        print(f"Creating vector store in {persist_dir}")
        collection = create_vector_store(
            chunks=chunks,
            collection_name=collection_name,
            embedding_model=embedding_model,
            persist_path=persist_dir,
            batch_size=args.batch_size
        )
        
        print(f"Vector store created with {collection.count()} chunks")
        print(f"Vector store saved to {persist_dir}")
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        print("\nRecommendations:")
        print("1. Make sure Ollama is running locally")
        print("2. Check that the nomic-embed-text model is available in Ollama (run: ollama pull nomic-embed-text)")
        print("3. Verify that the data directory contains valid text files")
        print("4. Try these options to resolve batch size issues:")
        print("   - Decrease batch size: --batch-size 2000")
        print("   - Increase chunk size: --chunk-size 2000")
    
if __name__ == "__main__":
    main() 