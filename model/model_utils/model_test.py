#!/usr/bin/env python3
"""
Script to test the RAG capabilities of the Ollama model

This script provides a command-line interface to interact with the Terraria AI Guide.
It loads the vector store and allows testing queries with the multi-step reasoning pipeline.

Usage:
    python Model/model_test.py [--model MODEL] [--verbose]

Options:
    --model MODEL    Specify the Ollama model to use for generation (default: llama3.2:latest)
    --verbose        Print detailed debugging information

Examples:
    python Model/model_test.py
    python Model/model_test.py --temperature 0.8 --verbose
"""

import os
import sys
import argparse
import json
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

# We need to change to the root dir to run some tests
#here = Path(__file__).resolve()
#root_dir = here.parents[2]          
#os.chdir(root_dir)

# Import modules from our own model_utils folder
from vector_store import load_vector_store
from processor import QueryProcessor

# Constants
EMBEDDING_MODEL = "nomic-embed-text"  # Always use this for embeddings to ensure compatibility

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the Terraria AI Guide with RAG capabilities")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="Ollama model to use for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debugging information")
    parser.add_argument("--test-embedding", action="store_true", help="Test embedding function only")
    return parser.parse_args()

def check_vector_store(persist_path: str = "Model/model_utils/data/chroma_db") -> bool:
    """Check if the vector store exists"""
    return os.path.exists(persist_path) and os.path.isdir(persist_path)

def setup_vector_store(persist_path: str) -> Optional[Any]:
    """Set up the vector store"""
    if check_vector_store(persist_path):
        try:
            return load_vector_store(
                collection_name="terraria_guide",
                embedding_model=EMBEDDING_MODEL,  # Always use the same embedding model
                persist_path=persist_path
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("\nDetailed error information:")
            traceback.print_exc()
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running (ollama serve)")
            print(f"2. Check that you have pulled the embedding model: ollama pull {EMBEDDING_MODEL}")
            print("3. You may need to recreate the vector store: python Model/model_utils/create_vector_store.py")
            return None
    else:
        print("Vector store not found. You need to create it first. Run:")
        print("python Model/model_utils/create_vector_store.py")
        return None

def print_separator(char: str = "=", length: int = 80):
    """Print a separator line"""
    print(char * length)

def print_result(result: Dict[str, Any], verbose: bool = False):
    """Print the result of query processing"""
    print_separator()
    print(f"QUERY: {result['query']}")
    print_separator()
    
    # Always show the retrieved contexts
    print("\nRETRIEVED CONTEXTS:")
    print_separator("-")
    chunk_sources = result.get("chunk_sources", ["unknown source"] * len(result["contexts"]))
    for i, context in enumerate(result["contexts"], 1):
        print(f"Context {i} (from {chunk_sources[i-1]}):")
        # Print the context with a clear separator
        print("\n" + context + "\n")
        print("-" * 40)  # Smaller separator between contexts
    
    if verbose:
        print("\nPLANNING ANALYSIS:")
        print_separator("-")
        print(result["plan"]["planning_analysis"])
        
        print("\nSEARCH TERMS:")
        print_separator("-")
        for term in result["plan"]["search_terms"]:
            print(f"- {term}")
        
        if result["tool_usage"]:
            print("\nTOOL USAGE:")
            print_separator("-")
            print(f"Tool: {result['tool_usage']['tool']}")
            print(f"Parameters: {json.dumps(result['tool_usage']['parameters'], indent=2)}")
            print("\nTool Result:")
            print(result["tool_result"])
    
    print("\nRESPONSE:")
    print_separator("-")
    print(result["response"])
    print_separator()


def main():
    """Main function to run the test"""
    args = parse_args()
    
    print(f"Initializing Terraria AI Guide test with generation model: {args.model}")
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    
    # Set up vector store - only pass the persist path, not the model
    persist_path = "model/data/chroma_db"
    collection = setup_vector_store(persist_path)
    
    if collection is None:
        return
    
    # Set up query processor
    processor = QueryProcessor(
        vector_collection=collection,
        model=args.model,  # Use the specified model for generation
        embedding_model=EMBEDDING_MODEL,  # Always use consistent embeddings
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_results=3
    )
    
    print("Terraria AI Guide initialized. Type 'exit' to quit.")
    print_separator()
    
    # Main interaction loop
    while True:
        try:
            query = input("\nAsk a question about Terraria: ")
            
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            
            if not query.strip():
                continue
            
            print("Processing query...")
            result = processor.process_query(query)
            print_result(result, args.verbose)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                print("\nDetailed error information:")
                traceback.print_exc()

if __name__ == "__main__":
    main()