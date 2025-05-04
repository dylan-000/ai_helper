#!/usr/bin/env python3
"""
Vector store utilities for the AI Guide mod
This module handles the creation, loading, and querying of the ChromaDB vector store
"""

import os
import glob
from typing import List, Dict, Any, Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

class OllamaEmbeddingFunction:
    """Custom embedding function that uses Ollama for embeddings"""
    
    def __init__(self, model_name="nomic-embed-text", max_retries=3, retry_delay=2):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Import here to avoid circular imports
        import ollama
        import time
        self.ollama = ollama
        self.time = time
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Ollama"""
        # Process each text individually as Ollama expects a single string at a time
        embeddings = []
        
        for text in input:
            # Try with retries for robustness
            for attempt in range(self.max_retries):
                try:
                    response = self.ollama.embeddings(model=self.model_name, prompt=text)
                    embeddings.append(response['embedding'])
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(f"Embedding attempt {attempt+1} failed: {e}. Retrying in {self.retry_delay} seconds...")
                        self.time.sleep(self.retry_delay)
                    else:
                        print(f"All embedding attempts failed for text: {text[:50]}...")
                        raise
        
        return embeddings

def load_documents(data_dir: str, process_large_files: bool = True) -> Dict[str, str]:
    """
    Load text documents from a directory
    
    Args:
        data_dir: Path to directory containing text documents
        process_large_files: Whether to process large files
        
    Returns:
        Dictionary mapping filenames to document contents
    """
    documents = {}
    large_file_threshold = 50 * 1024 * 1024  # 50MB threshold for "large" files
    
    # Check all text files
    for file_path in glob.glob(os.path.join(data_dir, "*.txt")):
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        if file_size > large_file_threshold:
            print(f"Processing large text file: {file_name} ({file_size / (1024*1024):.2f} MB)")
            
            try:
                # For large text files, read in chunks and split into multiple documents
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Estimate reasonable chunk size based on file size
                    chunk_size = 5 * 1024 * 1024  # 5MB chunks
                    chunk_num = 0
                    
                    while True:
                        content = file.read(chunk_size)
                        if not content:
                            break
                            
                        doc_name = f"{file_name}_part_{chunk_num}"
                        documents[doc_name] = content
                        chunk_num += 1
                        
                print(f"  Split into {chunk_num} chunks")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents[file_name] = content
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Check for JSON files that might contain text data
    for file_path in glob.glob(os.path.join(data_dir, "*.json")):
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        if file_size > large_file_threshold:
            print(f"Processing large JSON file: {file_name} ({file_size / (1024*1024):.2f} MB)")
            
            try:
                # Process large JSON files by streaming and extracting key text
                import json
                chunk_num = 0
                
                # Use streaming JSON processing for large files
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Simple streaming for JSON objects - extract text from keys
                    buffer = ""
                    for line in f:
                        buffer += line
                        if len(buffer) > 1024 * 1024:  # Process each ~1MB
                            doc_name = f"{file_name}_stream_{chunk_num}"
                            documents[doc_name] = buffer
                            buffer = ""
                            chunk_num += 1
                    
                    # Add any remaining content
                    if buffer:
                        doc_name = f"{file_name}_stream_{chunk_num}"
                        documents[doc_name] = buffer
                        chunk_num += 1
                
                print(f"  Processed into {chunk_num} chunks")
            except Exception as e:
                print(f"Error processing large JSON file {file_path}: {e}")
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents[file_name] = content
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(documents)} documents from {data_dir}")
    return documents

def chunk_documents(documents: Dict[str, str], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split documents into smaller chunks for embedding
    
    Args:
        documents: Dictionary mapping document names to content
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of document chunks with metadata
    """
    chunked_documents = []
    
    # Create the chunker with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    for doc_name, content in documents.items():
        # Apply the chunker to the document text
        chunks = text_splitter.split_text(content)
        
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "id": f"{doc_name}_chunk_{i}",
                "text": chunk,
                "metadata": {"source": doc_name, "chunk": i}
            })
    
    print(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
    return chunked_documents

def create_vector_store(chunks: List[Dict[str, Any]], 
                      collection_name: str = "terraria_guide", 
                      embedding_model: str = "nomic-embed-text",
                      persist_path: Optional[str] = "model/data/chroma_db",
                      batch_size: int = 5000) -> chromadb.Collection:
    """
    Create a new ChromaDB collection with document chunks
    
    Args:
        chunks: List of document chunks to add to the collection
        collection_name: Name of the collection to create
        embedding_model: Ollama model to use for embeddings
        persist_path: Path to save the ChromaDB collection (or None for in-memory)
        batch_size: Maximum number of chunks to add at once
        
    Returns:
        ChromaDB collection object
    """
    # Initialize ChromaDB client based on whether persistence is requested
    if persist_path:
        os.makedirs(persist_path, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_path)
    else:
        client = chromadb.Client()
    
    # Create embedding function
    embedding_function = OllamaEmbeddingFunction(model_name=embedding_model)
    print(f"Using Ollama for embeddings with model: {embedding_model}")
    
    # Create or get collection
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    # Add documents to collection in batches to avoid exceeding max batch size
    total_chunks = len(chunks)
    print(f"Adding {total_chunks} chunks to collection in batches of {batch_size}...")
    
    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch = chunks[i:batch_end]
        
        print(f"Adding batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}: chunks {i} to {batch_end-1}")
        
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch]
        )
    
    print(f"Added {total_chunks} chunks to ChromaDB collection '{collection_name}'")
    return collection

def load_vector_store(collection_name: str = "terraria_guide", 
                    embedding_model: str = "nomic-embed-text",
                    persist_path: str = "model/data/chroma_db") -> chromadb.Collection:
    """
    Load an existing ChromaDB collection
    
    Args:
        collection_name: Name of the collection to load
        embedding_model: Ollama model to use for embeddings
        persist_path: Path where the ChromaDB collection is saved
        
    Returns:
        ChromaDB collection object
    """
    # Initialize ChromaDB persistent client
    client = chromadb.PersistentClient(path=persist_path)
    
    # Create embedding function
    embedding_function = OllamaEmbeddingFunction(model_name=embedding_model)
    
    # Get collection
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    print(f"Loaded ChromaDB collection '{collection_name}' with {collection.count()} chunks")
    return collection

def query_vector_store(collection: chromadb.Collection, query: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Query the vector store for relevant chunks
    
    Args:
        collection: ChromaDB collection to query
        query: User query string
        n_results: Number of results to return
        
    Returns:
        Dictionary with query results including documents and metadata
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results

def get_embedding_function(model_name: str = "nomic-embed-text") -> Any:
    """
    Create an embedding function with the specified model
    
    Args:
        model_name: Ollama model to use for embeddings
        
    Returns:
        Embedding function object
    """
    return OllamaEmbeddingFunction(model_name=model_name) 