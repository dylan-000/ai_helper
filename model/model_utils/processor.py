#!/usr/bin/env python3
"""
Processing pipeline for the AI Guide mod
This module implements a multi-step reasoning pipeline to process user queries
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import json

# Import local modules 
from vector_store import query_vector_store
from llm_utils import (
    format_rag_prompt, 
    format_tool_prompt,
    generate_response,
    parse_tool_usage
)
from tools import default_registry

class QueryProcessor:
    """
    Multi-step query processing pipeline
    
    This class implements a multi-step reasoning process:
    1. Query planning: Determine what information is needed and how to retrieve it
    2. Information retrieval: Get relevant context from the vector store
    3. Tool usage (if applicable): Use tools to get additional information
    4. Response generation: Generate the final response
    """
    
    def __init__(self, 
                vector_collection,
                model: str = "llama3.2:latest", 
                embedding_model: str = "nomic-embed-text",
                temperature: float = 0.7,
                max_tokens: int = 1024,
                n_results: int = 5,
                tool_registry = None):
        """
        Initialize the query processor
        
        Args:
            vector_collection: ChromaDB collection object for vector search
            model: Ollama model to use for text generation
            embedding_model: Ollama model to use for embeddings (should match vector store)
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            n_results: Number of chunks to retrieve from vector store
            tool_registry: Tool registry to use (default: default_registry)
        """
        self.vector_collection = vector_collection
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_results = n_results
        self.tool_registry = tool_registry or default_registry
    
    def plan_query(self, query: str) -> Dict[str, Any]:
        """
        First step: Plan how to process the query
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with planning information
        """
        # A simple planning prompt to analyze the query
        planning_prompt = f"""You are an AI assistant for Terraria. Analyze this query and determine:
1. What information is needed to answer it?
2. What specific search terms would help find this information?
3. Would any specialized tools be useful for answering this?

Be concrete and specific. Structure your thoughts step-by-step.

Query: {query}

Plan:"""
        
        plan = generate_response(
            prompt=planning_prompt,
            model=self.model,
            temperature=self.temperature * 0.7,  # Lower temperature for planning
            max_tokens=self.max_tokens // 2  # Use fewer tokens for planning
        )
        
        # Extract search terms from the plan
        search_terms = [query]  # Default to original query
        
        # Parse more specific search terms if possible
        if "search term" in plan.lower() or "search for" in plan.lower():
            lines = plan.split("\n")
            for line in lines:
                if "search term" in line.lower() or "search for" in line.lower():
                    # Extract what seems to be a search term
                    parts = line.split(":")
                    if len(parts) > 1:
                        term = parts[1].strip().strip('"').strip("'")
                        if term and term != query:
                            search_terms.append(term)
        
        return {
            "original_query": query,
            "search_terms": search_terms,
            "planning_analysis": plan,
            "use_tools": "tool" in plan.lower() or "specialized tool" in plan.lower()
        }
    
    def retrieve_information(self, plan: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Second step: Retrieve relevant information based on the plan
        
        Args:
            plan: Planning information from plan_query
            
        Returns:
            Tuple of (retrieved text chunks, source information)
        """
        all_chunks = []
        chunk_sources = []
        seen_chunks = set()
        
        # Query vector store for each search term
        for term in plan["search_terms"]:
            results = query_vector_store(
                collection=self.vector_collection,
                query=term,
                n_results=self.n_results
            )
            
            # Add unique chunks
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            
            for i, doc in enumerate(docs):
                if doc not in seen_chunks:
                    all_chunks.append(doc)
                    metadata = metadatas[i]
                    source = f"{metadata.get('source', 'unknown')} (chunk {metadata.get('chunk', 'unknown')})"
                    chunk_sources.append(source)
                    seen_chunks.add(doc)
                    
                    # Limit to 2x n_results to avoid overwhelming context
                    if len(all_chunks) >= self.n_results * 2:
                        break
            
            # If we have enough chunks, stop querying
            if len(all_chunks) >= self.n_results:
                break
        
        # Return at most n_results chunks and their sources
        return all_chunks[:self.n_results], chunk_sources[:self.n_results]
    
    def process_tools(self, query: str, contexts: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Third step: Process tools if applicable
        
        Args:
            query: User query string
            contexts: Retrieved text chunks
            
        Returns:
            Tuple of (tool response, tool usage info)
        """
        # Get tool definitions
        tool_defs = self.tool_registry.get_definitions()
        
        # Format prompt with tools
        prompt = format_tool_prompt(
            query=query,
            contexts=contexts,
            tools=tool_defs
        )
        
        # Generate response with tool usage
        response = generate_response(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Parse tool usage
        tool_usage = parse_tool_usage(response)
        
        # If a tool was used, execute it
        if tool_usage and "tool" in tool_usage and "parameters" in tool_usage:
            tool_name = tool_usage["tool"]
            parameters = tool_usage["parameters"]
            
            # Create a query function for the tool to use
            def query_func(search_query: str) -> str:
                results = query_vector_store(
                    collection=self.vector_collection,
                    query=search_query,
                    n_results=3
                )
                # Return just the documents as a string
                return "\n\n".join(results["documents"][0])
            
            # Execute the tool with the query function
            tool_result = self.tool_registry.execute_tool(
                tool_name=tool_name,
                parameters={**parameters, "query_func": query_func}
            )
            
            return tool_result, tool_usage
        
        return "", {}
    
    def generate_final_response(self, query: str, contexts: List[str], tool_result: str = "") -> str:
        """
        Final step: Generate the final response
        
        Args:
            query: User query string
            contexts: Retrieved text chunks
            tool_result: Result from tool execution (if any)
            
        Returns:
            Final response to the user
        """
        # If we have a tool result, add it to the context
        if tool_result:
            contexts = [tool_result] + contexts
        
        # Format prompt for final response
        prompt = format_rag_prompt(
            query=query,
            contexts=contexts
        )
        
        # Generate final response
        response = generate_response(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the entire pipeline
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with processing results and intermediate steps
        """
        # Step 1: Plan the query
        plan = self.plan_query(query)
        
        # Step 2: Retrieve information
        contexts, chunk_sources = self.retrieve_information(plan)
        
        # Step 3: Process tools if applicable
        tool_result = ""
        tool_usage = {}
        if plan["use_tools"]:
            tool_result, tool_usage = self.process_tools(query, contexts)
        
        # Step 4: Generate final response
        final_response = self.generate_final_response(query, contexts, tool_result)
        
        # Return all information for debugging and analysis
        return {
            "query": query,
            "plan": plan,
            "contexts": contexts,
            "chunk_sources": chunk_sources,
            "tool_usage": tool_usage,
            "tool_result": tool_result,
            "response": final_response
        } 