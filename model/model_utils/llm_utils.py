#!/usr/bin/env python3
"""
LLM utilities for the AI Guide mod
This module handles the interaction with the Ollama LLM, including prompt creation and response generation
"""

import json
from typing import List, Dict, Any, Optional, Union, Callable

# Import ollama here to avoid potential circular imports
import ollama

# Default system prompt for the AI guide
DEFAULT_SYSTEM_PROMPT = """You are an AI Guide for the game Terraria. Your purpose is to help players progress through the game by providing useful information about:
- Game progression and what to do next
- Crafting recipes and item information
- Boss strategies and preparation tips
- Class builds and equipment recommendations
- Farm designs and resource gathering

Always be concise but thorough, providing step-by-step advice when appropriate.
Never make up information - if you're unsure, admit it and suggest where the player might find the answer.

Remember that you're speaking directly to a player who is playing the game right now and needs your help.
"""

def format_rag_prompt(query: str, contexts: List[str], system_prompt: Optional[str] = None) -> str:
    """
    Format a prompt for the LLM using retrieved contexts
    
    Args:
        query: User query
        contexts: Retrieved text chunks from vector store
        system_prompt: Optional system prompt to use (default: DEFAULT_SYSTEM_PROMPT)
        
    Returns:
        Formatted prompt string
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    context_text = "\n\n---\n\n".join([f"CONTEXT {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = f"""{system_prompt}

Here are some relevant pieces of information from the Terraria guide database:

{context_text}

USER QUESTION: {query}

Provide a helpful, accurate response based on the information above. If the information doesn't contain the answer,
say so clearly rather than making up information."""

    return prompt

def format_tool_prompt(query: str, contexts: List[str], tools: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
    """
    Format a prompt for the LLM with tool calling capabilities
    
    Args:
        query: User query
        contexts: Retrieved text chunks from vector store
        tools: List of tool definitions (name, description, parameters)
        system_prompt: Optional system prompt to use (default: DEFAULT_SYSTEM_PROMPT)
        
    Returns:
        Formatted prompt string with tool definitions
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    context_text = "\n\n---\n\n".join([f"CONTEXT {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    tools_json = json.dumps(tools, indent=2)
    
    prompt = f"""{system_prompt}

Here are some relevant pieces of information from the Terraria guide database:

{context_text}

You have access to the following tools:
{tools_json}

To use a tool, respond with a JSON object that contains the tool name and parameters. For example:
{{
  "tool": "item_lookup",
  "parameters": {{
    "item_name": "Zenith"
  }}
}}

USER QUESTION: {query}

If a tool is appropriate to help answer this question, use it. Otherwise, provide a helpful response based on the context provided."""

    return prompt

def generate_response(prompt: str, model: str = "llama3.2:latest", temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """
    Generate a response from the LLM
    
    Args:
        prompt: Formatted prompt string
        model: Ollama model to use
        temperature: Temperature parameter for generation (higher = more creative)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated response string
    """
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    )
    
    return response["response"]

def parse_tool_usage(response: str) -> Dict[str, Any]:
    """
    Parse a response to extract tool usage if present
    
    Args:
        response: Generated response from the LLM
        
    Returns:
        Dictionary containing tool name and parameters, or empty dict if no tool usage
    """
    try:
        # Look for JSON-like structures in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx+1]
            tool_call = json.loads(json_str)
            
            if "tool" in tool_call and "parameters" in tool_call:
                return tool_call
    except:
        pass
    
    return {} 