#!/usr/bin/env python3
"""
Tool utilities for the AI Guide mod
This module defines tools that the LLM can use to perform specific tasks
"""

import json
from typing import List, Dict, Any, Callable, Optional

# Tool definitions with JSON schema format
TOOL_DEFINITIONS = [
    {
        "name": "item_lookup",
        "description": "Look up detailed information about a specific item in Terraria.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the item to look up"
                }
            },
            "required": ["item_name"]
        }
    },
    {
        "name": "boss_info",
        "description": "Get information about a specific boss, including strategies and requirements.",
        "parameters": {
            "type": "object",
            "properties": {
                "boss_name": {
                    "type": "string",
                    "description": "The name of the boss to look up"
                }
            },
            "required": ["boss_name"]
        }
    },
    {
        "name": "crafting_recipe",
        "description": "Look up the crafting recipe for a specific item.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the item to get the recipe for"
                }
            },
            "required": ["item_name"]
        }
    },
    {
        "name": "progression_check",
        "description": "Get advice on what to do next based on current game progress.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_stage": {
                    "type": "string",
                    "description": "The current game stage (e.g., 'early pre-hardmode', 'post-plantera')"
                },
                "defeated_bosses": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of bosses that have been defeated"
                },
                "class_type": {
                    "type": "string",
                    "description": "The player's class (melee, ranged, magic, summoner, mixed)"
                }
            },
            "required": ["current_stage"]
        }
    }
]

class ToolRegistry:
    """Registry for tool implementations"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.definitions: List[Dict[str, Any]] = TOOL_DEFINITIONS
    
    def register(self, name: str, func: Callable):
        """
        Register a tool implementation
        
        Args:
            name: Tool name (must match a name in TOOL_DEFINITIONS)
            func: Function that implements the tool
        """
        self.tools[name] = func
    
    def get_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all tool definitions
        
        Returns:
            List of tool definitions
        """
        return self.definitions
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Execute a registered tool
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Result of the tool execution as a string
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            return self.tools[tool_name](**parameters)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

# Create a default tool registry
default_registry = ToolRegistry()

# Implement example tool functions

def item_lookup(item_name: str, query_func: Optional[Callable] = None) -> str:
    """
    Look up information about a specific item
    
    Args:
        item_name: Name of the item to look up
        query_func: Optional function to query the vector store
        
    Returns:
        Information about the item
    """
    # If a query function is provided, use it to get specific information
    if query_func:
        return query_func(f"Information about the item {item_name} in Terraria")
    
    return f"Looking up information about '{item_name}'... [This would use the vector store to find detailed information]"

def boss_info(boss_name: str, query_func: Optional[Callable] = None) -> str:
    """
    Get information about a specific boss
    
    Args:
        boss_name: Name of the boss to look up
        query_func: Optional function to query the vector store
        
    Returns:
        Information about the boss
    """
    if query_func:
        return query_func(f"Information about the boss {boss_name} in Terraria, including strategies and requirements")
    
    return f"Looking up information about boss '{boss_name}'... [This would use the vector store to find detailed information]"

def crafting_recipe(item_name: str, query_func: Optional[Callable] = None) -> str:
    """
    Look up the crafting recipe for a specific item
    
    Args:
        item_name: Name of the item to get the recipe for
        query_func: Optional function to query the vector store
        
    Returns:
        Crafting recipe information
    """
    if query_func:
        return query_func(f"Crafting recipe for {item_name} in Terraria")
    
    return f"Looking up crafting recipe for '{item_name}'... [This would use the vector store to find the recipe]"

def progression_check(current_stage: str, defeated_bosses: Optional[List[str]] = None, class_type: Optional[str] = None, query_func: Optional[Callable] = None) -> str:
    """
    Get advice on what to do next based on current game progress
    
    Args:
        current_stage: The current game stage
        defeated_bosses: List of bosses that have been defeated
        class_type: The player's class
        query_func: Optional function to query the vector store
        
    Returns:
        Progression advice
    """
    if query_func:
        query = f"What to do at {current_stage} stage in Terraria"
        if defeated_bosses:
            query += f" after defeating {', '.join(defeated_bosses)}"
        if class_type:
            query += f" as a {class_type} player"
        return query_func(query)
    
    defeated = ", ".join(defeated_bosses) if defeated_bosses else "none"
    class_info = f" as a {class_type} player" if class_type else ""
    
    return f"Checking progression for '{current_stage}' stage, defeated bosses: {defeated}{class_info}... [This would use the vector store to find progression advice]"

# Register the example tool implementations
default_registry.register("item_lookup", item_lookup)
default_registry.register("boss_info", boss_info)
default_registry.register("crafting_recipe", crafting_recipe)
default_registry.register("progression_check", progression_check) 