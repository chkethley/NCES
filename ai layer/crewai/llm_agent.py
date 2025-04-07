"""
LLM-powered CrewAI Agents

This module provides LLM-powered agents for CrewAI.
"""

import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from uuid import uuid4

from ..utils.logging import get_logger
from ..llm import LLMRequest, LLMResponse, get_llm_manager
from . import CrewMember, Tool

logger = get_logger(__name__)

@dataclass
class Message:
    """Message in a conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

@dataclass
class Conversation:
    """Conversation history for an agent."""
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 100
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        
        # Trim if needed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        self.add_message(Message(role="system", content=content))
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.add_message(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.add_message(Message(role="assistant", content=content))
    
    def add_tool_message(self, content: str, name: str, tool_call_id: str) -> None:
        """Add a tool message to the conversation."""
        self.add_message(Message(
            role="tool",
            content=content,
            name=name,
            tool_call_id=tool_call_id
        ))
    
    def get_formatted_messages(self) -> List[Dict[str, Any]]:
        """Get formatted messages for LLM input."""
        formatted = []
        
        for msg in self.messages:
            message = {"role": msg.role, "content": msg.content}
            
            if msg.name:
                message["name"] = msg.name
            
            if msg.tool_calls:
                message["tool_calls"] = msg.tool_calls
            
            if msg.tool_call_id:
                message["tool_call_id"] = msg.tool_call_id
            
            formatted.append(message)
        
        return formatted
    
    def clear(self) -> None:
        """Clear the conversation."""
        self.messages = []

class LLMAgent(CrewMember):
    """
    LLM-powered agent for CrewAI.
    
    This class extends CrewMember to use an LLM for reasoning and decision making.
    """
    
    def __init__(self, name: str, role: str, tools: List[Tool] = None, 
                 model_name: str = None, memory_size: int = 100,
                 system_message: str = None):
        """
        Initialize an LLM agent.
        
        Args:
            name: Agent name
            role: Agent role
            tools: List of tools
            model_name: Name of the LLM model to use
            memory_size: Size of conversation memory
            system_message: System message for the agent
        """
        super().__init__(name=name, role=role, tools=tools or [], memory_size=memory_size)
        
        self.model_name = model_name
        self.conversation = Conversation(max_messages=memory_size)
        
        # Add system message
        if system_message:
            self.conversation.add_system_message(system_message)
        else:
            # Default system message
            default_system_message = (
                f"You are {name}, an AI assistant with the role: {role}. "
                f"You have access to the following tools: "
                f"{', '.join(t.name for t in self.tools)}. "
                f"Always think step by step and use tools when appropriate."
            )
            self.conversation.add_system_message(default_system_message)
    
    async def process_message(self, message: str) -> str:
        """
        Process a message and generate a response.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        # Add user message to conversation
        self.conversation.add_user_message(message)
        
        # Get LLM manager
        llm_manager = get_llm_manager()
        
        # Format conversation for LLM
        formatted_messages = self.conversation.get_formatted_messages()
        
        # Create tool descriptions for the LLM
        tool_descriptions = []
        for tool in self.tools:
            tool_desc = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param: {"type": "string", "description": desc}
                            for param, desc in tool.parameters.items()
                        },
                        "required": list(tool.parameters.keys())
                    }
                }
            }
            tool_descriptions.append(tool_desc)
        
        # Create prompt
        prompt = json.dumps(formatted_messages)
        
        # Create LLM request
        request = LLMRequest(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            extra_params={
                "tools": tool_descriptions,
                "response_format": {"type": "json_object"}
            }
        )
        
        # Generate response
        response = await llm_manager.generate(request, model_name=self.model_name)
        
        # Parse response
        try:
            response_data = json.loads(response.text)
            
            # Check for tool calls
            if "tool_calls" in response_data:
                tool_calls = response_data["tool_calls"]
                
                # Process each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    tool_call_id = tool_call["id"]
                    
                    # Add assistant message with tool call
                    self.conversation.add_message(Message(
                        role="assistant",
                        content="",
                        tool_calls=[{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args)
                            }
                        }]
                    ))
                    
                    # Execute tool
                    try:
                        tool_result = await self.use_tool(tool_name, **tool_args)
                        
                        # Add tool result to conversation
                        self.conversation.add_tool_message(
                            content=str(tool_result),
                            name=tool_name,
                            tool_call_id=tool_call_id
                        )
                    except Exception as e:
                        # Add error message
                        self.conversation.add_tool_message(
                            content=f"Error: {str(e)}",
                            name=tool_name,
                            tool_call_id=tool_call_id
                        )
                
                # Generate final response after tool calls
                final_request = LLMRequest(
                    prompt=json.dumps(self.conversation.get_formatted_messages()),
                    max_tokens=1024,
                    temperature=0.7,
                    top_p=0.9
                )
                
                final_response = await llm_manager.generate(final_request, model_name=self.model_name)
                
                # Add assistant message
                self.conversation.add_assistant_message(final_response.text)
                
                return final_response.text
            else:
                # Add assistant message
                content = response_data.get("content", response.text)
                self.conversation.add_assistant_message(content)
                
                return content
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            
            # Add assistant message
            self.conversation.add_assistant_message(response.text)
            
            return response.text
    
    async def execute_task(self, task: str) -> str:
        """
        Execute a task.
        
        Args:
            task: Task description
            
        Returns:
            Task result
        """
        return await self.process_message(task)
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        # Keep system message
        system_messages = [msg for msg in self.conversation.messages if msg.role == "system"]
        self.conversation.clear()
        
        # Restore system messages
        for msg in system_messages:
            self.conversation.add_message(msg)
