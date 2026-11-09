"""LLM layer: the `LLMClient` abstraction and its adapters (Ollama for now)."""

from .base import (
    ChatOptions,
    LLMClient,
    LLMError,
    LLMMetrics,
    LLMResponse,
    Message,
    ToolCall,
    ToolSchema,
    system_message,
    tool_result_message,
    user_message,
)
from .ollama import OllamaClient

__all__ = [
    "ChatOptions",
    "LLMClient",
    "LLMError",
    "LLMMetrics",
    "LLMResponse",
    "Message",
    "OllamaClient",
    "ToolCall",
    "ToolSchema",
    "system_message",
    "tool_result_message",
    "user_message",
]
