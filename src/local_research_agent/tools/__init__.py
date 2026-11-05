"""Tools: definitions (`Tool`), results (`ToolResult`), the `ToolRegistry`, and built-in tools."""

from typing import Any

from .base import Tool, ToolError, ToolResult
from .calculator import calculator_tool
from .current_date import current_date_tool
from .registry import DEFAULT_MAX_OUTPUT_CHARS, ToolRegistry


def build_default_registry(**kwargs: Any) -> ToolRegistry:
    """Registry with every built-in tool. kwargs are forwarded to `ToolRegistry`."""
    return ToolRegistry([calculator_tool, current_date_tool], **kwargs)


__all__ = [
    "DEFAULT_MAX_OUTPUT_CHARS",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "calculator_tool",
    "current_date_tool",
]
