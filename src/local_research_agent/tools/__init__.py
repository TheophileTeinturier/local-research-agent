"""Tools: definitions (`Tool`), results (`ToolResult`), the `ToolRegistry`, and built-in tools."""

from typing import Any

from .arxiv import ArxivClient, make_arxiv_tools
from .base import Tool, ToolError, ToolResult
from .calculator import calculator_tool
from .current_date import current_date_tool
from .registry import DEFAULT_MAX_OUTPUT_CHARS, ToolRegistry


def build_default_registry(
    *, arxiv_client: ArxivClient | None = None, **kwargs: Any
) -> ToolRegistry:
    """Registry with every built-in tool. kwargs are forwarded to `ToolRegistry`.

    `arxiv_client` defaults to `ArxivClient()`, which opens no connection until the
    first arXiv request. Pass one built on a fake transport for offline runs.
    """
    client = arxiv_client if arxiv_client is not None else ArxivClient()
    return ToolRegistry([calculator_tool, current_date_tool, *make_arxiv_tools(client)], **kwargs)


__all__ = [
    "DEFAULT_MAX_OUTPUT_CHARS",
    "ArxivClient",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "calculator_tool",
    "current_date_tool",
]
