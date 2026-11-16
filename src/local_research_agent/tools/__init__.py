"""Tools: definitions (`Tool`), results (`ToolResult`), the `ToolRegistry`, and built-in tools."""

from typing import Any

import httpx

from .arxiv import ArxivClient, make_arxiv_tools
from .arxiv.testing import offline_arxiv_client
from .base import Tool, ToolError, ToolResult
from .calculator import calculator_tool
from .current_date import DateClock, current_date_tool, make_current_date_tool
from .registry import DEFAULT_MAX_OUTPUT_CHARS, ToolRegistry


def build_default_registry(
    *, arxiv_client: ArxivClient | None = None, clock: DateClock | None = None, **kwargs: Any
) -> ToolRegistry:
    """Registry with every built-in tool. kwargs are forwarded to `ToolRegistry`.

    `arxiv_client` defaults to `ArxivClient()`, which opens no connection until the
    first arXiv request. Pass one built on a fake transport for offline runs.
    `clock` is what `current_date` reads (default: the real local time).
    """
    client = arxiv_client if arxiv_client is not None else ArxivClient()
    date_tool = current_date_tool if clock is None else make_current_date_tool(clock)
    return ToolRegistry([calculator_tool, date_tool, *make_arxiv_tools(client)], **kwargs)


def build_registry(
    *,
    arxiv_transport: httpx.BaseTransport | None = None,
    clock: DateClock | None = None,
    **kwargs: Any,
) -> ToolRegistry:
    """`build_default_registry` with injection points for offline runs (the eval harness).

    `arxiv_transport`: the arXiv tools send their HTTP requests through it, on a client that
    never sleeps (no real API, so no rate limit to respect). None = the real arXiv API.
    `clock`: frozen "now" for `current_date`. With both None, this is the default registry.
    """
    client = None if arxiv_transport is None else offline_arxiv_client(arxiv_transport)
    return build_default_registry(arxiv_client=client, clock=clock, **kwargs)


__all__ = [
    "DEFAULT_MAX_OUTPUT_CHARS",
    "ArxivClient",
    "DateClock",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "build_registry",
    "calculator_tool",
    "current_date_tool",
    "make_current_date_tool",
]
