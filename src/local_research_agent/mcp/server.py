"""MCP server (stdio) exposing the tools of a `ToolRegistry`. Low-level SDK API, no FastMCP.

    python -m local_research_agent.mcp.server [--registry-factory module:function]
                                              [--log-level LEVEL]

`--registry-factory` names a zero-argument function returning the `ToolRegistry` to serve
(default: `build_default_registry`, the real tools). It is resolved with importlib, so tests
and the evaluation harness can serve a registry built on a fake arXiv transport, a frozen
clock or extra test tools WITHOUT any test code living in this module.

Single source of truth: `tools/list` returns, for each tool, the name, description and JSON
schema taken from `registry.schemas()`, i.e. exactly what the local registry sends to the
LLM. `tools/call` delegates to `registry.execute()`.

The server VALIDATES every call itself (Pydantic, inside `registry.execute`): it cannot trust
its client. Our own agent already sends well-formed calls, but any MCP client may connect
(a desktop app, an inspector, a buggy script), and the SDK's low-level server does not apply
the advertised `inputSchema` (v2 removed that). The server never raises on bad input: an
unknown tool, invalid arguments and tool exceptions all come back as `isError` results.

STDOUT IS SACRED. In stdio mode, stdout is the protocol channel: every byte on it must be a
JSON-RPC message, and one stray `print()` corrupts the stream (the client fails to parse the
line, and the request it belonged to never completes). So nothing here writes to stdout: logs
go to stderr (`configure_logging`), argparse errors go to stderr, and the tools never print
(tests/test_mcp_server.py checks it at the file-descriptor level). The SDK's `stdio_server`
adds defence in depth while serving (it points fd 1 at stderr and writes the protocol to a
private duplicate), but it cannot protect what runs before it starts: imports and the
registry factory.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import anyio
import anyio.to_thread
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .. import __version__
from ..observability import configure_logging, log_event
from ..tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

SERVER_NAME = "local-research-agent"
DEFAULT_REGISTRY_FACTORY = "local_research_agent.tools:build_default_registry"

# `_meta` keys (MCP reserves `_meta` for implementation data; keys carry a vendor prefix).
META_PREFIX = "local-research-agent/"
# On a tools/call error result: the `ToolResult.metadata["error_type"]` (e.g. "not_found"),
# which the text alone does not carry. Clients that do not know it simply ignore it.
ERROR_TYPE_META_KEY = META_PREFIX + "error_type"
# On the tools/list result: the server's PID, for diagnostics and the orphan-process tests.
SERVER_PID_META_KEY = META_PREFIX + "server_pid"

RegistryFactory = Callable[[], ToolRegistry]


# --- registry -> MCP (pure conversions) -----------------------------------------------------


def mcp_tools(registry: ToolRegistry) -> list[types.Tool]:
    """The registry's tools as MCP tool definitions, built from `registry.schemas()`."""
    return [
        types.Tool(
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            input_schema=schema["function"]["parameters"],
        )
        for schema in registry.schemas()
    ]


def to_call_tool_result(result: ToolResult) -> types.CallToolResult:
    """`ToolResult` -> MCP result: one text block, `isError` when the call failed."""
    meta = None
    if not result.ok and result.metadata.get("error_type"):
        meta = {ERROR_TYPE_META_KEY: result.metadata["error_type"]}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=result.content)],
        is_error=not result.ok,
        _meta=meta,
    )


async def call_tool(
    registry: ToolRegistry, name: str, arguments: Mapping[str, Any] | None
) -> types.CallToolResult:
    """Run one call off the event loop. Never raises for a bad call or a failing tool.

    `registry.execute` is synchronous (it may block on HTTP), so it runs in a worker thread:
    the event loop keeps reading stdin (pings, cancellations) meanwhile. If the client
    cancels (its timeout), the handler stops waiting, but the thread cannot be killed and
    runs the tool to completion; its result is then discarded.
    """
    try:
        result = await anyio.to_thread.run_sync(
            registry.execute, name, arguments, abandon_on_cancel=True
        )
    except Exception as exc:  # registry.execute never raises; this is belt and braces
        logger.exception("tools/call %s failed", name)
        result = ToolResult(
            ok=False,
            content=f"Error: Tool {name!r} failed: {type(exc).__name__}: {exc}",
            metadata={"error_type": "exception"},
        )
    return to_call_tool_result(result)


def build_server(registry: ToolRegistry) -> Server[Any]:
    """A low-level `Server` with exactly two handlers: tools/list and tools/call."""
    tools = mcp_tools(registry)  # computed once: the registry does not change while serving

    async def handle_list_tools(
        ctx: Any, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools, _meta={SERVER_PID_META_KEY: os.getpid()})

    async def handle_call_tool(
        ctx: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        return await call_tool(registry, params.name, params.arguments)

    return Server(
        SERVER_NAME,
        version=__version__,
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )


async def serve_stdio(registry: ToolRegistry) -> None:
    """Serve until the client closes our stdin (then `server.run` returns)."""
    server = build_server(registry)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


# --- entry point ----------------------------------------------------------------------------


def load_registry_factory(spec: str) -> RegistryFactory:
    """Resolve 'package.module:function'. Raises `ValueError` with a readable message."""
    module_name, sep, attribute = spec.partition(":")
    if not sep or not module_name or not attribute:
        raise ValueError(f"registry factory {spec!r} must look like 'package.module:function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import {module_name!r} for registry factory: {exc}") from exc
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"{module_name!r} has no callable {attribute!r}")
    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_research_agent.mcp.server",
        description="Serve the agent's tools over MCP (stdio). Logs go to stderr.",
    )
    parser.add_argument(
        "--registry-factory",
        default=DEFAULT_REGISTRY_FACTORY,
        help="'module:function' returning the ToolRegistry to serve (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "WARNING"),
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the registry, then serve. Exit code 1 if the registry cannot be built."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)  # stderr only
    try:
        registry = load_registry_factory(args.registry_factory)()
        if not isinstance(registry, ToolRegistry):
            raise TypeError(f"factory returned {type(registry).__name__}, not a ToolRegistry")
    except Exception as exc:
        # The client sees the pipe close during `initialize` and reports a startup failure;
        # the reason is here, on stderr.
        log_event(
            logger,
            "mcp_server_startup_failed",
            level=logging.ERROR,
            factory=args.registry_factory,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1
    log_event(
        logger,
        "mcp_server_started",
        pid=os.getpid(),
        factory=args.registry_factory,
        tools=registry.names,
    )
    anyio.run(serve_stdio, registry)
    log_event(logger, "mcp_server_stopped", pid=os.getpid())
    return 0


if __name__ == "__main__":
    sys.exit(main())
