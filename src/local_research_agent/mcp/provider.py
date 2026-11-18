"""`McpToolProvider`: the tools of an MCP server, seen as an ordinary `ToolProvider`.

The agent loop and the registry do not know the tools are remote: schemas have the exact
format of local ones, and every outcome (success, tool error, timeout, dead server) is a
`ToolResult`. `execute` never raises for an expected failure.

What crosses MCP and what does not (see docs/REVIEW_GUIDE.md, "Decisions taken"):
- content and ok/error: always (text content + `isError`);
- `error_type`: through the result's `_meta` (our server sets it; another server would not,
  and the generic "tool_error" is used then);
- `truncated` / `original_length`: NOT sent; recomputed from the cap's marker in the text;
- `duration_ms`: the server's own timing is lost; the client registry measures the round trip
  (which is the latency the agent actually pays).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import mcp.types as types
from pydantic.json_schema import GenerateJsonSchema

from ..tools import ToolResult
from ..tools.providers import (
    RawArguments,
    ToolSchema,
    decode_arguments,
    error_result,
    tool_schema,
    unknown_tool_result,
)
from ..tools.registry import truncation_info
from .client import (
    DEFAULT_CALL_TIMEOUT_S,
    McpClientBridge,
    McpError,
    McpServerDied,
    McpTimeout,
)
from .server import ERROR_TYPE_META_KEY

logger = logging.getLogger(__name__)

# How much earlier than the agent loop's tool timeout the MCP call gives up (seconds).
TIMEOUT_MARGIN_S = 0.5


def call_timeout_for(tool_timeout_s: float) -> float:
    """The MCP call timeout to use under the agent loop's `tool_timeout_s`.

    Slightly SHORTER, so the bridge times out first: it then cancels the request properly and
    reports an MCP timeout. With an equal or longer timeout, the loop would give up first and
    leave the request pending in the bridge until its own timeout.
    """
    return max(tool_timeout_s - TIMEOUT_MARGIN_S, tool_timeout_s / 2)


class McpToolProvider:
    """Tools served by the MCP server behind `bridge` (which must be started)."""

    def __init__(
        self, bridge: McpClientBridge, *, call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S
    ) -> None:
        self._bridge = bridge
        self.call_timeout_s = call_timeout_s
        # Listed once (the bridge lists at startup); the server's tool set does not change.
        self._schemas = [mcp_tool_to_schema(tool) for tool in bridge.list_tools()]
        self._names = [schema["function"]["name"] for schema in self._schemas]

    def schemas(self) -> list[ToolSchema]:
        return copy.deepcopy(self._schemas)  # callers may not mutate our cache

    def has(self, name: str) -> bool:
        return name in self._names

    def execute(self, name: str, raw_args: RawArguments) -> ToolResult:
        """Forward the call; map every MCP outcome to a `ToolResult`. Never raises."""
        if name not in self._names:
            return unknown_tool_result(name, self._names)
        # The ONLY client-side processing: MCP needs a JSON object, so raw JSON text is
        # decoded here (same helper, hence same error text, as the local provider). Field
        # validation is the server's job: there is no Pydantic model on this side.
        arguments = decode_arguments(raw_args)
        if isinstance(arguments, ToolResult):
            return arguments
        try:
            result = self._bridge.call_tool(name, dict(arguments), timeout_s=self.call_timeout_s)
        except McpTimeout:
            return error_result(
                "timeout",
                f"Tool {name!r} timed out after {self.call_timeout_s:g} s.",
            )
        except McpServerDied:
            return error_result(
                "mcp_server_died",
                "The tool server has stopped, so no tool can be used for the rest of this "
                "run. Answer with the information you already have, and say that tools "
                "became unavailable.",
            )
        except McpError as exc:
            return error_result("mcp_error", f"Tool {name!r} failed (MCP error: {exc}).")
        return from_call_tool_result(result)


def mcp_tool_to_schema(tool: types.Tool) -> ToolSchema:
    """An MCP tool definition in the LLM's format (same builder as local tools).

    Key ORDER is restored too. JSON objects are unordered, and the SDK's versioned wire
    model emits its declared fields (`type`, `properties`, `required`) before the others, so
    `inputSchema` arrives reordered. But the LLM sees the schema as text, and the local
    schemas come out of Pydantic in its canonical order (keys sorted, except inside
    `properties`, which keeps field order). Re-applying that same rule here gives back the
    exact local bytes (checked by the contract test).
    """
    parameters = GenerateJsonSchema().sort(dict(tool.input_schema))
    return tool_schema(tool.name, tool.description or "", parameters)


def from_call_tool_result(result: types.CallToolResult) -> ToolResult:
    """MCP result -> `ToolResult` (`isError` -> ok=False)."""
    parts = []
    for block in result.content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        else:  # our server only sends text; another server might not
            parts.append(f"[{block.type} content omitted]")
    content = "\n".join(parts)

    metadata: dict[str, Any] = {}
    if result.is_error:
        error_type = (result.meta or {}).get(ERROR_TYPE_META_KEY)
        metadata["error_type"] = error_type if isinstance(error_type, str) else "tool_error"
    info = truncation_info(content)
    if info is not None:
        metadata["truncated"] = True
        metadata["original_length"] = info[1]
    return ToolResult(ok=not result.is_error, content=content, metadata=metadata)
