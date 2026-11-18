"""Tool providers: where tool calls are actually executed.

A provider owns a set of tools and answers three questions: which tools (`schemas`), do you
have this one (`has`), run it (`execute`). The `ToolRegistry` aggregates providers and adds
what must hold whatever the provider: routing, the output cap, timing, logging, and a last
safety net that turns any provider exception into an observation.

- `LocalToolProvider`: `Tool` objects run in this process (Pydantic validation, execution,
  every failure turned into `ToolResult(ok=False)`).
- `McpToolProvider` (`local_research_agent.mcp.provider`): the same tools, run by an MCP
  server in a subprocess.

The helpers at the bottom (`tool_schema`, `decode_arguments`, `error_result`) are shared by
every provider, so a given input produces the same schema and the same error text locally and
through MCP.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from .base import Tool, ToolError, ToolResult

logger = logging.getLogger(__name__)

# One tool definition in Ollama's `tools` request format (the OpenAI function format).
# Same alias as `llm.base.ToolSchema`; redefined here so `tools` does not depend on `llm`.
ToolSchema = dict[str, Any]

# What the model sent as arguments: a decoded object, raw JSON text, or nothing.
RawArguments = Mapping[str, Any] | str | None


@runtime_checkable
class ToolProvider(Protocol):
    """A source of tools. `execute` must return a `ToolResult` for every expected failure."""

    def schemas(self) -> list[ToolSchema]:
        """The provider's tools, in the format sent to the LLM. Must not change after
        the provider is added to a registry (names are checked for duplicates then)."""
        ...

    def has(self, name: str) -> bool: ...

    def execute(self, name: str, raw_args: RawArguments) -> ToolResult: ...


class LocalToolProvider:
    """Tools defined with `Tool` and executed in this process."""

    def __init__(self, tools: Iterable[Tool[Any]] = ()) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def schemas(self) -> list[ToolSchema]:
        return [
            tool_schema(tool.name, tool.description, tool.parameters_schema())
            for tool in self._tools.values()
        ]

    def execute(self, name: str, raw_args: RawArguments) -> ToolResult:
        """Validate and run one call. Never raises, except for Ctrl-C and SystemExit."""
        tool = self._tools.get(name)
        if tool is None:
            return unknown_tool_result(name, self.names)

        args_obj = decode_arguments(raw_args)
        if isinstance(args_obj, ToolResult):
            return args_obj

        # Pydantic is the gatekeeper: the tool function only ever receives a valid model
        # instance. The error message lists every faulty field and repeats the schema, so the
        # model has what it needs to fix its next call.
        try:
            args = tool.args_model.model_validate(args_obj)
        except ValidationError as exc:
            schema = json.dumps(tool.parameters_schema(), separators=(",", ":"))
            return error_result(
                "invalid_arguments",
                f"Invalid arguments for {name!r}: {_format_validation_error(exc)}. "
                f"Expected JSON schema: {schema}",
            )

        try:
            output = tool.function(args)
        except ToolError as exc:
            return error_result(exc.error_type, str(exc))
        except Exception as exc:
            # Deliberately broad: a buggy tool must not crash the agent. The model sees a short
            # error; the traceback goes to the logs for the developer.
            logger.exception("tool %s raised", name, extra={"fields": {"tool": name}})
            return error_result("exception", f"Tool {name!r} failed: {type(exc).__name__}: {exc}")

        return ToolResult(ok=True, content=_to_text(output))


# --- helpers shared by every provider ---------------------------------------------------------


def tool_schema(name: str, description: str, parameters: dict[str, Any]) -> ToolSchema:
    """THE place that builds a tool definition for the LLM (local and MCP tools alike)."""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def decode_arguments(raw_args: RawArguments) -> Mapping[str, Any] | ToolResult:
    """Return the argument object, or an error ToolResult if it is not a JSON object."""
    if raw_args is None:
        return {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as exc:
            return error_result("invalid_arguments", f"Arguments are not valid JSON: {exc.msg}.")
    if not isinstance(raw_args, Mapping):
        return error_result(
            "invalid_arguments",
            f"Arguments must be a JSON object, got {type(raw_args).__name__}.",
        )
    return raw_args


def error_result(error_type: str, message: str) -> ToolResult:
    """An error observation. It always starts with "Error:" so the model can't mistake it
    for data; `error_type` lets logs and evaluations tell failures apart."""
    return ToolResult(ok=False, content=f"Error: {message}", metadata={"error_type": error_type})


def unknown_tool_result(name: str, available: Iterable[str]) -> ToolResult:
    listed = ", ".join(available) or "none"
    return error_result("unknown_tool", f"Unknown tool {name!r}. Available tools: {listed}.")


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False):
        location = ".".join(str(part) for part in err["loc"]) or "<arguments>"
        # rstrip: the caller appends ". Expected JSON schema", so avoid "..".
        parts.append(f"{location}: {err['msg'].rstrip('.')}")
    return "; ".join(parts)


def _to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, default=str, separators=(",", ":"))
