"""ToolRegistry: exposes tool schemas to the LLM and executes tool calls safely.

`execute()` is the trust boundary between model output and Python code. Whatever the model
sends (unknown name, malformed JSON, wrong types, extra fields) and whatever the tool does
(raises, returns huge output), `execute()` returns a `ToolResult` and never raises.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from ..observability import log_event
from .base import Tool, ToolError, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_CHARS = 4000


class ToolRegistry:
    def __init__(
        self, tools: Iterable[Tool[Any]] = (), *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    ) -> None:
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")
        self.max_output_chars = max_output_chars
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

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        """A new registry holding only `names` (same `Tool` objects, same output cap).

        A call to a tool outside the subset is then an ordinary unknown tool: an error
        observation listing only the tools of the subset. Raises `ValueError` for a name that
        is not registered here, because that is a configuration mistake, not model behaviour.
        """
        wanted = set(names)
        missing = sorted(wanted - self._tools.keys())
        if missing:
            raise ValueError(f"Unknown tool(s) in subset: {', '.join(missing)}")
        return ToolRegistry(
            (tool for tool in self._tools.values() if tool.name in wanted),
            max_output_chars=self.max_output_chars,
        )

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions in Ollama's `tools` request format (the OpenAI function format)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, raw_args: Mapping[str, Any] | str | None) -> ToolResult:
        """Validate, run and cap one tool call. Never raises, except for Ctrl-C and SystemExit."""
        start = time.perf_counter()
        result = self._cap_output(self._run(name, raw_args))
        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        result = replace(result, metadata={**result.metadata, "duration_ms": duration_ms})
        log_event(
            logger,
            "tool_execution",
            level=logging.INFO if result.ok else logging.WARNING,
            tool=name,
            ok=result.ok,
            duration_ms=duration_ms,
            error_type=result.metadata.get("error_type"),
            truncated=result.metadata.get("truncated", False),
        )
        return result

    def _run(self, name: str, raw_args: Mapping[str, Any] | str | None) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "none"
            return _error("unknown_tool", f"Unknown tool {name!r}. Available tools: {available}.")

        args_obj = _decode_arguments(raw_args)
        if isinstance(args_obj, ToolResult):
            return args_obj

        # Pydantic is the gatekeeper: the tool function only ever receives a valid model
        # instance. The error message lists every faulty field and repeats the schema, so the
        # model has what it needs to fix its next call.
        try:
            args = tool.args_model.model_validate(args_obj)
        except ValidationError as exc:
            schema = json.dumps(tool.parameters_schema(), separators=(",", ":"))
            return _error(
                "invalid_arguments",
                f"Invalid arguments for {name!r}: {_format_validation_error(exc)}. "
                f"Expected JSON schema: {schema}",
            )

        try:
            output = tool.function(args)
        except ToolError as exc:
            return _error("tool_error", str(exc))
        except Exception as exc:
            # Deliberately broad: a buggy tool must not crash the agent. The model sees a short
            # error; the traceback goes to the logs for the developer.
            logger.exception("tool %s raised", name, extra={"fields": {"tool": name}})
            return _error("exception", f"Tool {name!r} failed: {type(exc).__name__}: {exc}")

        return ToolResult(ok=True, content=_to_text(output))

    def _cap_output(self, result: ToolResult) -> ToolResult:
        # The cap applies to every result, error messages included (a validation error that
        # echoes a huge input could flood the context too).
        length = len(result.content)
        if length <= self.max_output_chars:
            return result
        content = (
            result.content[: self.max_output_chars]
            + f"\n[truncated: showing {self.max_output_chars} of {length} characters]"
        )
        metadata = {**result.metadata, "truncated": True, "original_length": length}
        return replace(result, content=content, metadata=metadata)


def _decode_arguments(raw_args: Mapping[str, Any] | str | None) -> Mapping[str, Any] | ToolResult:
    """Return the argument object, or an error ToolResult if it is not a JSON object."""
    if raw_args is None:
        return {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as exc:
            return _error("invalid_arguments", f"Arguments are not valid JSON: {exc.msg}.")
    if not isinstance(raw_args, Mapping):
        return _error(
            "invalid_arguments",
            f"Arguments must be a JSON object, got {type(raw_args).__name__}.",
        )
    return raw_args


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False):
        location = ".".join(str(part) for part in err["loc"]) or "<arguments>"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def _error(error_type: str, message: str) -> ToolResult:
    # Error observations always start with "Error:" so the model can't mistake one for data.
    return ToolResult(ok=False, content=f"Error: {message}", metadata={"error_type": error_type})


def _to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, default=str, separators=(",", ":"))
