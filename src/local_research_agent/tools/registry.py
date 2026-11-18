"""ToolRegistry: exposes tool schemas to the LLM and executes tool calls safely.

The registry aggregates `ToolProvider`s (local tools, MCP tools, ...) behind one contract:

- `schemas()`: every tool of every provider, in the LLM's format;
- `execute(name, raw_args)`: the trust boundary between model output and Python code.
  Whatever the model sends (unknown name, malformed JSON, wrong types, extra fields) and
  whatever the provider does (raises, returns huge output), `execute()` returns a
  `ToolResult` and never raises;
- `subset(names)`: the same providers, restricted to some tools.

What holds WHATEVER the provider is enforced here, on the final result: a tool name is
served by exactly one provider (duplicates are rejected when a provider is added), the output
cap, the timing, the `tool_execution` log event, and the conversion of a provider exception
into an observation.

`ToolRegistry(tools)` keeps its original meaning: a registry whose tools live in its own
`LocalToolProvider`. `ToolRegistry(providers=[...])` aggregates providers.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ..observability import log_event
from .base import Tool, ToolResult
from .providers import (
    LocalToolProvider,
    ToolProvider,
    ToolSchema,
    error_result,
    unknown_tool_result,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_CHARS = 4000

# Appended by the cap. Parsed back by `truncation_info` (an MCP result carries no metadata,
# so the client recovers `truncated` / `original_length` from the text itself).
_TRUNCATION_MARKER_RE = re.compile(r"\n\[truncated: showing (\d+) of (\d+) characters\]\Z")


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[Tool[Any]] = (),
        *,
        providers: Iterable[ToolProvider] = (),
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")
        self.max_output_chars = max_output_chars
        # `register()` adds to this provider. It is always first, so a registry built from
        # `tools` has exactly the tool order it had before providers existed.
        self._local = LocalToolProvider()
        self._providers: list[ToolProvider] = [self._local]
        self._allowed: frozenset[str] | None = None  # set only on a `subset()`
        for tool in tools:
            self.register(tool)
        for provider in providers:
            self.add_provider(provider)

    # --- building ---------------------------------------------------------------------------

    def register(self, tool: Tool[Any]) -> None:
        """Add a local tool. Raises `ValueError` if any provider already has this name."""
        self._check_mutable()
        if tool.name in self._all_names():
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._local.register(tool)

    def add_provider(self, provider: ToolProvider) -> None:
        """Add a provider. Raises `ValueError` naming every tool another provider already has
        (or that the provider lists twice): a call must route to exactly one provider."""
        self._check_mutable()
        existing = set(self._all_names())
        seen: set[str] = set()
        duplicates = []
        for name in _provider_names(provider):
            if name in existing or name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate tool name(s) across providers: {', '.join(sorted(set(duplicates)))}"
            )
        self._providers.append(provider)

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        """A registry exposing only `names`: same providers, same `Tool` objects, same cap.

        A call to a tool outside the subset is then an ordinary unknown tool: an error
        observation listing only the tools of the subset. Raises `ValueError` for a name that
        is not registered here, because that is a configuration mistake, not model behaviour.
        """
        wanted = frozenset(names)
        missing = sorted(wanted - set(self.names))
        if missing:
            raise ValueError(f"Unknown tool(s) in subset: {', '.join(missing)}")
        restricted = ToolRegistry(max_output_chars=self.max_output_chars)
        restricted._local = self._local
        restricted._providers = list(self._providers)
        restricted._allowed = wanted
        return restricted

    def _check_mutable(self) -> None:
        if self._allowed is not None:
            # A subset shares its providers with its parent: adding here would add there.
            raise ValueError("Cannot add tools to a subset; add them to the full registry")

    # --- reading ----------------------------------------------------------------------------

    def get(self, name: str) -> Tool[Any] | None:
        """The local `Tool` called `name` (None for an unknown, remote or filtered-out tool)."""
        if not self._visible(name):
            return None
        for provider in self._providers:
            if isinstance(provider, LocalToolProvider) and (tool := provider.get(name)):
                return tool
        return None

    @property
    def names(self) -> list[str]:
        return [name for name in self._all_names() if self._visible(name)]

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._route(name) is not None

    def schemas(self) -> list[ToolSchema]:
        """Tool definitions in Ollama's `tools` request format (the OpenAI function format)."""
        return [
            schema
            for provider in self._providers
            for schema in provider.schemas()
            if self._visible(schema["function"]["name"])
        ]

    # --- executing --------------------------------------------------------------------------

    def execute(self, name: str, raw_args: Mapping[str, Any] | str | None) -> ToolResult:
        """Route, run and cap one tool call. Never raises, except for Ctrl-C and SystemExit."""
        start = time.perf_counter()
        result = cap_output(self._run(name, raw_args), self.max_output_chars)
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
        provider = self._route(name)
        if provider is None:
            return unknown_tool_result(name, self.names)
        try:
            return provider.execute(name, raw_args)
        except Exception as exc:
            # Providers convert their own expected failures. This is the safety net for a
            # buggy provider: the agent loop must never see an exception from a tool call.
            logger.exception("provider for %s raised", name, extra={"fields": {"tool": name}})
            return error_result(
                "provider_error", f"Tool {name!r} failed: {type(exc).__name__}: {exc}"
            )

    def _route(self, name: str) -> ToolProvider | None:
        if not self._visible(name):
            return None
        return next((p for p in self._providers if p.has(name)), None)

    def _visible(self, name: str) -> bool:
        return self._allowed is None or name in self._allowed

    def _all_names(self) -> list[str]:
        return [name for provider in self._providers for name in _provider_names(provider)]


def _provider_names(provider: ToolProvider) -> list[str]:
    if isinstance(provider, LocalToolProvider):
        return provider.names  # cheaper than generating the JSON schemas
    return [schema["function"]["name"] for schema in provider.schemas()]


# --- output cap -------------------------------------------------------------------------------


def cap_output(result: ToolResult, max_chars: int) -> ToolResult:
    """Cut `result.content` to `max_chars` characters plus a visible marker.

    Applies to every result, error messages included (a validation error that echoes a huge
    input could flood the context too). IDEMPOTENT: a result already capped to at most
    `max_chars` (marked `truncated`) is returned unchanged, so a result capped by an MCP
    server is not cut again by the client's registry.
    """
    body, original_length = result.content, len(result.content)
    if result.metadata.get("truncated"):
        info = truncation_info(result.content)
        if info is not None:
            body, original_length = info
    if len(body) <= max_chars:
        return result
    content = (
        body[:max_chars] + f"\n[truncated: showing {max_chars} of {original_length} characters]"
    )
    metadata = {**result.metadata, "truncated": True, "original_length": original_length}
    return replace(result, content=content, metadata=metadata)


def truncation_info(content: str) -> tuple[str, int] | None:
    """(kept text, original length) if `content` ends with the cap's marker, else None."""
    match = _TRUNCATION_MARKER_RE.search(content)
    if match is None:
        return None
    return content[: match.start()], int(match.group(2))
