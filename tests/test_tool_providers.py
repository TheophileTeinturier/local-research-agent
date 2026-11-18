"""ToolRegistry as an aggregator of providers: routing, duplicates, subset, cap, safety net."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from local_research_agent.tools import (
    LocalToolProvider,
    Tool,
    ToolProvider,
    ToolRegistry,
    ToolResult,
)
from local_research_agent.tools.providers import RawArguments, ToolSchema, tool_schema
from local_research_agent.tools.registry import cap_output, truncation_info


class TextArgs(BaseModel):
    text: str = ""


def _tool(name: str, output: str = "local") -> Tool[TextArgs]:
    return Tool(name, f"The {name} tool.", TextArgs, lambda args: f"{output}:{args.text}")


class FakeProvider:
    """A non-local provider: records calls, answers with a fixed result or raises."""

    def __init__(self, names: list[str], *, content: str = "remote", raises: bool = False):
        self._names = names
        self.content = content
        self.raises = raises
        self.calls: list[tuple[str, RawArguments]] = []

    def schemas(self) -> list[ToolSchema]:
        return [tool_schema(n, f"Remote {n}.", {"type": "object"}) for n in self._names]

    def has(self, name: str) -> bool:
        return name in self._names

    def execute(self, name: str, raw_args: RawArguments) -> ToolResult:
        self.calls.append((name, raw_args))
        if self.raises:
            raise ConnectionError("provider bug")
        return ToolResult(ok=True, content=self.content)


def test_fake_provider_satisfies_the_protocol() -> None:
    assert isinstance(FakeProvider(["x"]), ToolProvider)
    assert isinstance(LocalToolProvider(), ToolProvider)


def test_routing_across_providers() -> None:
    remote = FakeProvider(["remote_a", "remote_b"])
    registry = ToolRegistry([_tool("local_a")], providers=[remote])

    assert registry.names == ["local_a", "remote_a", "remote_b"]
    assert [s["function"]["name"] for s in registry.schemas()] == registry.names
    assert registry.execute("local_a", {"text": "x"}).content == "local:x"
    result = registry.execute("remote_b", '{"q": 1}')
    assert result.ok and result.content == "remote"
    assert remote.calls == [("remote_b", '{"q": 1}')]  # raw arguments, untouched
    assert "remote_a" in registry and "nope" not in registry
    assert registry.get("remote_a") is None  # `get` returns local Tool objects only


def test_unknown_tool_lists_every_provider() -> None:
    registry = ToolRegistry([_tool("local_a")], providers=[FakeProvider(["remote_a"])])
    result = registry.execute("nope", {})
    assert not result.ok and result.metadata["error_type"] == "unknown_tool"
    assert "Available tools: local_a, remote_a." in result.content


def test_duplicate_names_across_providers_are_rejected() -> None:
    registry = ToolRegistry([_tool("shared")])
    with pytest.raises(ValueError, match=r"Duplicate tool name.*shared"):
        registry.add_provider(FakeProvider(["other", "shared"]))
    assert registry.names == ["shared"]  # nothing was added

    with pytest.raises(ValueError, match=r"Duplicate tool name.*twice"):
        ToolRegistry(providers=[FakeProvider(["twice", "twice"])])

    registry = ToolRegistry(providers=[FakeProvider(["remote"])])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_tool("remote"))


def test_subset_across_providers() -> None:
    remote = FakeProvider(["remote_a", "remote_b"])
    registry = ToolRegistry([_tool("local_a"), _tool("local_b")], providers=[remote])
    subset = registry.subset(["remote_b", "local_a"])

    assert subset.names == ["local_a", "remote_b"]
    assert [s["function"]["name"] for s in subset.schemas()] == ["local_a", "remote_b"]
    assert subset.execute("remote_b", {}).ok
    blocked = subset.execute("remote_a", {})
    assert not blocked.ok and "Available tools: local_a, remote_b." in blocked.content
    assert ("remote_a", {}) not in remote.calls  # never reached the provider
    with pytest.raises(ValueError, match="subset"):
        subset.register(_tool("new"))
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.subset(["missing"])


def test_provider_exception_becomes_an_observation() -> None:
    registry = ToolRegistry(providers=[FakeProvider(["flaky"], raises=True)])
    result = registry.execute("flaky", {})
    assert not result.ok
    assert result.metadata["error_type"] == "provider_error"
    assert result.content == "Error: Tool 'flaky' failed: ConnectionError: provider bug"


def test_cap_applies_to_any_provider() -> None:
    registry = ToolRegistry(
        providers=[FakeProvider(["big"], content="y" * 30)], max_output_chars=10
    )
    result = registry.execute("big", {})
    assert result.content == "y" * 10 + "\n[truncated: showing 10 of 30 characters]"
    assert result.metadata["truncated"] is True and result.metadata["original_length"] == 30


def test_cap_is_idempotent() -> None:
    capped = cap_output(ToolResult(ok=True, content="z" * 50), 20)
    assert cap_output(capped, 20) == capped  # same cap: untouched
    assert cap_output(capped, 100) == capped  # larger cap: untouched
    smaller = cap_output(capped, 5)  # smaller cap: cut again, original length preserved
    assert smaller.content == "zzzzz\n[truncated: showing 5 of 50 characters]"
    assert truncation_info(smaller.content) == ("zzzzz", 50)
    assert truncation_info("no marker") is None


def test_local_provider_alone() -> None:
    provider = LocalToolProvider([_tool("a")])
    assert provider.has("a") and not provider.has("b")
    assert provider.execute("a", {"text": "t"}).content == "local:t"
    assert provider.execute("b", {}).metadata["error_type"] == "unknown_tool"
    bad: Any = provider.execute("a", {"text": 3})
    assert bad.metadata["error_type"] == "invalid_arguments"
