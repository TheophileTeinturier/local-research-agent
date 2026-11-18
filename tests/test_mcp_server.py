"""The MCP server's pure parts, in-process: schema conversion, results, stdout discipline."""

from __future__ import annotations

import logging

import anyio
import mcp.types as types
import pytest

from local_research_agent.mcp.provider import from_call_tool_result, mcp_tool_to_schema
from local_research_agent.mcp.server import (
    ERROR_TYPE_META_KEY,
    build_parser,
    call_tool,
    load_registry_factory,
    main,
    mcp_tools,
    to_call_tool_result,
)
from local_research_agent.tools import ToolResult, build_default_registry
from mcp_helpers import parity_registry

# One simple call per real tool, all covered by the recorded fixtures.
REAL_CALLS: list[tuple[str, dict[str, object]]] = [
    ("calculator", {"expression": "17 * 23 + 4"}),
    ("current_date", {}),
    (
        "search_arxiv",
        {
            "query": "graph neural networks protein-protein interaction prediction",
            "sort": "date",
            "max_results": 5,
        },
    ),
    ("get_paper_details", {"arxiv_id": "2606.01781"}),
]


def test_mcp_tools_round_trip_to_the_exact_local_schemas() -> None:
    registry = build_default_registry()
    assert [mcp_tool_to_schema(t) for t in mcp_tools(registry)] == registry.schemas()


def test_result_conversion_round_trip() -> None:
    ok = ToolResult(ok=True, content="17 * 23 + 4 = 395")
    error = ToolResult(ok=False, content="Error: nope", metadata={"error_type": "not_found"})

    mcp_ok = to_call_tool_result(ok)
    assert not mcp_ok.is_error and mcp_ok.meta is None
    assert mcp_ok.content == [types.TextContent(type="text", text="17 * 23 + 4 = 395")]
    mcp_error = to_call_tool_result(error)
    assert mcp_error.is_error and mcp_error.meta == {ERROR_TYPE_META_KEY: "not_found"}

    assert from_call_tool_result(mcp_ok) == ok
    assert from_call_tool_result(mcp_error) == error


def test_result_from_a_foreign_server() -> None:
    """No `_meta`, a non-text block: generic error type, placeholder text."""
    result = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="part 1"),
            types.ImageContent(type="image", data="AAAA", mime_type="image/png"),
        ],
        is_error=True,
    )
    converted = from_call_tool_result(result)
    assert converted.content == "part 1\n[image content omitted]"
    assert converted.metadata == {"error_type": "tool_error"}


def test_truncation_metadata_is_recomputed_from_the_text() -> None:
    capped = "x" * 10 + "\n[truncated: showing 10 of 99 characters]"
    result = from_call_tool_result(
        types.CallToolResult(content=[types.TextContent(type="text", text=capped)])
    )
    assert result.metadata == {"truncated": True, "original_length": 99}


@pytest.mark.parametrize(
    ("name", "arguments", "error_type"),
    [
        ("nope", {}, "unknown_tool"),
        ("calculator", {"expression": 12}, "invalid_arguments"),
        ("calculator", {"expression": "1+1", "unit": "kg"}, "invalid_arguments"),
        ("boom", {"text": "x"}, "exception"),
    ],
)
def test_server_never_raises_on_bad_input(
    name: str, arguments: dict[str, object], error_type: str
) -> None:
    result = anyio.run(call_tool, parity_registry(), name, arguments)
    assert result.is_error
    assert result.meta == {ERROR_TYPE_META_KEY: error_type}


def test_stdout_stays_empty_while_every_real_tool_runs(capfd: pytest.CaptureFixture[str]) -> None:
    """stdout is the protocol channel: the tools must not write to it, not even at fd level.

    `capfd` captures file descriptors 1 and 2, so a C-level write or a subprocess inheriting
    stdout would be caught too, not only Python's `print`.
    """
    registry = parity_registry()  # building the registry must not print either
    for name, arguments in REAL_CALLS:
        result = anyio.run(call_tool, registry, name, arguments)
        assert not result.is_error, (name, result.content)
    errors = [anyio.run(call_tool, registry, "calculator", {"expression": "1/0"})]
    assert errors[0].is_error
    assert capfd.readouterr().out == ""


def test_factory_loading() -> None:
    assert load_registry_factory("mcp_helpers:parity_registry") is parity_registry
    for bad in ["no_colon", "missing_module_xyz:f", "mcp_helpers:nope", "mcp_helpers:"]:
        with pytest.raises(ValueError):
            load_registry_factory(bad)


def test_main_reports_a_broken_factory_on_stderr_only(capfd: pytest.CaptureFixture[str]) -> None:
    try:
        assert main(["--registry-factory", "mcp_helpers:broken_registry"]) == 1
    finally:  # main() configured the package logger: undo it for the other tests
        logger = logging.getLogger("local_research_agent")
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "mcp_server_startup_failed" in captured.err and "factory exploded" in captured.err


def test_default_factory_is_the_production_registry() -> None:
    args = build_parser().parse_args([])
    assert args.registry_factory == "local_research_agent.tools:build_default_registry"
