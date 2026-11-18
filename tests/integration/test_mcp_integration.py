"""Real Ollama, tools through a real MCP server: one eval case end to end. Opt-in:

uv run pytest -m integration tests/integration/test_mcp_integration.py
"""

from __future__ import annotations

import os

import pytest

from evals.runner import EvalConfig, run_case
from evals.suite import load_suite, resolve_suite_path, select_cases
from local_research_agent.llm import OllamaClient
from local_research_agent.llm.ollama import DEFAULT_MODEL

pytestmark = [pytest.mark.integration, pytest.mark.mcp]


def test_minimal_case_with_mcp_tools() -> None:
    [case] = select_cases(load_suite(resolve_suite_path("minimal")), ["calc_basic"])
    config = EvalConfig(model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL), tool_backend="mcp")
    with OllamaClient.from_env(model=config.model) as llm:
        record = run_case(case, config, 0, llm=llm)

    assert record.harness_error is None, record.harness_error
    assert record.tool_backend == "mcp" and record.mcp_startup_ms is not None
    assert any(call.name == "calculator" and call.ok for call in record.trace.tool_calls)
