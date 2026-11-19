"""Provenance: the tool-schema hash, what the agent really sends, Ollama metadata, git state."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.provenance import (
    SchemaRecorder,
    canonical_schemas_json,
    git_commit,
    read_ollama_info,
    tool_schemas_sha256,
    working_tree_changes,
)
from evals.runner import EvalConfig, run_case
from evals.suite import Case
from fakes import FakeLLM, make_eval_fixtures, make_git_repo, text_response, tool_call_response
from local_research_agent.agent import AgentConfig, run_agent
from local_research_agent.tools import ToolRegistry, build_registry, calculator_tool
from local_research_agent.tools.current_date import current_date_tool

# --- the hash -------------------------------------------------------------------------------


def test_hash_is_stable_across_registries_and_calls() -> None:
    first = tool_schemas_sha256(build_registry().schemas())
    assert first == tool_schemas_sha256(build_registry().schemas())
    assert len(first) == 64


def test_canonical_json_sorts_keys_but_keeps_the_list_order() -> None:
    schemas: list[Any] = [{"b": 1, "a": "é"}, {"z": 0}]
    assert canonical_schemas_json(schemas) == '[{"a":"é","b":1},{"z":0}]'.encode()
    reordered_keys: list[Any] = [{"a": "é", "b": 1}, {"z": 0}]
    assert tool_schemas_sha256(schemas) == tool_schemas_sha256(reordered_keys)
    assert tool_schemas_sha256(schemas) != tool_schemas_sha256(schemas[::-1])


def test_hash_changes_with_a_description_or_the_tool_order() -> None:
    base = ToolRegistry([calculator_tool, current_date_tool])
    reference = tool_schemas_sha256(base.schemas())

    reworded = replace(calculator_tool, description=calculator_tool.description + " Exact.")
    assert tool_schemas_sha256(ToolRegistry([reworded, current_date_tool]).schemas()) != reference
    swapped = ToolRegistry([current_date_tool, calculator_tool])
    assert tool_schemas_sha256(swapped.schemas()) != reference


def test_recorder_hashes_exactly_what_the_agent_sends_after_subset() -> None:
    registry = build_registry()
    recorder = SchemaRecorder(FakeLLM([text_response("4")]))
    run_agent(recorder, registry, AgentConfig(allowed_tools=("calculator",)), "2+2?")

    subset = tool_schemas_sha256(registry.subset(["calculator"]).schemas())
    assert recorder.sha256 == subset != tool_schemas_sha256(registry.schemas())


def test_recorder_flags_calls_with_different_tools() -> None:
    recorder = SchemaRecorder(FakeLLM([text_response("a"), text_response("b")]))
    assert recorder.sha256 is None  # no call yet
    recorder.chat([], tools=[{"x": 1}])  # type: ignore[list-item]
    recorder.chat([], tools=[{"x": 2}])  # type: ignore[list-item]
    assert recorder.sha256 is not None and recorder.sha256.startswith("inconsistent:")


CASE = {
    "id": "calc",
    "category": "single_tool",
    "question": "What is 2+2?",
    "checks": [{"name": "tool_called", "params": {"tool": "calculator"}}],
}
SCRIPT = [tool_call_response(("calculator", {"expression": "2+2"})), text_response("4")]


def test_every_run_records_the_same_hash(tmp_path: Path) -> None:
    fixtures, case = make_eval_fixtures(tmp_path), Case.model_validate(CASE)
    hashes = {
        run_case(case, EvalConfig(), r, llm=FakeLLM(SCRIPT), fixtures_dir=fixtures)
        .tool_schemas_sha256
        for r in range(3)
    }
    assert hashes == {tool_schemas_sha256(build_registry().schemas())}


@pytest.mark.mcp
def test_local_and_mcp_backends_send_the_same_schemas(tmp_path: Path) -> None:
    """Runtime contract: through MCP (list_tools, rebuilt client-side) the LLM receives the
    very same schema list as in-process, on the same registry (`evals.mcp_factory:build`)."""
    fixtures, case = make_eval_fixtures(tmp_path), Case.model_validate(CASE)
    records = {
        backend: run_case(
            case, EvalConfig(tool_backend=backend), 0, llm=FakeLLM(SCRIPT), fixtures_dir=fixtures
        )
        for backend in ("local", "mcp")
    }
    assert all(r.harness_error is None for r in records.values())
    local, mcp = (records[b].tool_schemas_sha256 for b in ("local", "mcp"))
    assert local is not None and local == mcp


# --- Ollama ---------------------------------------------------------------------------------

TAGS = {
    "models": [
        {"name": "llama3.2:latest", "model": "llama3.2:latest", "digest": "aaa"},
        {"name": "qwen3:8b", "model": "qwen3:8b", "digest": "bbb"},
    ]
}


def mock(routes: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        if isinstance(body, Exception):
            raise body
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_reads_version_and_digest() -> None:
    transport = mock({"/api/version": {"version": "0.40.14"}, "/api/tags": TAGS})
    info = read_ollama_info("http://ollama", "qwen3:8b", transport=transport)
    assert (info.version, info.model_digest) == ("0.40.14", "bbb")
    # A name without a tag is Ollama's ":latest".
    latest = read_ollama_info("http://ollama", "llama3.2", transport=transport)
    assert latest.model_digest == "aaa"


@pytest.mark.parametrize(
    "routes",
    [
        {},  # both endpoints 404
        {"/api/version": "not json", "/api/tags": "not json"},
        {"/api/version": {"version": 3}, "/api/tags": {"models": "nope"}},
        {"/api/version": httpx.ConnectError("down"), "/api/tags": httpx.ConnectError("down")},
        {"/api/version": {"version": "0.1"}, "/api/tags": {"models": [{"name": "other"}]}},
    ],
)
def test_unreadable_values_are_none_and_never_raise(routes: dict[str, Any]) -> None:
    info = read_ollama_info("http://ollama", "qwen3:8b", transport=mock(routes))
    assert info.model_digest is None
    assert info.version in (None, "0.1")


def test_no_host_means_unknown_without_any_request() -> None:
    assert read_ollama_info(None, "qwen3:8b") == read_ollama_info("", "qwen3:8b")
    assert read_ollama_info(None, "qwen3:8b").version is None


# --- git ------------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_git_repo(tmp_path / "repo")


def test_working_tree_changes(repo: Path) -> None:
    assert working_tree_changes(repo) == []
    (repo / "ignored.txt").write_text("x", encoding="utf-8")  # covered by .gitignore
    assert working_tree_changes(repo) == []
    clean = git_commit(repo)
    assert clean is not None and not clean.endswith("+dirty")

    (repo / "tracked.txt").write_text("v2\n", encoding="utf-8")
    assert [line[3:] for line in working_tree_changes(repo) or []] == ["tracked.txt"]
    assert str(git_commit(repo)).endswith("+dirty")


def test_untracked_files_count_as_changes(repo: Path) -> None:
    (repo / "new.txt").write_text("x", encoding="utf-8")
    assert working_tree_changes(repo) == ["?? new.txt"]


def test_not_a_repository(tmp_path: Path) -> None:
    outside = tmp_path / "plain"
    outside.mkdir()
    # tmp_path may itself sit inside some repository on a developer machine; only assert the
    # contract when git really reports "not a repository" here.
    result = subprocess.run(["git", "status"], cwd=outside, capture_output=True)
    if result.returncode != 0:
        assert working_tree_changes(outside) is None and git_commit(outside) is None


def test_json_dumps_is_the_canonical_form() -> None:
    schemas = build_registry().schemas()
    text = canonical_schemas_json(schemas).decode("utf-8")
    assert json.loads(text) == schemas  # nothing lost, only the formatting is fixed
    assert text == json.dumps(schemas, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
