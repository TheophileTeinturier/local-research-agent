"""The eval harness in MCP mode (real server subprocesses, FakeLLM), store migration, report."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from evals import runner
from evals.records import RunRecord, RunTrace, ToolCallTrace
from evals.report import Stats, render_comparison, render_report, summarize
from evals.runner import EvalConfig, run_case
from evals.store import SCHEMA_VERSION, Batch, ResultStore
from evals.suite import Case
from fakes import FakeLLM, make_eval_fixtures, text_response, tool_call_response


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    return make_eval_fixtures(tmp_path / "fixtures")


def make_case(**overrides: Any) -> Case:
    base = {
        "id": "search_case",
        "category": "tool_choice",
        "question": "Find papers on graph neural networks",
        "checks": [{"name": "tool_called", "params": {"tool": "search_arxiv"}}],
    }
    return Case.model_validate({**base, **overrides})


SEARCH_SCRIPT = [
    tool_call_response(("search_arxiv", {"query": "graph neural network"})),
    text_response("Here are the papers."),
]


@pytest.mark.mcp
def test_mcp_run_records_backend_startup_and_latency(fixtures: Path) -> None:
    case = make_case(arxiv={"search": "search.xml"})
    record = run_case(
        case, EvalConfig(tool_backend="mcp"), 0, llm=FakeLLM(SEARCH_SCRIPT), fixtures_dir=fixtures
    )
    assert record.harness_error is None, record.harness_error
    assert record.tool_backend == "mcp"
    assert record.mcp_startup_ms is not None and record.mcp_startup_ms > 0
    # End to end includes the server startup; the agent's own wall time does not.
    assert record.e2e_wall_time_ms is not None
    assert record.e2e_wall_time_ms >= record.mcp_startup_ms
    assert record.e2e_wall_time_ms >= record.wall_time_ms
    [call] = record.trace.tool_calls
    assert call.ok and "2606.01781" in call.content  # served by the fixture, in the server
    assert call.duration_ms is not None and call.duration_ms > 0
    assert record.passed is True and record.fixture_misses == ()


@pytest.mark.mcp
def test_mcp_fixture_miss_crosses_the_process_boundary(fixtures: Path) -> None:
    case = make_case()  # no search fixture, no default_search: any search is a miss
    record = run_case(
        case, EvalConfig(tool_backend="mcp"), 0, llm=FakeLLM(SEARCH_SCRIPT), fixtures_dir=fixtures
    )
    assert record.harness_error is None, record.harness_error
    assert len(record.fixture_misses) == 1 and "search_query" in record.fixture_misses[0]
    assert record.excluded and record.passed is None


@pytest.mark.mcp
def test_mcp_failure_mode_crosses_the_process_boundary(fixtures: Path) -> None:
    case = make_case(arxiv={"failure_mode": "http_503"})
    record = run_case(
        case, EvalConfig(tool_backend="mcp"), 0, llm=FakeLLM(SEARCH_SCRIPT), fixtures_dir=fixtures
    )
    [call] = record.trace.tool_calls
    assert not call.ok and call.error_type == "arxiv_unavailable"  # carried by `_meta`
    assert record.harness_error is None and record.fixture_misses == ()


@pytest.mark.mcp
def test_mcp_startup_failure_is_a_harness_error(
    fixtures: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "MCP_REGISTRY_FACTORY", "evals.no_such_module:build")
    record = run_case(
        make_case(), EvalConfig(tool_backend="mcp"), 0, llm=FakeLLM([]), fixtures_dir=fixtures
    )
    assert record.harness_error is not None and "McpStartupError" in record.harness_error
    assert record.excluded and record.passed is None  # never counted as an agent failure


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="tool_backend"):
        EvalConfig(tool_backend="http")  # type: ignore[arg-type]


# --- store ------------------------------------------------------------------------------------


def _record(**overrides: Any) -> RunRecord:
    call = ToolCallTrace(1, "calculator", '{"expression":"1"}', True, "1 = 1", duration_ms=12.5)
    base: dict[str, Any] = {
        "case_id": "c",
        "category": "single_tool",
        "repetition": 0,
        "model": "m",
        "think": False,
        "prompt_variant": "baseline",
        "seed": 0,
        "trace": RunTrace("q", "a", "final_answer", 2, (call,), 0),
        "expected_stop_reason": "final_answer",
    }
    return RunRecord(**{**base, **overrides})


def test_store_round_trips_the_new_fields(tmp_path: Path) -> None:
    record = _record(
        tool_backend="mcp",
        mcp_startup_ms=850.0,
        llm_prompt_tokens=(1034, 1216),
        e2e_wall_time_ms=4321.5,
        synthetic=True,
    )
    with ResultStore(tmp_path / "db.sqlite") as store:
        batch = store.create_batch("x", {"tool_backend": "mcp"}, git_commit=None)
        store.save_run(batch, record)
        [loaded] = store.load_runs(batch)
    assert loaded == record
    assert (loaded.tool_backend, loaded.mcp_startup_ms) == ("mcp", 850.0)
    assert loaded.trace.tool_calls[0].duration_ms == 12.5
    assert (loaded.peak_prompt_tokens, loaded.e2e_wall_time_ms) == (1216, 4321.5)


def test_rows_recorded_before_the_new_trace_fields_still_load(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    with ResultStore(path) as store:
        batch = store.create_batch("x", {}, git_commit=None)
        store.save_run(batch, _record(llm_prompt_tokens=(5,), e2e_wall_time_ms=1.0))
    with sqlite3.connect(path) as connection:  # strip the keys, as an older harness wrote it
        [(raw,)] = connection.execute("SELECT trace_json FROM runs").fetchall()
        old = json.loads(raw)
        del old["llm_prompt_tokens"], old["e2e_wall_time_ms"]
        connection.execute("UPDATE runs SET trace_json = ?", (json.dumps(old),))
    connection.close()
    with ResultStore(path) as store:
        [loaded] = store.load_runs(batch)
    assert loaded.llm_prompt_tokens == () and loaded.peak_prompt_tokens is None
    assert loaded.e2e_wall_time_ms is None


def test_v1_database_is_migrated_in_place(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    with ResultStore(path) as store:
        batch = store.create_batch("old", {"model": "m"}, git_commit=None)
        store.save_run(batch, _record())
    # Turn it into a version-1 database: drop the v2 columns, set the version back.
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE runs DROP COLUMN tool_backend")
        connection.execute("ALTER TABLE runs DROP COLUMN mcp_startup_ms")
        connection.execute("UPDATE schema_version SET version = 1")
    connection.close()

    with ResultStore(path) as store:
        [run] = store.load_runs(batch)
        assert (run.tool_backend, run.mcp_startup_ms) == ("local", None)
        assert run.trace.final_answer == "a"  # old data intact
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version FROM schema_version").fetchall() == [
            (SCHEMA_VERSION,)
        ]
    connection.close()


# --- report -----------------------------------------------------------------------------------


def _batch(batch_id: int, label: str, config: dict[str, Any]) -> Batch:
    return Batch(batch_id, label, "2026-11-16T10:00:00", "abc", config, None)


def test_report_shows_backend_startup_and_tool_latency() -> None:
    summary = summarize(
        _batch(1, "mcp", {"tool_backend": "mcp"}),
        [_record(tool_backend="mcp", mcp_startup_ms=900.0)],
    )
    text = render_report(summary)
    assert "tools backend `mcp`" in text
    assert "| MCP server startup (ms) | 900 | 900 |" in text
    assert "| calculator | 12.50 | 12.50 |" in text


def test_comparison_shows_backend_difference_and_latency_medians() -> None:
    local = summarize(_batch(1, "loc", {"model": "m"}), [_record()])  # pre-MCP config
    remote = summarize(
        _batch(2, "rem", {"model": "m", "tool_backend": "mcp"}),
        [_record(tool_backend="mcp", mcp_startup_ms=800.0)],
    )
    text = render_comparison(local, remote)
    assert "| tool_backend | local | mcp |" in text
    assert "## Tool call latency (local vs mcp)" in text
    assert "| calculator | 12.50 (n=1) | 12.50 (n=1) | +0.00 | x1.0 |" in text
    assert "| MCP server startup (ms) | - | 800 |" in text
    same = render_comparison(local, local)
    assert "tool_backend" not in same.split("## Pass counts")[0]  # missing key = "local"
    assert "## Tool call latency (median ms)" in same  # same backend: plain side by side


def _timed(tool_backend: str, *durations: float, **overrides: Any) -> RunRecord:
    calls = tuple(
        ToolCallTrace(1, name, "{}", True, "ok", duration_ms=ms)
        for name, ms in zip(("search_arxiv", "get_paper_details"), durations, strict=False)
    )
    return _record(
        tool_backend=tool_backend,
        trace=RunTrace("q", "a", "final_answer", 2, calls, 0),
        **overrides,
    )


def test_local_vs_mcp_latency_side_by_side_in_either_order() -> None:
    local = summarize(
        _batch(1, "loc", {"tool_backend": "local"}),
        [_timed("local", 200.0, 50.0), _timed("local", 300.0, 70.0), _timed("local", 250.0)],
    )
    remote = summarize(
        _batch(2, "rem", {"tool_backend": "mcp"}),
        [_timed("mcp", 260.0, 90.0, mcp_startup_ms=900.0)],
    )
    for text in (render_comparison(local, remote), render_comparison(remote, local)):
        section = text.split("## Tool call latency (local vs mcp)")[1]
        # The local batch is always the first column, whatever the argument order.
        assert "| tool | local (loc) | mcp (rem) | mcp - local | mcp / local |" in section
        assert "| search_arxiv | 250.00 (n=3) | 260.00 (n=1) | +10.00 | x1.0 |" in section
        assert "| get_paper_details | 60.00 (n=2) | 90.00 (n=1) | +30.00 | x1.5 |" in section


def test_end_to_end_time_and_peak_prompt_in_the_report() -> None:
    summary = summarize(
        _batch(1, "mcp", {"tool_backend": "mcp"}),
        [
            _record(llm_prompt_tokens=(100, 400, 300), e2e_wall_time_ms=2000.0),
            _record(llm_prompt_tokens=(100, 200), e2e_wall_time_ms=3000.0, repetition=1),
            _record(repetition=2),  # an older row: neither metric, left out of both
        ],
    )
    assert summary.metrics["peak prompt tokens per call"] == Stats(300.0, 400.0, 2)
    assert summary.metrics["end-to-end wall time (ms)"] == Stats(2500.0, 3000.0, 2)
    assert summary.metrics["wall time (ms)"].n == 3
    [case] = summary.cases
    assert case.medians["peak_prompt_tokens"] == 300.0
    text = render_report(summary)
    assert "| peak prompt tokens per call | 300 | 400 |" in text
    assert "| end-to-end wall time (ms) | 2500 | 3000 |" in text
