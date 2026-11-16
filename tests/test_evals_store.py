"""SQLite store: a RunRecord survives the round trip unchanged; the schema is versioned."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from evals.records import CheckResult, RunRecord, RunTrace, ToolCallTrace
from evals.store import SCHEMA_VERSION, ResultStore, StoreError


def make_record(**overrides: object) -> RunRecord:
    trace = RunTrace(
        question="Summarize arXiv:1706.02216",
        final_answer="GraphSAGE (1706.02216) is an inductive method. é ✓",
        stop_reason="final_answer",
        iterations=2,
        tool_calls=(
            ToolCallTrace(1, "get_paper_details", '{"arxiv_id":"1706.02216"}', True, "arXiv:..."),
            ToolCallTrace(1, "calculator", '{"expression":"x"}', False, "Error: ...", False, "x"),
        ),
        tool_errors=1,
    )
    values: dict[str, object] = {
        "case_id": "details_known_id",
        "category": "tool_choice",
        "repetition": 1,
        "model": "qwen3:8b",
        "think": False,
        "prompt_variant": "baseline",
        "seed": 1,
        "trace": trace,
        "expected_stop_reason": "final_answer",
        "llm_calls": 2,
        "prompt_tokens": 2100,
        "completion_tokens": 180,
        "llm_latency_ms": 3456.7,
        "llm_load_ms": 12.5,
        "wall_time_ms": 3500.1,
        "agent_error": None,
        "fixture_misses": (),
        "harness_error": None,
        "checks": (
            CheckResult("tool_called(tool=get_paper_details)", True, False, "1 time(s)"),
            CheckResult("answer_signals_failure(regex=x)", False, True, "no match"),
        ),
    }
    values.update(overrides)
    return RunRecord(**values)  # type: ignore[arg-type]


def test_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "evals.sqlite"  # parent directories are created
    records = [
        make_record(),
        make_record(repetition=2, fixture_misses=("search_query=all%3Ax",)),
        make_record(
            repetition=3,
            trace=RunTrace("q", None, None, 0, (), 0),
            checks=(),
            harness_error="RuntimeError: boom",
        ),
    ]
    with ResultStore(db) as store:
        batch_id = store.create_batch(
            "baseline",
            {"model": "qwen3:8b", "seed": None},
            git_commit="abc123+dirty",
            warmup={"ok": True, "latency_ms": 5000.0},
        )
        for record in records:
            store.save_run(batch_id, record)

    with ResultStore(db) as store:  # reopened: the data is on disk
        batch = store.latest_batch("baseline")
        assert batch is not None and batch.id == batch_id
        assert batch.config == {"model": "qwen3:8b", "seed": None}
        assert batch.git_commit == "abc123+dirty" and batch.warmup == {
            "ok": True,
            "latency_ms": 5000.0,
        }
        assert store.load_runs(batch_id) == records


def test_passed_columns_match_the_record(tmp_path: Path) -> None:
    db = tmp_path / "evals.sqlite"
    with ResultStore(db) as store:
        batch_id = store.create_batch("b", {}, git_commit=None)
        store.save_run(batch_id, make_record())  # deterministic ok, heuristic failed
        store.save_run(batch_id, make_record(harness_error="x", checks=()))
    with closing(sqlite3.connect(db)) as connection:
        rows = connection.execute(
            "SELECT passed, passed_with_heuristics, tool_calls, fixture_miss FROM runs ORDER BY id"
        ).fetchall()
        count = connection.execute("SELECT COUNT(*) FROM check_results").fetchone()
    assert rows == [(1, 0, 2, 0), (None, None, 2, 0)]
    assert count == (2,)


def test_latest_batch_and_labels() -> None:
    with ResultStore(":memory:") as store:
        first = store.create_batch("baseline", {"n": 1}, git_commit=None)
        store.create_batch("other", {}, git_commit=None)
        last = store.create_batch("baseline", {"n": 2}, git_commit=None)
        batch = store.latest_batch("baseline")
        assert batch is not None and batch.id == last != first and batch.config == {"n": 2}
        assert store.latest_batch("missing") is None
        assert store.labels() == ["baseline", "other"]


def test_schema_version_is_checked(tmp_path: Path) -> None:
    db = tmp_path / "evals.sqlite"
    ResultStore(db).close()
    with closing(sqlite3.connect(db)) as connection:
        versions = connection.execute("SELECT version FROM schema_version").fetchall()
        assert versions == [(SCHEMA_VERSION,)]
        with connection:
            connection.execute("UPDATE schema_version SET version = 999")
    with pytest.raises(StoreError, match="schema version 999"):
        ResultStore(db)
