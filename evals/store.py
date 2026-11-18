"""SQLite persistence of batches, runs and check results (stdlib `sqlite3`).

    batches        one row per `evals.run` invocation: label, git commit, full config
    runs           one row per RunRecord (scalar columns + `trace_json` for the rest)
    check_results  one row per check verdict of a run

The schema is created if missing. `schema_version` holds one integer. A version-1 database
(before MCP) is migrated in place to version 2 in ONE transaction (two added columns, old
runs get `tool_backend='local'`): either everything is applied or nothing is. Any other
version is refused rather than silently misread.
`passed` / `passed_with_heuristics` are stored for SQL convenience but recomputed from the
check rows on load (`RunRecord.passed`), so the two can never disagree in the harness.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .records import CheckResult, RunRecord, RunTrace, ToolCallTrace
from .suite import EVALS_DIR

SCHEMA_VERSION = 2
DEFAULT_DB_PATH = EVALS_DIR / "results" / "evals.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    git_commit TEXT,
    config_json TEXT NOT NULL,
    warmup_json TEXT
);
CREATE INDEX IF NOT EXISTS batches_by_label ON batches (label);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES batches (id),
    case_id TEXT NOT NULL,
    category TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    model TEXT NOT NULL,
    think INTEGER NOT NULL,
    prompt_variant TEXT NOT NULL,
    seed INTEGER,
    passed INTEGER,                  -- NULL: excluded (fixture miss or harness error)
    passed_with_heuristics INTEGER,
    stop_reason TEXT,
    iterations INTEGER NOT NULL,
    llm_calls INTEGER NOT NULL,
    tool_calls INTEGER NOT NULL,
    tool_errors INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    llm_latency_ms REAL NOT NULL,
    llm_load_ms REAL NOT NULL,
    wall_time_ms REAL NOT NULL,
    fixture_miss INTEGER NOT NULL,   -- number of arXiv requests no fixture covered
    harness_error TEXT,
    final_answer TEXT,
    trace_json TEXT NOT NULL,
    tool_backend TEXT NOT NULL DEFAULT 'local',  -- 'local' | 'mcp' (schema v2)
    mcp_startup_ms REAL                          -- NULL for local runs (schema v2)
);
CREATE INDEX IF NOT EXISTS runs_by_batch ON runs (batch_id);
CREATE TABLE IF NOT EXISTS check_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs (id),
    check_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    heuristic INTEGER NOT NULL,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS checks_by_run ON check_results (run_id);
"""

# v1 -> v2: the MCP tool backend. Applied inside one explicit transaction (SQLite DDL is
# transactional), together with the version bump.
_MIGRATION_V1_TO_V2 = (
    "ALTER TABLE runs ADD COLUMN tool_backend TEXT NOT NULL DEFAULT 'local'",
    "ALTER TABLE runs ADD COLUMN mcp_startup_ms REAL",
    "UPDATE schema_version SET version = 2",
)


class StoreError(RuntimeError):
    """The database exists but has an unexpected schema version."""


@dataclass(frozen=True)
class Batch:
    id: int
    label: str
    created_at: str
    git_commit: str | None
    config: dict[str, Any]
    warmup: dict[str, Any] | None


class ResultStore:
    """A connection to the results database. Use as a context manager, or call `close()`."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        try:
            self._init_schema()
        except Exception:
            self._db.close()
            raise

    def _init_schema(self) -> None:
        # Check the version BEFORE running the CREATE script: on an existing database the
        # script is a no-op (IF NOT EXISTS), so it never touches an unknown schema.
        version = self._existing_version()
        if version is not None and version not in (1, SCHEMA_VERSION):
            raise StoreError(f"database schema version {version}, expected {SCHEMA_VERSION}")
        with self._db:
            self._db.executescript(_SCHEMA)
            if version is None:
                self._db.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        if version == 1:
            self._migrate(_MIGRATION_V1_TO_V2)

    def _existing_version(self) -> int | None:
        exists = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
        if exists is None:
            return None
        row = self._db.execute("SELECT version FROM schema_version").fetchone()
        return None if row is None else int(row["version"])

    def _migrate(self, statements: tuple[str, ...]) -> None:
        """All statements in one transaction: on any error, the database is left untouched."""
        self._db.execute("BEGIN")
        try:
            for statement in statements:
                self._db.execute(statement)
        except BaseException:
            self._db.rollback()
            raise
        self._db.commit()

    # --- writing ------------------------------------------------------------------------------

    def create_batch(
        self,
        label: str,
        config: dict[str, Any],
        *,
        git_commit: str | None,
        warmup: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> int:
        """Insert a batch and return its id."""
        created = created_at or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        with self._db:
            cursor = self._db.execute(
                "INSERT INTO batches (label, created_at, git_commit, config_json, warmup_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    label,
                    created,
                    git_commit,
                    json.dumps(config, sort_keys=True, default=str),
                    None if warmup is None else json.dumps(warmup, sort_keys=True),
                ),
            )
        return _rowid(cursor)

    def save_run(self, batch_id: int, record: RunRecord) -> int:
        """Insert one run and its check results in a single transaction; return the run id."""
        trace = record.trace
        values = {
            "batch_id": batch_id,
            "case_id": record.case_id,
            "category": record.category,
            "repetition": record.repetition,
            "model": record.model,
            "think": int(record.think),
            "prompt_variant": record.prompt_variant,
            "seed": record.seed,
            "passed": _opt_int(record.passed),
            "passed_with_heuristics": _opt_int(record.passed_with_heuristics),
            "stop_reason": trace.stop_reason,
            "iterations": trace.iterations,
            "llm_calls": record.llm_calls,
            "tool_calls": len(trace.tool_calls),
            "tool_errors": trace.tool_errors,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "llm_latency_ms": record.llm_latency_ms,
            "llm_load_ms": record.llm_load_ms,
            "wall_time_ms": record.wall_time_ms,
            "fixture_miss": len(record.fixture_misses),
            "harness_error": record.harness_error,
            "final_answer": trace.final_answer,
            "trace_json": record.to_trace_json(),
            "tool_backend": record.tool_backend,
            "mcp_startup_ms": record.mcp_startup_ms,
        }
        columns = ", ".join(values)
        marks = ", ".join(f":{name}" for name in values)
        with self._db:
            cursor = self._db.execute(f"INSERT INTO runs ({columns}) VALUES ({marks})", values)
            run_id = _rowid(cursor)
            self._db.executemany(
                "INSERT INTO check_results (run_id, check_name, passed, heuristic, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (run_id, c.name, int(c.passed), int(c.heuristic), c.detail)
                    for c in record.checks
                ],
            )
        return run_id

    # --- reading ------------------------------------------------------------------------------

    def latest_batch(self, label: str) -> Batch | None:
        """The most recent batch with this label (labels may be reused)."""
        row = self._db.execute(
            "SELECT * FROM batches WHERE label = ? ORDER BY id DESC LIMIT 1", (label,)
        ).fetchone()
        return None if row is None else _batch(row)

    def labels(self) -> list[str]:
        rows = self._db.execute("SELECT DISTINCT label FROM batches ORDER BY label").fetchall()
        return [row["label"] for row in rows]

    def load_runs(self, batch_id: int) -> list[RunRecord]:
        """Every run of a batch, in insertion order, rebuilt as `RunRecord`s."""
        rows = self._db.execute(
            "SELECT * FROM runs WHERE batch_id = ? ORDER BY id", (batch_id,)
        ).fetchall()
        checks: dict[int, list[CheckResult]] = {row["id"]: [] for row in rows}
        for check in self._db.execute(
            "SELECT c.* FROM check_results c JOIN runs r ON r.id = c.run_id "
            "WHERE r.batch_id = ? ORDER BY c.id",
            (batch_id,),
        ):
            checks[check["run_id"]].append(
                CheckResult(
                    name=check["check_name"],
                    passed=bool(check["passed"]),
                    heuristic=bool(check["heuristic"]),
                    detail=check["detail"],
                )
            )
        return [_record(row, tuple(checks[row["id"]])) for row in rows]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _batch(row: sqlite3.Row) -> Batch:
    return Batch(
        id=row["id"],
        label=row["label"],
        created_at=row["created_at"],
        git_commit=row["git_commit"],
        config=json.loads(row["config_json"]),
        warmup=None if row["warmup_json"] is None else json.loads(row["warmup_json"]),
    )


def _record(row: sqlite3.Row, checks: tuple[CheckResult, ...]) -> RunRecord:
    extra = json.loads(row["trace_json"])
    trace = RunTrace(
        question=extra["question"],
        final_answer=row["final_answer"],
        stop_reason=row["stop_reason"],
        iterations=row["iterations"],
        tool_calls=tuple(ToolCallTrace(**call) for call in extra["tool_calls"]),
        tool_errors=row["tool_errors"],
    )
    return RunRecord(
        case_id=row["case_id"],
        category=row["category"],
        repetition=row["repetition"],
        model=row["model"],
        think=bool(row["think"]),
        prompt_variant=row["prompt_variant"],
        seed=row["seed"],
        trace=trace,
        expected_stop_reason=extra["expected_stop_reason"],
        llm_calls=row["llm_calls"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        llm_latency_ms=row["llm_latency_ms"],
        llm_load_ms=row["llm_load_ms"],
        wall_time_ms=row["wall_time_ms"],
        agent_error=extra["agent_error"],
        fixture_misses=tuple(extra["fixture_misses"]),
        harness_error=row["harness_error"],
        checks=checks,
        tool_backend=row["tool_backend"],
        mcp_startup_ms=row["mcp_startup_ms"],
    )


def _opt_int(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _rowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:  # pragma: no cover - sqlite always sets it after an INSERT
        raise StoreError("INSERT returned no row id")
    return cursor.lastrowid
