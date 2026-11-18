"""Aggregate stored runs into a markdown report; optionally compare two labels.

    uv run python -m evals.report --label baseline [--compare research_workflow]

Prints the report and writes it to `evals/results/<label>.md` (or `<a>_vs_<b>.md`).
`summarize` is pure (records in, numbers out) and tested on hand-built records; the
`render_*` functions only format. Pass counts are `k/n`, never bare percentages: with n=3,
"67%" hides that one run flipped. n counts only runs that are NOT excluded:
- a fixture miss makes a run INCONCLUSIVE (the fake arXiv was asked something no fixture
  covers): listed in its own section with the uncovered request, the search tool call(s) and
  their arguments, and the checks it would have failed. `evals.run` exits non-zero when more
  than `INCONCLUSIVE_LIMIT` of the runs are inconclusive;
- a harness error: listed under "Excluded runs".
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs

from .records import RunRecord
from .store import DEFAULT_DB_PATH, Batch, ResultStore

RESULTS_DIR = DEFAULT_DB_PATH.parent

# Above this fraction of inconclusive (fixture-miss) runs, a batch says too little about the
# agent: `evals.run` warns and exits non-zero. Strictly greater: 1 run in 10 is tolerated.
INCONCLUSIVE_LIMIT = 0.10

# The only tool that sends arXiv *search* requests, i.e. the only one that can cause a miss.
SEARCH_TOOL = "search_arxiv"

# (label in the report, RunRecord attribute or trace attribute)
METRICS: tuple[tuple[str, str], ...] = (
    ("iterations", "iterations"),
    ("tool calls", "tool_call_count"),
    ("prompt tokens", "prompt_tokens"),
    ("completion tokens", "completion_tokens"),
    ("LLM latency (ms)", "llm_latency_ms"),
    ("wall time (ms)", "wall_time_ms"),
)


@dataclass
class PassCount:
    """k/n over the counted (non-excluded) runs, plus the excluded ones."""

    passed: int = 0
    passed_with_heuristics: int = 0
    counted: int = 0
    fixture_miss: int = 0
    harness_error: int = 0

    def add(self, record: RunRecord) -> None:
        if record.harness_error is not None:
            self.harness_error += 1
        elif record.fixture_misses:
            self.fixture_miss += 1
        else:
            self.counted += 1
            self.passed += bool(record.passed)
            self.passed_with_heuristics += bool(record.passed_with_heuristics)

    @property
    def fraction(self) -> str:
        return f"{self.passed}/{self.counted}"

    @property
    def fraction_with_heuristics(self) -> str:
        return f"{self.passed_with_heuristics}/{self.counted}"

    @property
    def runs(self) -> int:
        """Every run, counted or not."""
        return self.counted + self.fixture_miss + self.harness_error

    @property
    def excluded(self) -> str:
        parts = []
        if self.fixture_miss:
            parts.append(f"{self.fixture_miss} inconclusive (fixture_miss)")
        if self.harness_error:
            parts.append(f"{self.harness_error} harness_error")
        return ", ".join(parts) or "-"


@dataclass(frozen=True)
class Stats:
    median: float
    max: float


@dataclass
class CaseSummary:
    case_id: str
    category: str
    count: PassCount = field(default_factory=PassCount)
    # check label -> (times failed among counted runs, first failure detail)
    failing: dict[str, tuple[int, str]] = field(default_factory=dict)
    failing_heuristic: dict[str, tuple[int, str]] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ExcludedRun:
    """A run the harness failed to complete (harness error)."""

    case_id: str
    repetition: int
    detail: str


@dataclass(frozen=True)
class InconclusiveRun:
    """A run with a fixture miss: what was not covered, and which tool call asked for it."""

    case_id: str
    repetition: int
    uncovered: tuple[str, ...]  # readable form of each uncovered request
    tool_calls: tuple[str, ...]  # e.g. 'search_arxiv {"query":"2406.99999"} -> ok'
    would_fail: tuple[str, ...]  # deterministic checks that failed anyway


@dataclass(frozen=True)
class BatchSummary:
    batch: Batch
    total: PassCount
    categories: dict[str, PassCount]
    cases: list[CaseSummary]
    metrics: dict[str, Stats]
    measured_runs: int
    inconclusive: list[InconclusiveRun]
    excluded: list[ExcludedRun]  # harness errors
    # Per tool: latency of the executed calls (repetition warnings excluded), measured by the
    # agent's registry: in-process (local) or round trip through MCP (mcp).
    tool_latency: dict[str, Stats] = field(default_factory=dict)
    mcp_startup: Stats | None = None  # server spawn + handshake, mcp runs only

    @property
    def too_many_inconclusive(self) -> bool:
        """More than INCONCLUSIVE_LIMIT of all runs are inconclusive."""
        runs = self.total.runs
        return runs > 0 and self.total.fixture_miss / runs > INCONCLUSIVE_LIMIT


def summarize(batch: Batch, runs: Sequence[RunRecord]) -> BatchSummary:
    """Pure aggregation of one batch. Case and category order follow first appearance."""
    total = PassCount()
    categories: dict[str, PassCount] = {}
    cases: dict[str, CaseSummary] = {}
    per_case_values: dict[str, dict[str, list[float]]] = {}
    inconclusive: list[InconclusiveRun] = []
    excluded: list[ExcludedRun] = []

    for record in runs:
        total.add(record)
        categories.setdefault(record.category, PassCount()).add(record)
        case = cases.setdefault(record.case_id, CaseSummary(record.case_id, record.category))
        case.count.add(record)

        if record.harness_error is not None:
            excluded.append(ExcludedRun(record.case_id, record.repetition, record.harness_error))
            continue
        if record.fixture_misses:
            inconclusive.append(_inconclusive(record))
            continue
        for check in record.checks:
            if not check.passed:
                target = case.failing_heuristic if check.heuristic else case.failing
                times, detail = target.get(check.name, (0, check.detail))
                target[check.name] = (times + 1, detail)

    measured = [r for r in runs if r.harness_error is None]
    for record in measured:
        values = per_case_values.setdefault(record.case_id, {})
        for _label, attr in METRICS:
            values.setdefault(attr, []).append(_metric(record, attr))
    for case_id, values in per_case_values.items():
        cases[case_id].medians = {attr: statistics.median(v) for attr, v in values.items()}

    metrics = {
        label: Stats(statistics.median(vals), max(vals))
        for label, attr in METRICS
        if (vals := [_metric(r, attr) for r in measured])
    }
    latencies: dict[str, list[float]] = {}
    for record in measured:
        for call in record.trace.tool_calls:
            if not call.repeated and call.duration_ms is not None:
                latencies.setdefault(call.name, []).append(call.duration_ms)
    startups = [r.mcp_startup_ms for r in measured if r.mcp_startup_ms is not None]
    return BatchSummary(
        batch=batch,
        total=total,
        categories=categories,
        cases=list(cases.values()),
        metrics=metrics,
        measured_runs=len(measured),
        inconclusive=inconclusive,
        excluded=excluded,
        tool_latency={name: _stats(values) for name, values in latencies.items()},
        mcp_startup=_stats(startups) if startups else None,
    )


def _stats(values: Sequence[float]) -> Stats:
    return Stats(statistics.median(values), max(values))


def _metric(record: RunRecord, attr: str) -> float:
    if attr == "iterations":
        return float(record.trace.iterations)
    if attr == "tool_call_count":
        return float(len(record.trace.tool_calls))
    return float(getattr(record, attr))


def _inconclusive(record: RunRecord) -> InconclusiveRun:
    """The uncovered requests and the tool calls that can have sent them.

    A miss is (in practice) a search, and only `search_arxiv` searches, so its calls are shown.
    If the run made none (a request of another kind), every tool call is shown instead.
    """
    calls = record.trace.tool_calls
    shown = [c for c in calls if c.name == SEARCH_TOOL] or list(calls)
    return InconclusiveRun(
        case_id=record.case_id,
        repetition=record.repetition,
        uncovered=tuple(describe_request(key) for key in record.fixture_misses),
        tool_calls=tuple(
            f"{c.name} {c.arguments} -> "
            + ("REPEATED" if c.repeated else "ok" if c.ok else f"ERROR ({c.error_type})")
            for c in shown
        ),
        would_fail=tuple(c.name for c in record.checks if not c.passed and not c.heuristic),
    )


def describe_request(query_string: str) -> str:
    """'search all:x AND all:y (max_results=5)' for a search, else the raw query string."""
    params = {k: v[0] for k, v in parse_qs(query_string).items()}
    if "search_query" in params:
        return f"search {params['search_query']} (max_results={params.get('max_results', '?')})"
    return query_string


# --- rendering ------------------------------------------------------------------------------


def render_report(summary: BatchSummary) -> str:
    """The markdown report of one batch."""
    batch, total = summary.batch, summary.total
    lines = [f"# Eval report: {batch.label}", "", *_config_lines(batch), ""]
    lines += [
        "## Results",
        "",
        f"Overall: **{total.fraction}** pass (deterministic checks), "
        f"**{total.fraction_with_heuristics}** with heuristic checks too. "
        f"Not counted: {total.fixture_miss} inconclusive (fixture_miss), "
        f"{total.harness_error} harness_error.",
        "",
        "### By category",
        "",
        "| category | pass | pass + heuristics | excluded |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {name} | {c.fraction} | {c.fraction_with_heuristics} | {c.excluded} |"
        for name, c in summary.categories.items()
    ]
    lines += [
        "",
        "### By case",
        "",
        "| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out "
        "| med. LLM ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in summary.cases:
        m = case.medians
        tokens = f"{m['prompt_tokens']:.0f}/{m['completion_tokens']:.0f}" if m else "-"
        iters = f"{m['iterations']:g}" if m else "-"
        latency = f"{m['llm_latency_ms']:.0f}" if m else "-"
        lines.append(
            f"| {case.case_id} | {case.category} | {case.count.fraction} | "
            f"{case.count.fraction_with_heuristics} | {case.count.excluded} | {iters} | "
            f"{tokens} | {latency} |"
        )
    lines += ["", "### Failing checks", ""]
    lines += _failing_lines(summary.cases) or ["None."]
    if summary.inconclusive:
        lines += ["", *_inconclusive_lines(summary)]
    if summary.excluded:
        lines += ["", "### Excluded runs (harness_error, not counted above)", ""]
        lines += [f"- {r.case_id} rep {r.repetition}: {r.detail}" for r in summary.excluded]
    lines += [
        "",
        f"## Metrics (all runs without harness_error, n={summary.measured_runs}; warm-up excluded)",
        "",
        "| metric | median | max |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {s.median:.0f} | {s.max:.0f} |" for k, s in summary.metrics.items()]
    if summary.mcp_startup is not None:
        s = summary.mcp_startup
        lines.append(f"| MCP server startup (ms) | {s.median:.0f} | {s.max:.0f} |")
    if summary.tool_latency:
        lines += [
            "",
            f"## Tool call latency ({_backend(summary.batch.config)} backend, ms)",
            "",
            "| tool | median | max |",
            "|---|---|---|",
        ]
        lines += [
            f"| {name} | {s.median:.2f} | {s.max:.2f} |" for name, s in summary.tool_latency.items()
        ]
    return "\n".join(lines) + "\n"


def render_comparison(a: BatchSummary, b: BatchSummary) -> str:
    """Side-by-side k/n per case and category, the config differences, and metric medians."""
    la, lb = a.batch.label, b.batch.label
    lines = [f"# Eval comparison: {la} vs {lb}", ""]
    lines += [f"- {la}: batch {a.batch.id}, {a.batch.created_at}, git {a.batch.git_commit}"]
    lines += [f"- {lb}: batch {b.batch.id}, {b.batch.created_at}, git {b.batch.git_commit}", ""]

    diffs = _config_diff(a.batch.config, b.batch.config)
    lines += ["## Configuration differences", ""]
    if diffs:
        lines += [f"| key | {la} | {lb} |", "|---|---|---|"]
        lines += [f"| {key} | {va} | {vb} |" for key, va, vb in diffs]
    else:
        lines.append("None.")

    lines += ["", "## Pass counts (deterministic checks)", ""]
    lines += [f"| case | {la} | {lb} | diff |", "|---|---|---|---|"]
    a_cases = {c.case_id: c.count for c in a.cases}
    b_cases = {c.case_id: c.count for c in b.cases}
    for case_id in [*a_cases, *(c for c in b_cases if c not in a_cases)]:
        ca, cb = a_cases.get(case_id), b_cases.get(case_id)
        lines.append(f"| {case_id} | {_frac(ca)} | {_frac(cb)} | {_diff(ca, cb)} |")
    lines.append(
        f"| **total** | {a.total.fraction} | {b.total.fraction} | {_diff(a.total, b.total)} |"
    )

    lines += ["", "## By category", "", f"| category | {la} | {lb} | diff |", "|---|---|---|---|"]
    for name in [*a.categories, *(c for c in b.categories if c not in a.categories)]:
        ca, cb = a.categories.get(name), b.categories.get(name)
        lines.append(f"| {name} | {_frac(ca)} | {_frac(cb)} | {_diff(ca, cb)} |")

    lines += ["", "## Metric medians", "", f"| metric | {la} | {lb} |", "|---|---|---|"]
    for key, _attr in METRICS:
        sa, sb = a.metrics.get(key), b.metrics.get(key)
        lines.append(f"| {key} | {_med(sa)} | {_med(sb)} |")
    lines.append(f"| MCP server startup (ms) | {_med(a.mcp_startup)} | {_med(b.mcp_startup)} |")

    lines += [
        "",
        "## Tool call latency (median ms)",
        "",
        f"Backends: {la} = {_backend(a.batch.config)}, {lb} = {_backend(b.batch.config)}. "
        "Local = in-process execution; mcp = round trip through the bridge and the server.",
        "",
        f"| tool | {la} | {lb} |",
        "|---|---|---|",
    ]
    tools = [*a.tool_latency, *(t for t in b.tool_latency if t not in a.tool_latency)]
    lines += [
        f"| {t} | {_med(a.tool_latency.get(t), 2)} | {_med(b.tool_latency.get(t), 2)} |"
        for t in tools
    ] or ["| (no tool calls) | - | - |"]
    lines.append("")
    lines.append(
        f"Excluded runs: {la} {a.total.excluded}; {lb} {b.total.excluded}. "
        "Failing checks per case: see each label's own report."
    )
    return "\n".join(lines) + "\n"


def _config_lines(batch: Batch) -> list[str]:
    c = batch.config
    seed = c.get("seed")
    warm = batch.warmup or {}
    warm_text = (
        f"ok, {warm.get('latency_ms', 0):.0f} ms (model load {warm.get('load_ms', 0):.0f} ms)"
        if warm.get("ok")
        else f"FAILED: {warm.get('error')}"
        if warm
        else "none recorded"
    )
    return [
        f"- batch {batch.id}, created {batch.created_at}, git {batch.git_commit}",
        f"- model `{c.get('model')}`, think {'on' if c.get('think') else 'off'}, prompt variant "
        f"`{c.get('prompt_variant')}` (sha256 {str(c.get('system_prompt_sha256', ''))[:12]}), "
        f"num_ctx {c.get('num_ctx')}, temperature {c.get('temperature')}, "
        f"seed {'= repetition index' if seed is None else seed}, "
        f"repetitions {c.get('repetitions')}, suite `{c.get('suite')}`, "
        f"tools backend `{_backend(c)}`",
        f"- warm-up (excluded from metrics): {warm_text}",
    ]


def _inconclusive_lines(summary: BatchSummary) -> list[str]:
    total = summary.total
    lines = [
        "### Inconclusive runs (fixture_miss, not counted above)",
        "",
        "The fake arXiv was asked for something no fixture covers, so these runs say nothing "
        "about the agent. Fix the case (declare a `search` fixture or `default_search`), "
        "not the agent.",
    ]
    if summary.too_many_inconclusive:
        lines += [
            "",
            f"**Warning: {total.fixture_miss}/{total.runs} runs are inconclusive, more than "
            f"{INCONCLUSIVE_LIMIT:.0%}: this batch measures the fixtures more than the agent.**",
        ]
    lines.append("")
    for run in summary.inconclusive:
        lines.append(f"- **{run.case_id}** rep {run.repetition}")
        lines += [f"  - uncovered request: `{request}`" for request in run.uncovered]
        lines += [f"  - tool call: `{call}`" for call in run.tool_calls]
        if run.would_fail:
            lines.append(f"  - would fail: {', '.join(f'`{n}`' for n in run.would_fail)}")
    return lines


def _failing_lines(cases: Sequence[CaseSummary]) -> list[str]:
    lines = []
    for case in cases:
        if not case.failing and not case.failing_heuristic:
            continue
        lines.append(f"- **{case.case_id}** ({case.count.fraction} pass)")
        for name, (times, detail) in case.failing.items():
            lines.append(f"  - `{name}` failed {times}/{case.count.counted}: {detail}")
        for name, (times, detail) in case.failing_heuristic.items():
            lines.append(f"  - heuristic `{name}` failed {times}/{case.count.counted}: {detail}")
    return lines


# Keys added to the batch config after some batches were stored, with the value those older
# batches implicitly had: compared as such, so "missing" vs "local" is not a difference.
_CONFIG_DEFAULTS: dict[str, object] = {"tool_backend": "local"}


def _backend(config: dict[str, object]) -> object:
    return config.get("tool_backend", _CONFIG_DEFAULTS["tool_backend"])


def _config_diff(a: dict[str, object], b: dict[str, object]) -> list[tuple[str, object, object]]:
    a, b = {**_CONFIG_DEFAULTS, **a}, {**_CONFIG_DEFAULTS, **b}
    keys = sorted(a.keys() | b.keys())
    return [(k, a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k) and k != "label"]


def _frac(count: PassCount | None) -> str:
    return "-" if count is None else count.fraction


def _diff(a: PassCount | None, b: PassCount | None) -> str:
    """+1 when both have the same n; else the pass-rate difference in percentage points."""
    if a is None or b is None or a.counted == 0 or b.counted == 0:
        return "n/a"
    if a.counted == b.counted:
        return f"{b.passed - a.passed:+d}"
    points = 100 * (b.passed / b.counted - a.passed / a.counted)
    return f"{points:+.0f} pp"


def _med(stats: Stats | None, decimals: int = 0) -> str:
    return "-" if stats is None else f"{stats.median:.{decimals}f}"


# --- CLI ------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals.report", description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="batch label (the latest batch is used)")
    parser.add_argument("--compare", help="second label: side-by-side comparison")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    with ResultStore(args.db) as store:
        summaries = []
        for label in filter(None, [args.label, args.compare]):
            batch = store.latest_batch(label)
            if batch is None:
                known = ", ".join(store.labels()) or "none"
                print(f"No batch labelled {label!r} in {args.db} (known: {known}).")
                return 2
            summaries.append(summarize(batch, store.load_runs(batch.id)))

    if args.compare:
        text = render_comparison(summaries[0], summaries[1])
        name = f"{args.label}_vs_{args.compare}.md"
    else:
        text = render_report(summaries[0])
        name = f"{args.label}.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / name
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"(written to {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
