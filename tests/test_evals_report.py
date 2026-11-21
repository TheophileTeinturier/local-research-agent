"""Report aggregation and compare mode on a small hand-built dataset."""

from dataclasses import replace
from pathlib import Path

import pytest

from evals import report
from evals.records import CheckResult, RunRecord, RunTrace, ToolCallTrace
from evals.report import describe_request, render_comparison, render_report, summarize
from evals.store import Batch, ResultStore

ANSWER_CHECK = "answer_matches(regex=294)"
HEURISTIC = "answer_signals_failure(regex=x)"

# What the client really sends for search_arxiv {"query": "2406.99999"} (default 5, relevance).
MISSED_SEARCH = (
    "search_query=all%3A2406+AND+all%3A99999&start=0&max_results=5"
    "&sortBy=relevance&sortOrder=descending"
)
# The trace of an unknown-id run whose model fell back to a search nobody declared.
MISS_CALLS = (
    ToolCallTrace(
        1,
        "get_paper_details",
        '{"arxiv_id":"2406.99999"}',
        False,
        "Error: ...",
        error_type="not_found",
    ),
    ToolCallTrace(2, "search_arxiv", '{"query":"2406.99999"}', True, "No papers matched"),
)


def rec(
    case_id: str,
    category: str,
    *,
    ok: bool = True,
    heuristic_ok: bool = True,
    repetition: int = 0,
    latency: float = 1000.0,
    iterations: int = 2,
    miss: bool = False,
    error: bool = False,
) -> RunRecord:
    checks = (
        CheckResult(ANSWER_CHECK, ok, False, "detail of the failure" if not ok else "ok"),
        CheckResult(HEURISTIC, heuristic_ok, True, "no match" if not heuristic_ok else "ok"),
    )
    return RunRecord(
        case_id=case_id,
        category=category,
        repetition=repetition,
        model="qwen3:8b",
        think=False,
        prompt_variant="baseline",
        seed=repetition,
        trace=RunTrace("q", "a", "final_answer", iterations, MISS_CALLS if miss else (), 0),
        expected_stop_reason="final_answer",
        llm_calls=iterations,
        prompt_tokens=1000 * iterations,
        completion_tokens=50,
        llm_latency_ms=latency,
        wall_time_ms=latency + 10,
        fixture_misses=(MISSED_SEARCH,) if miss else (),
        harness_error="RuntimeError: boom" if error else None,
        checks=() if error else checks,
    )


def batch(label: str, batch_id: int = 1, **config: object) -> Batch:
    base = {
        "model": "qwen3:8b",
        "think": False,
        "prompt_variant": "baseline",
        "seed": None,
        "repetitions": 3,
        "suite": "evals/cases/minimal.json",
    }
    return Batch(
        batch_id,
        label,
        "2026-11-10T10:00:00+00:00",
        "abc1234",
        {**base, **config},
        {"ok": True, "latency_ms": 5000.0, "load_ms": 4800.0},
    )


RUNS = [
    rec("calc_basic", "single_tool", repetition=0, latency=1000),
    rec("calc_basic", "single_tool", repetition=1, latency=3000),
    rec("calc_basic", "single_tool", ok=False, repetition=2, latency=2000),
    rec("research", "multi_step", ok=False, heuristic_ok=False, iterations=5, latency=9000),
    rec("research", "multi_step", miss=True, ok=False),
    rec("research", "multi_step", error=True),
]


def test_summarize_counts_k_over_n_and_excludes() -> None:
    summary = summarize(batch("baseline"), RUNS)
    assert (summary.total.fraction, summary.total.fraction_with_heuristics) == ("2/4", "2/4")
    assert (summary.total.fixture_miss, summary.total.harness_error) == (1, 1)
    assert summary.categories["single_tool"].fraction == "2/3"
    assert summary.categories["multi_step"].fraction == "0/1"

    calc, research = summary.cases
    assert (calc.case_id, calc.count.fraction) == ("calc_basic", "2/3")
    assert calc.failing == {ANSWER_CHECK: (1, "detail of the failure")}
    assert research.count.fraction == "0/1" and research.count.excluded == (
        "1 inconclusive (fixture_miss), 1 harness_error"
    )
    # The excluded fixture-miss run's failure is not counted, the counted one is.
    assert research.failing == {ANSWER_CHECK: (1, "detail of the failure")}
    assert research.failing_heuristic == {HEURISTIC: (1, "no match")}

    # Metrics: every run without harness_error (fixture-miss runs did run): 5 runs.
    assert summary.measured_runs == 5
    latency = summary.metrics["LLM latency (ms)"]
    assert (latency.median, latency.max) == (2000.0, 9000.0)
    assert summary.metrics["iterations"].max == 5
    assert calc.medians["llm_latency_ms"] == 2000.0

    [inconclusive] = summary.inconclusive
    assert inconclusive.case_id == "research" and inconclusive.would_fail == (ANSWER_CHECK,)
    assert inconclusive.uncovered == ("search all:2406 AND all:99999 (max_results=5)",)
    # Only the search call is shown: get_paper_details cannot send a search request.
    assert inconclusive.tool_calls == ('search_arxiv {"query":"2406.99999"} -> ok',)
    assert [(e.case_id, e.detail) for e in summary.excluded] == [("research", "RuntimeError: boom")]


def test_render_report() -> None:
    text = render_report(summarize(batch("baseline"), RUNS))
    assert text.startswith("# Eval report: baseline\n")
    assert "model `qwen3:8b`, think off" in text and "seed = repetition index" in text
    assert "warm-up (excluded from metrics): ok, 5000 ms (model load 4800 ms)" in text
    assert "Overall: **2/4** pass" in text
    assert "| single_tool | 2/3 | 2/3 | - |" in text
    assert (
        "| research | multi_step | 0/1 | 0/1 | 1 inconclusive (fixture_miss), 1 harness_error |"
        in text
    )
    assert "Not counted: 1 inconclusive (fixture_miss), 1 harness_error." in text
    assert f"  - `{ANSWER_CHECK}` failed 1/3: detail of the failure" in text
    assert f"  - heuristic `{HEURISTIC}` failed 1/1: no match" in text
    inconclusive = text.split("### Inconclusive runs (fixture_miss, not counted above)")[1]
    assert (
        "- **research** rep 0\n"
        "  - uncovered request: `search all:2406 AND all:99999 (max_results=5)`\n"
        '  - tool call: `search_arxiv {"query":"2406.99999"} -> ok`\n'
        f"  - would fail: `{ANSWER_CHECK}`\n"
    ) in inconclusive
    assert "**Warning:" in inconclusive  # 1 of 6 runs > 10%
    assert "### Excluded runs (harness_error, not counted above)\n\n" in text
    assert "- research rep 0: RuntimeError: boom" in text
    assert "| LLM latency (ms) | 2000 | 9000 |" in text


def test_no_inconclusive_section_without_misses() -> None:
    text = render_report(summarize(batch("baseline"), RUNS[:4]))
    assert "Inconclusive runs" not in text and "Excluded runs" not in text


@pytest.mark.parametrize(
    ("misses", "total", "too_many"),
    [(0, 10, False), (1, 10, False), (2, 10, True), (1, 9, True), (0, 0, False)],
)
def test_inconclusive_threshold_is_strictly_above_ten_percent(
    misses: int, total: int, too_many: bool
) -> None:
    runs = [rec(f"c{i}", "x", miss=i < misses) for i in range(total)]
    assert summarize(batch("b"), runs).too_many_inconclusive is too_many


def test_describe_request() -> None:
    assert describe_request(MISSED_SEARCH) == "search all:2406 AND all:99999 (max_results=5)"
    assert describe_request("foo=bar") == "foo=bar"  # not a search: shown raw


def test_inconclusive_without_search_calls_shows_every_call() -> None:
    record = rec("x", "edge", miss=True)
    lookup_only = RunTrace("q", "a", "final_answer", 2, MISS_CALLS[:1], 1)
    record = replace(record, trace=lookup_only)
    [run] = summarize(batch("b"), [record]).inconclusive
    assert run.tool_calls == ('get_paper_details {"arxiv_id":"2406.99999"} -> ERROR (not_found)',)


def test_compare_mode() -> None:
    other_runs = [rec("calc_basic", "single_tool", repetition=r, latency=500) for r in range(3)] + [
        rec("research", "multi_step", ok=True),
        rec("research", "multi_step", ok=False),
    ]
    a = summarize(batch("baseline"), RUNS)
    b = summarize(batch("workflow", 2, prompt_variant="research_workflow"), other_runs)
    text = render_comparison(a, b)
    assert "# Eval comparison: baseline vs workflow" in text
    assert "| prompt_variant | baseline | research_workflow |" in text
    assert "| calc_basic | 2/3 | 3/3 | +1 |" in text  # same n: difference in runs
    assert "| research | 0/1 | 1/2 | +50 pp |" in text  # different n: percentage points
    assert "| **total** | 2/4 | 4/5 | +30 pp |" in text
    assert "| single_tool | 2/3 | 3/3 | +1 |" in text
    assert "| LLM latency (ms) | 2000 | 500 |" in text


def test_compare_mode_diff_legend_under_each_diff_table() -> None:
    a = summarize(batch("baseline"), RUNS)
    b = summarize(batch("workflow", 2, prompt_variant="research_workflow"), RUNS)
    lines = render_comparison(a, b).splitlines()
    legend = "diff = workflow minus baseline (positive = workflow passes more)"
    assert lines.count(legend) == 2
    for i in (i for i, line in enumerate(lines) if line == legend):
        assert lines[i - 1] == ""  # separated from the table, so not parsed as a row
        assert lines[i - 2].startswith("| ") and lines[i - 2].endswith(" |")
    total = next(i for i, line in enumerate(lines) if line.startswith("| **total** |"))
    category = lines.index("## By category")
    metrics = lines.index("## Metric medians")
    assert total < lines.index(legend) < category
    assert category < len(lines) - 1 - lines[::-1].index(legend) < metrics


def with_calls(record: RunRecord, *calls: tuple[str, str]) -> RunRecord:
    traced = tuple(ToolCallTrace(1, name, args, True, "ok") for name, args in calls)
    return replace(record, trace=replace(record.trace, tool_calls=traced))


def test_distinct_trajectories_per_case() -> None:
    search3, search10 = (
        ("search_arxiv", '{"max_results":3}'),
        ("search_arxiv", '{"max_results":10}'),
    )
    opened = ("get_paper_details", '{"arxiv_id":"2606.01781"}')
    runs = [
        with_calls(rec("research", "multi_step", repetition=0), search3, opened),
        with_calls(rec("research", "multi_step", repetition=1), search3, opened),
        with_calls(rec("research", "multi_step", repetition=2), search10, opened),
        with_calls(rec("research", "multi_step", repetition=3), opened, search3),  # order counts
        with_calls(rec("research", "multi_step", error=True), search10),  # harness error: ignored
        rec("calc_basic", "single_tool", repetition=0),
        rec("calc_basic", "single_tool", repetition=1),
    ]
    summary = summarize(batch("b"), runs)
    research, calc = summary.cases
    assert (research.distinct_trajectories, research.trajectory_runs) == (3, 4)
    assert (calc.distinct_trajectories, calc.trajectory_runs) == (1, 2)  # no tool, twice
    text = render_report(summary)
    assert "| distinct trajectories |" in text
    assert text.count("| 3 of 4 runs |") == 1 and "| 1 of 2 runs |" in text


def test_peak_prompt_median_per_case_in_the_report() -> None:
    runs = [
        replace(rec("calc_basic", "single_tool", repetition=r), llm_prompt_tokens=tokens)
        for r, tokens in enumerate([(900, 1200), (900, 1500), (800, 1300)])
    ]
    summary = summarize(batch("b"), runs)
    assert summary.cases[0].medians["peak_prompt_tokens"] == 1300.0
    assert "| 1300 | 1 of 3 runs |" in render_report(summary)
    # Older runs without per-call tokens: the column shows "-".
    assert "| - | 1 of 3 runs |" in render_report(summarize(batch("b"), RUNS[:3]))


SUITE_V2 = {
    "suite_sha256": "a" * 64,
    "cases_sha256": "c" * 64,
    "fixtures_manifest_sha256": "m" * 64,
}


@pytest.mark.parametrize(
    ("config_a", "config_b", "expected"),
    [
        (SUITE_V2, SUITE_V2, None),
        (SUITE_V2, {}, "suite hash unknown for b"),
        (
            SUITE_V2,
            {**SUITE_V2, "suite_sha256": "b" * 64, "fixtures_manifest_sha256": "n" * 64},
            "different suite hashes (aaaaaaaaaaaa vs bbbbbbbbbbbb): the fixtures manifest differs",
        ),
        (
            SUITE_V2,
            {**SUITE_V2, "suite_sha256": "b" * 64, "cases_sha256": "d" * 64},
            "the cases file differs",
        ),
        (
            SUITE_V2,
            {"suite_sha256": "b" * 64, "cases_sha256": "d" * 64, "fixtures_manifest_sha256": "n"},
            "the cases file and the fixtures manifest differ",
        ),
        (  # recorded when suite_sha256 covered the cases file only
            SUITE_V2,
            {"suite_sha256": "b" * 64},
            "b predates the fixtures-manifest hash",
        ),
    ],
)
def test_suite_mismatch(
    config_a: dict[str, object], config_b: dict[str, object], expected: str | None
) -> None:
    warning = report.suite_mismatch(batch("a", **config_a), batch("b", 2, **config_b))
    if expected is None:
        assert warning is None
    else:
        assert warning is not None and expected in warning


IDENTITY = {
    **SUITE_V2,
    "tool_schemas_sha256": "t" * 64,
    "system_prompt_sha256": "p" * 64,
    "ollama_version": "0.40.14",
    "model_digest": "sha256:" + "d" * 64,
    "temperature": 0.2,
}


def ident_batch(label: str, batch_id: int, git: str = "abc1234", **config: object) -> Batch:
    return replace(batch(label, batch_id, **{**IDENTITY, **config}), git_commit=git)


def warnings_of(a: Batch, b: Batch, vary: list[str] | None = None) -> tuple[str, ...]:
    return report.compare_batches(a, b, vary=vary).warnings


def test_compare_warns_loudly_on_different_suites() -> None:
    a = summarize(ident_batch("a", 1), RUNS)
    b_config = {"suite_sha256": "b" * 64, "cases_sha256": "d" * 64, "temperature": 0.7}
    b = summarize(ident_batch("b", 2, **b_config), RUNS)
    text = render_comparison(a, b)
    assert text.splitlines()[2].startswith("> **WARNING: these batches differ")
    assert "> - **different suite hashes (aaaaaaaaaaaa vs bbbbbbbbbbbb)" in text
    assert "> Intentionally varied: `temperature` (inferred" in text
    assert "temperature 0.2" in text and "temperature 0.7" in text
    assert "| temperature | 0.2 | 0.7 |" in text  # also in the configuration differences
    same = render_comparison(a, summarize(ident_batch("c", 3, temperature=0.9), RUNS))
    assert "WARNING" not in same and "Intentionally varied: `temperature`" in same


def test_identical_batches_have_no_warning_and_git_is_only_shown() -> None:
    check = report.compare_batches(ident_batch("a", 1), ident_batch("b", 2, git="def5678"))
    assert check.warnings == () and check.varied == () and check.varied_source == "none"
    assert check.expected == ("git commit: abc1234 (a) vs def5678 (b)",)


@pytest.mark.parametrize(
    ("key", "name"),
    [
        ("tool_schemas_sha256", "tool schemas differs"),
        ("system_prompt_sha256", "system prompt differs"),
        ("ollama_version", "Ollama version differs"),
        ("model_digest", "model digest differs"),
    ],
)
def test_every_identity_difference_warns(key: str, name: str) -> None:
    other = ident_batch("b", 2, **{key: "something-else"})
    [warning] = warnings_of(ident_batch("a", 1), other, vary=["tool_backend"])
    assert warning.startswith(name)


def test_unknown_identity_values_warn() -> None:
    old = ident_batch("old", 2, ollama_version=None, model_digest=None)
    found = warnings_of(ident_batch("a", 1), old)
    assert "Ollama version unknown for old: cannot tell whether they match" in found
    assert "model digest unknown for old: cannot tell whether they match" in found


def test_settings_beyond_the_varied_key_warn() -> None:
    b = ident_batch("b", 2, temperature=0.7, num_ctx=4096)
    found = warnings_of(ident_batch("a", 1, num_ctx=8192), b)  # two settings: nothing inferred
    assert "config `temperature` differs: 0.2 (a) vs 0.7 (b)" in found
    assert "config `num_ctx` differs: 8192 (a) vs 4096 (b)" in found
    only_num_ctx = warnings_of(ident_batch("a", 1, num_ctx=8192), b, vary=["temperature"])
    assert only_num_ctx == ("config `num_ctx` differs: 8192 (a) vs 4096 (b)",)


def test_a_varied_prompt_variant_makes_the_prompt_hash_expected() -> None:
    a = ident_batch("a", 1, prompt_variant="baseline")
    b = ident_batch("b", 2, prompt_variant="research_workflow", system_prompt_sha256="q" * 64)
    check = report.compare_batches(a, b)
    assert check.varied == ("prompt_variant",) and check.warnings == ()
    assert any(e.startswith("system prompt differs") for e in check.expected)
    # Varying something else: the same prompt difference warns.
    assert any("system prompt differs" in w for w in warnings_of(a, b, vary=["temperature"]))


def test_dirty_trees_warn_in_compare_mode_and_in_the_header() -> None:
    dirty = ident_batch("d", 2, git="abc1234+dirty")
    found = warnings_of(ident_batch("a", 1), dirty)
    message = "d ran on uncommitted changes (git abc1234+dirty): its code is not exactly any commit"
    assert found == (message,)
    header = render_report(summarize(dirty, RUNS))
    assert "> **WARNING: uncommitted changes.**" in header.split("## Results")[0]
    clean = render_report(summarize(ident_batch("a", 1), RUNS))
    assert "uncommitted" not in clean


def test_header_shows_ollama_digest_and_schema_hash_or_unknown() -> None:
    text = render_report(summarize(ident_batch("a", 1), RUNS))
    digest, schemas = "sha256:" + "d" * 64, "t" * 64
    assert f"- Ollama 0.40.14, model digest {digest}, tool schemas sha256 {schemas}" in text
    unknown = render_report(summarize(batch("b"), RUNS))  # nothing recorded
    assert "- Ollama unknown, model digest unknown, tool schemas sha256 unknown" in unknown


def test_single_report_shows_temperature_and_suite_hash() -> None:
    text = render_report(summarize(batch("b", **SUITE_V2, temperature=0.7), RUNS))
    assert "temperature 0.7" in text and "(sha256 aaaaaaaaaaaa)" in text


def test_synthetic_cases_are_tagged_in_the_by_case_table() -> None:
    runs = [
        replace(rec("injection_in_abstract", "edge"), synthetic=True),
        rec("calc_basic", "single_tool"),
    ]
    text = render_report(summarize(batch("b"), runs))
    assert "| injection_in_abstract `synthetic` | edge | 1/1 |" in text
    assert "| calc_basic | single_tool | 1/1 |" in text


def test_cli_writes_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "evals.sqlite"
    with ResultStore(db) as store:
        for label, runs in [("a", RUNS), ("b", RUNS[:3])]:
            batch_id = store.create_batch(label, {"model": "m"}, git_commit=None)
            for record in runs:
                store.save_run(batch_id, record)

    assert report.main(["--label", "a", "--db", str(db), "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "a.md").read_text(encoding="utf-8").startswith("# Eval report: a")
    assert "Overall: **2/4** pass" in capsys.readouterr().out

    args = ["--label", "a", "--compare", "b", "--db", str(db), "--out-dir", str(tmp_path)]
    assert report.main(args) == 0
    assert "| calc_basic | 2/3 | 2/3 | +0 |" in (tmp_path / "a_vs_b.md").read_text("utf-8")
    # Neither batch recorded a suite hash: the warning also goes to stderr.
    assert "WARNING: suite hash unknown for a, b" in capsys.readouterr().err

    assert report.main(["--label", "zzz", "--db", str(db), "--out-dir", str(tmp_path)]) == 2
    assert "No batch labelled 'zzz'" in capsys.readouterr().out
