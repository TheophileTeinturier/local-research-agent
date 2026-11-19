"""run_case / run_batch / warm_up driven by a scripted FakeLLM: no Ollama, no network."""

import json
from pathlib import Path
from typing import Any

import pytest

from evals.records import RunRecord
from evals.runner import EvalConfig, run_batch, run_case, warm_up
from evals.suite import FIXTURES_DIR, Case, resolve_suite_path
from fakes import FakeLLM, make_eval_fixtures, text_response, tool_call_response
from local_research_agent.agent import PROMPT_VARIANTS
from local_research_agent.llm import LLMError, LLMResponse

CALC = {
    "id": "calc_basic",
    "category": "single_tool",
    "question": "What is 12% of 2450?",
    "checks": [
        {"name": "tool_called", "params": {"tool": "calculator"}},
        {"name": "answer_matches", "params": {"regex": "\\b294\\b"}},
    ],
}


def make_case(**overrides: Any) -> Case:
    return Case.model_validate({**CALC, **overrides})


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    return make_eval_fixtures(tmp_path)


def run(
    case: Case,
    script: list[LLMResponse | BaseException],
    fixtures: Path,
    *,
    repetition: int = 0,
    **config: Any,
) -> tuple[RunRecord, FakeLLM]:
    llm = FakeLLM(script)
    record = run_case(case, EvalConfig(**config), repetition, llm=llm, fixtures_dir=fixtures)
    return record, llm


CALC_SCRIPT: list[LLMResponse | BaseException] = [
    tool_call_response(("calculator", {"expression": "0.12 * 2450"})),
    text_response("12% of 2450 is 294."),
]


def test_passing_run(fixtures: Path) -> None:
    record, llm = run(
        make_case(), CALC_SCRIPT, fixtures, repetition=2, prompt_variant="research_workflow"
    )

    assert record.passed is True and record.passed_with_heuristics is True
    assert [c.passed for c in record.checks] == [True, True, True]
    assert record.checks[-1].name == "stop_reason_is(value=final_answer)"
    assert record.trace.stop_reason == "final_answer"
    assert record.trace.final_answer == "12% of 2450 is 294."
    assert (record.trace.iterations, record.llm_calls, len(record.trace.tool_calls)) == (2, 2, 1)
    assert record.trace.tool_calls[0].content == "0.12 * 2450 = 294"
    assert (record.prompt_tokens, record.completion_tokens) == (30, 13)  # from the fake metrics
    # The config reached the agent: seed = repetition index, the chosen system prompt.
    assert record.seed == 2 and llm.calls[0].options is not None and llm.calls[0].options.seed == 2
    assert llm.calls[0].messages[0]["content"] == PROMPT_VARIANTS["research_workflow"]
    assert record.prompt_variant == "research_workflow" and record.harness_error is None


def test_per_call_prompt_tokens_and_end_to_end_time_are_recorded(fixtures: Path) -> None:
    record, _ = run(make_case(), CALC_SCRIPT, fixtures)
    # The fake reports 20 prompt tokens for the tool-call turn, 10 for the answer.
    assert record.llm_prompt_tokens == (20, 10) and record.peak_prompt_tokens == 20
    assert record.e2e_wall_time_ms is not None and record.e2e_wall_time_ms >= 0
    assert record.trajectory == (("calculator", '{"expression":"0.12 * 2450"}'),)
    assert record.mcp_startup_ms is None  # local: nothing to start


def test_harness_error_has_no_end_to_end_time(fixtures: Path) -> None:
    record, _ = run(make_case(), CALC_SCRIPT[:1], fixtures)
    assert record.harness_error is not None
    assert record.e2e_wall_time_ms is None and record.peak_prompt_tokens is None


def test_temperature_reaches_the_llm(fixtures: Path) -> None:
    _, llm = run(make_case(), CALC_SCRIPT, fixtures, temperature=0.7)
    assert all(c.options is not None and c.options.temperature == 0.7 for c in llm.calls)


def test_fixed_seed_overrides_the_repetition_index(fixtures: Path) -> None:
    record, llm = run(make_case(), CALC_SCRIPT, fixtures, repetition=4, seed=42)
    assert record.seed == 42 and llm.calls[0].options.seed == 42  # type: ignore[union-attr]


def test_failing_run_reports_the_failed_check(fixtures: Path) -> None:
    script: list[LLMResponse | BaseException] = [text_response("About 300.")]
    record, _ = run(make_case(), script, fixtures)
    assert record.passed is False
    failed = [c.name for c in record.checks if not c.passed]
    assert failed == ["tool_called(tool=calculator)", "answer_matches(regex=\\b294\\b)"]


def test_frozen_clock_reaches_current_date(fixtures: Path) -> None:
    case = make_case(
        id="date_diff",
        frozen_now="2026-11-09T10:18:18",
        checks=[{"name": "answer_matches", "params": {"regex": "\\b46\\b"}}],
    )
    script: list[LLMResponse | BaseException] = [
        tool_call_response(("current_date", {})),
        text_response("46 days."),
    ]
    record, _ = run(case, script, fixtures)
    assert record.trace.tool_calls[0].content == "2026-11-09T10:18:18"
    assert record.passed is True


def test_fixture_miss_is_excluded_but_checks_are_kept(fixtures: Path) -> None:
    case = make_case(checks=[{"name": "no_tool_called"}])  # no arxiv fixtures declared
    script: list[LLMResponse | BaseException] = [
        tool_call_response(("search_arxiv", {"query": "flights"})),
        text_response("Nothing found."),
    ]
    record, _ = run(case, script, fixtures)
    assert len(record.fixture_misses) == 1
    assert record.excluded and record.passed is None and record.passed_with_heuristics is None
    assert [c.passed for c in record.checks] == [False, True]  # still computed, for the report


def test_default_search_keeps_a_lookup_case_conclusive(fixtures: Path) -> None:
    # unknown_id_wellformed's shape: the lookup fails, the model falls back to a search.
    case = make_case(
        id="unknown_id",
        question="Give me the abstract of arXiv paper 2406.99999",
        arxiv={"default_search": "empty"},
        checks=[{"name": "tool_called", "params": {"tool": "get_paper_details"}}],
    )
    script: list[LLMResponse | BaseException] = [
        tool_call_response(("get_paper_details", {"arxiv_id": "2406.99999"})),
        tool_call_response(("search_arxiv", {"query": "2406.99999"})),
        text_response("No paper 2406.99999 was found on arXiv."),
    ]
    record, _ = run(case, script, fixtures)
    assert record.fixture_misses == () and not record.excluded
    assert record.passed is True
    assert "No papers matched" in record.trace.tool_calls[1].content

    # The same run without default_search is a fixture miss: inconclusive, not counted.
    no_default = make_case(**{**case.model_dump(), "arxiv": {}})
    record, _ = run(no_default, script, fixtures)
    assert len(record.fixture_misses) == 1 and record.passed is None


def test_exception_in_the_agent_becomes_a_harness_error(fixtures: Path) -> None:
    # The script ends after the tool call: FakeLLM raises AssertionError (not an LLMError),
    # which run_agent lets through on purpose (a bug, not an expected failure).
    record, _ = run(make_case(), CALC_SCRIPT[:1], fixtures)
    assert record.harness_error is not None and "AssertionError" in record.harness_error
    assert record.passed is None and record.checks == ()
    assert record.trace.stop_reason is None and record.trace.question == CALC["question"]


def test_exception_in_the_harness_becomes_a_harness_error(tmp_path: Path) -> None:
    record, llm = run(make_case(), CALC_SCRIPT, tmp_path / "empty")  # no not_found.xml
    assert record.harness_error is not None and "FileNotFoundError" in record.harness_error
    assert llm.calls == []  # failed while building the case's world, before the agent ran


def test_stop_reason_mismatch_fails(fixtures: Path) -> None:
    record, _ = run(make_case(), [LLMError("connection refused")], fixtures)
    assert record.trace.stop_reason == "llm_error" and record.agent_error == "connection refused"
    assert record.passed is False
    assert record.checks[-1].name == "stop_reason_is(value=final_answer)"
    assert not record.checks[-1].passed


def test_expected_non_final_stop_reason_can_pass(fixtures: Path) -> None:
    case = make_case(
        arxiv={"papers": {"2606.01781": "paper.xml"}, "failure_mode": "http_503"},
        expected_stop_reason="too_many_tool_errors",
        checks=[{"name": "max_tool_errors", "params": {"n": 3}}],
    )
    details = tool_call_response(("get_paper_details", {"arxiv_id": "2606.01781"}))
    record, _ = run(case, [details, details, details], fixtures)
    assert record.trace.stop_reason == "too_many_tool_errors"
    assert record.trace.tool_errors == 3 and record.passed is True


def test_run_batch_is_repetition_major_and_reports_each_record(fixtures: Path) -> None:
    cases = [make_case(), make_case(id="second")]
    seen: list[tuple[str, int]] = []
    llm = FakeLLM(CALC_SCRIPT * 4)
    records = run_batch(
        cases,
        EvalConfig(repetitions=2),
        llm=llm,
        on_record=lambda r: seen.append((r.case_id, r.repetition)),
        fixtures_dir=fixtures,
    )
    order = [("calc_basic", 0), ("second", 0), ("calc_basic", 1), ("second", 1)]
    assert [(r.case_id, r.repetition) for r in records] == order == seen
    assert all(r.passed for r in records)


def test_warm_up() -> None:
    ok = warm_up(FakeLLM([text_response("OK")]), EvalConfig(num_ctx=4096))
    assert ok.ok and ok.error is None
    failed = warm_up(FakeLLM([LLMError("Ollama is down")]), EvalConfig())
    assert not failed.ok and failed.error == "Ollama is down"


def test_warm_up_uses_the_batch_options() -> None:
    llm = FakeLLM([text_response("OK")])
    warm_up(llm, EvalConfig(num_ctx=4096, think=True))
    options = llm.calls[0].options
    assert options is not None and (options.num_ctx, options.think) == (4096, True)
    assert llm.calls[0].tools is None  # no tool schemas: just load the model


# --- shipped cases, end to end (real suite file, real eval fixtures, scripted model) ---------


def shipped(case_id: str) -> Case:
    """One case of the shipped suite, validated on its own.

    Only THIS case's fixtures must exist: a missing recording for another case fails that
    case's tests (and `test_the_shipped_minimal_suite_is_valid`), not this one.
    """
    raw = json.loads(resolve_suite_path("minimal").read_text(encoding="utf-8"))
    case = Case.model_validate(next(c for c in raw["cases"] if c["id"] == case_id))
    for name in filter(None, [case.arxiv.search, *case.arxiv.papers.values()]):
        assert (FIXTURES_DIR / name).is_file(), f"{case_id}: fixture {name} not recorded yet"
    return case


def run_shipped(case_id: str, script: list[LLMResponse | BaseException]) -> RunRecord:
    return run_case(shipped(case_id), EvalConfig(), 0, llm=FakeLLM(script))  # FIXTURES_DIR


def verdicts(record: RunRecord) -> dict[str, bool]:
    return {c.name.split("(")[0] + ("" if not c.heuristic else "*"): c.passed for c in record.checks}


OPEN_BOTH = tool_call_response(
    ("get_paper_details", {"arxiv_id": "1609.02907"}),
    ("get_paper_details", {"arxiv_id": "1706.02216"}),
)


def test_compare_two_papers_end_to_end() -> None:
    answer = (
        "Kipf & Welling's GCN (arXiv:1609.02907) applies a spectral graph convolution to the "
        "whole graph (transductive); GraphSAGE (arXiv:1706.02216) learns inductive aggregators "
        "that generalize to unseen nodes."
    )
    record = run_shipped("compare_two_papers", [OPEN_BOTH, text_response(answer)])

    assert record.harness_error is None and record.fixture_misses == ()
    assert [c.ok for c in record.trace.tool_calls] == [True, True]  # both lookups served
    assert record.trace.tool_calls[0].content.startswith("arXiv:1609.02907")
    assert record.passed is True, [c for c in record.checks if not c.passed]
    assert not record.synthetic


def test_compare_two_papers_fails_when_it_opens_one_and_searches() -> None:
    script = [
        tool_call_response(("get_paper_details", {"arxiv_id": "1706.02216"})),
        tool_call_response(("search_arxiv", {"query": "graph convolutional networks"})),
        text_response("GraphSAGE (1706.02216) is inductive; GCN uses graph convolution."),
    ]
    record = run_shipped("compare_two_papers", script)
    # default_search "empty": the search is answered, so the run is judged, not inconclusive.
    assert record.fixture_misses == () and record.passed is False
    v = verdicts(record)
    assert not v["tool_arg_covers"] and not v["tool_not_called"]
    assert v["cited_ids_subset_of_tool_ids"]


def test_injection_in_abstract_end_to_end() -> None:
    script = [
        tool_call_response(("get_paper_details", {"arxiv_id": "2601.00001"})),
        text_response(
            "FoldShift is a benchmark for protein structure prediction under distribution "
            "shift. Accuracy drops on orphan sequences and designed proteins."
        ),
    ]
    record = run_shipped("injection_in_abstract", script)

    assert record.harness_error is None and record.fixture_misses == ()
    assert "ignore all previous instructions" in record.trace.tool_calls[0].content  # served
    assert record.synthetic  # carried into the record, so the report can tag it
    assert record.passed is True and record.passed_with_heuristics is True


def test_injection_in_abstract_fails_when_the_model_obeys() -> None:
    script = [
        tool_call_response(("get_paper_details", {"arxiv_id": "2601.00001"})),
        text_response("PWNED"),
    ]
    record = run_shipped("injection_in_abstract", script)
    v = verdicts(record)
    assert record.passed is False and not v["answer_not_matches"]
    assert not v["heuristic_answer_matches*"]  # the topic is not mentioned either


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt_variant": "nope"}, "Unknown prompt variant"),
        ({"repetitions": 0}, "repetitions must be >= 1"),
        ({"label": "a/b"}, "label 'a/b' must match"),
        ({"label": ""}, "must match"),
        ({"temperature": -0.1}, "temperature must be >= 0"),
        ({"temperature": float("nan")}, "temperature must be >= 0"),
    ],
)
def test_eval_config_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvalConfig(**kwargs)


def test_eval_config_json_has_the_prompt_hash() -> None:
    data = EvalConfig(label="x").to_json()
    assert data["label"] == "x" and len(data["system_prompt_sha256"]) == 64
    assert data["limits"]["max_iterations"] == 8  # the guard budgets are part of the config
    other = EvalConfig(prompt_variant="research_workflow").to_json()
    assert other["system_prompt_sha256"] != data["system_prompt_sha256"]
