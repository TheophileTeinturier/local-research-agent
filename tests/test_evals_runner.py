"""run_case / run_batch / warm_up driven by a scripted FakeLLM: no Ollama, no network."""

from pathlib import Path
from typing import Any

import pytest

from evals.records import RunRecord
from evals.runner import EvalConfig, run_batch, run_case, warm_up
from evals.suite import Case
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt_variant": "nope"}, "Unknown prompt variant"),
        ({"repetitions": 0}, "repetitions must be >= 1"),
        ({"label": "a/b"}, "label 'a/b' must match"),
        ({"label": ""}, "must match"),
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
