"""Every check with a passing and a failing trace. Checks are pure: hand-built traces only."""

import json
import re
from typing import Any

import pytest

from evals.checks import (
    CHECKS,
    CheckSpecError,
    answer_matches,
    answer_not_matches,
    answer_signals_failure,
    check_label,
    cited_ids_count,
    cited_ids_subset_of_opened_ids,
    cited_ids_subset_of_tool_ids,
    extract_arxiv_ids,
    heuristic_answer_matches,
    max_iterations,
    max_tool_errors,
    no_tool_called,
    run_check,
    stop_reason_is,
    tool_arg_covers,
    tool_arg_equals,
    tool_arg_min,
    tool_called,
    tool_not_called,
    validate_check,
)
from evals.records import RunTrace, ToolCallTrace


def call(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    ok: bool = True,
    content: str = "",
    repeated: bool = False,
) -> ToolCallTrace:
    return ToolCallTrace(1, name, json.dumps(args or {}), ok, content, repeated)


def trace(
    answer: str | None = "",
    *calls: ToolCallTrace,
    question: str = "q?",
    stop: str | None = "final_answer",
    iterations: int = 1,
    tool_errors: int = 0,
) -> RunTrace:
    return RunTrace(question, answer, stop, iterations, calls, tool_errors)


SEARCH_OUT = "[1] 2606.01781 | 2026 | cs.AI | Title\n[2] 2605.07838v2 | 2026 | q-bio | T"


# --- ids ------------------------------------------------------------------------------------


def test_extract_arxiv_ids_normalizes_and_ignores_numbers() -> None:
    text = (
        "See arXiv:1706.02216v3, 2606.01781 and hep-th/9901001v1, math.GT/0309136. "
        "Not ids: 443272593, 12.1706.02216, 1706.022161, 2026-12-25, 294.0"
    )
    assert extract_arxiv_ids(text) == {
        "1706.02216",
        "2606.01781",
        "hep-th/9901001",
        "math/0309136",
    }
    assert extract_arxiv_ids(None) == set()


# --- tool checks ----------------------------------------------------------------------------


def test_tool_called() -> None:
    t = trace("", call("calculator"), call("calculator", repeated=True))
    assert tool_called(t, tool="calculator")[0]
    assert not tool_called(t, tool="calculator", min=2)[0]  # the repeat is not an execution
    assert not tool_called(t, tool="current_date")[0]


def test_tool_not_called() -> None:
    assert tool_not_called(trace("", call("calculator")), tool="search_arxiv")[0]
    passed, detail = tool_not_called(
        trace("", call("search_arxiv", repeated=True)), tool="search_arxiv"
    )
    assert not passed and "requested 1" in detail


def test_no_tool_called() -> None:
    assert no_tool_called(trace("answer"))[0]
    passed, detail = no_tool_called(trace("", call("calculator")))
    assert not passed and "calculator" in detail


def test_tool_arg_min() -> None:
    t = trace(
        "", call("search_arxiv", {"max_results": 3}), call("search_arxiv", {"max_results": 10})
    )
    assert tool_arg_min(t, tool="search_arxiv", arg="max_results", value=6)[0]
    passed, detail = tool_arg_min(
        trace("", call("search_arxiv", {"max_results": 5})),
        tool="search_arxiv",
        arg="max_results",
        value=6,
    )
    assert not passed and "[5]" in detail
    # Absent argument (the tool default would apply) does not count; a numeric string does.
    assert not tool_arg_min(
        trace("", call("search_arxiv", {})), tool="search_arxiv", arg="max_results", value=1
    )[0]
    assert tool_arg_min(
        trace("", call("search_arxiv", {"max_results": "8"})),
        tool="search_arxiv",
        arg="max_results",
        value=6,
    )[0]
    assert not tool_arg_min(trace(""), tool="search_arxiv", arg="max_results", value=6)[0]


def test_tool_arg_equals() -> None:
    ok = trace("", call("get_paper_details", {"arxiv_id": "1706.02216"}))
    assert tool_arg_equals(ok, tool="get_paper_details", arg="arxiv_id", value="1706.02216")[0]
    prefixed = trace("", call("get_paper_details", {"arxiv_id": "arXiv:1706.02216"}))
    passed, detail = tool_arg_equals(
        prefixed, tool="get_paper_details", arg="arxiv_id", value="1706.02216"
    )
    assert not passed and "arXiv:1706.02216" in detail


COVERED = ["1609.02907", "1706.02216"]


def test_tool_arg_covers_counts_ok_and_failed_calls_and_ignores_versions() -> None:
    t = trace(
        "",
        details("1609.02907v4"),  # version stripped
        details(" 1706.02216", ok=False),  # a failed call counts too (arXiv down, say)
    )
    passed, detail = tool_arg_covers(t, tool="get_paper_details", arg="arxiv_id", values=COVERED)
    assert passed and "all of" in detail
    # A value given with a version in the case is normalized as well.
    assert tool_arg_covers(t, tool="get_paper_details", arg="arxiv_id", values=["1706.02216v1"])[0]


def test_tool_arg_covers_fails_on_a_missing_value() -> None:
    only_one = trace("", details("1609.02907"), call("search_arxiv", {"arxiv_id": "1706.02216"}))
    passed, detail = tool_arg_covers(
        only_one, tool="get_paper_details", arg="arxiv_id", values=COVERED
    )
    assert not passed and "missing ['1706.02216']" in detail
    # A repetition warning is not a call; a prefixed id is another value.
    for second in (details("1706.02216", repeated=True), details("arXiv:1706.02216")):
        t = trace("", details("1609.02907"), second)
        assert not tool_arg_covers(t, tool="get_paper_details", arg="arxiv_id", values=COVERED)[0]
    no_calls = trace("")
    assert not tool_arg_covers(no_calls, tool="get_paper_details", arg="arxiv_id", values=COVERED)[
        0
    ]


def test_tool_arg_covers_non_string_values() -> None:
    t = trace("", call("search_arxiv", {"max_results": 5}), call("search_arxiv", {"x": {"a": 1}}))
    assert tool_arg_covers(t, tool="search_arxiv", arg="max_results", values=[5])[0]
    assert not tool_arg_covers(t, tool="search_arxiv", arg="max_results", values=[10])[0]
    assert tool_arg_covers(t, tool="search_arxiv", arg="x", values=[{"a": 1}])[0]  # unhashable


def test_raw_unparseable_arguments_are_empty() -> None:
    raw = ToolCallTrace(1, "search_arxiv", "not json", False, "Error: ...")
    assert raw.args() == {}
    assert not tool_arg_min(trace("", raw), tool="search_arxiv", arg="max_results", value=1)[0]


# --- answer checks --------------------------------------------------------------------------


def test_answer_matches_and_not_matches() -> None:
    assert answer_matches(trace("It is 294."), regex=r"\b294\b")[0]
    assert not answer_matches(trace("It is 2940."), regex=r"\b294\b")[0]
    assert not answer_matches(trace(None), regex=r".*x")[0]  # no answer never matches
    assert answer_not_matches(trace("fine"), regex="(?i)graphsage")[0]
    assert not answer_not_matches(trace("GraphSAGE is"), regex="(?i)graphsage")[0]
    assert answer_not_matches(trace(None), regex="x")[0]


@pytest.mark.parametrize(
    "answer", ["443272593", "443,272,593", "443 272 593", "443\u202f272\u202f593"]
)
def test_big_number_with_or_without_separators(answer: str) -> None:
    regex = "443[,\\s\\u202f.']?272[,\\s\\u202f.']?593"  # the calc_big pattern
    assert answer_matches(trace(f"= {answer}"), regex=regex)[0]


def test_answer_signals_failure_is_heuristic() -> None:
    assert CHECKS["answer_signals_failure"].heuristic
    assert answer_signals_failure(trace("The paper was not found."), regex="(?i)not found")[0]
    assert not answer_signals_failure(trace("Here is the abstract"), regex="(?i)not found")[0]


# --- cited ids ------------------------------------------------------------------------------


def test_cited_ids_subset_of_tool_ids() -> None:
    search = call("search_arxiv", content=SEARCH_OUT)
    good = trace("See 2606.01781 and arXiv:2605.07838v1.", search)
    assert cited_ids_subset_of_tool_ids(good)[0]
    passed, detail = cited_ids_subset_of_tool_ids(trace("See 2301.00001.", search))
    assert not passed and "2301.00001" in detail


def test_ids_from_error_results_do_not_count_but_the_question_does() -> None:
    error = call(
        "get_paper_details", ok=False, content="Error: No arXiv paper has the id '2406.99999'."
    )
    answer = "Paper 2406.99999 was not found."
    assert cited_ids_subset_of_tool_ids(trace(answer, error, question="Abstract of 2406.99999?"))[0]
    assert not cited_ids_subset_of_tool_ids(
        trace(answer, error, question="Abstract of 2406.99999?"), include_question=False
    )[0]
    assert not cited_ids_subset_of_tool_ids(trace("2301.99999 exists", error, question="q"))[0]


def details(arxiv_id: Any, *, ok: bool = True, repeated: bool = False) -> ToolCallTrace:
    return call("get_paper_details", {"arxiv_id": arxiv_id}, ok=ok, repeated=repeated)


def test_cited_ids_subset_of_opened_ids_passes_when_every_cited_id_was_opened() -> None:
    t = trace(
        "1. arXiv:2606.01781v2 2. 2605.07838",
        call("search_arxiv", content=SEARCH_OUT),
        details("2606.01781"),
        details("arXiv:2605.07838v1"),  # the argument is normalized like the answer
    )
    passed, detail = cited_ids_subset_of_opened_ids(t)
    assert passed and detail == "2 cited id(s), all opened"
    assert cited_ids_subset_of_opened_ids(trace("No ids here."))[0]  # nothing cited


def test_ids_seen_only_in_search_results_or_the_question_were_not_opened() -> None:
    search = call("search_arxiv", content=SEARCH_OUT)
    t = trace("See 2606.01781 and 2605.07838.", search, details("2606.01781"), question="q")
    assert cited_ids_subset_of_tool_ids(t)[0]  # the looser check is satisfied...
    passed, detail = cited_ids_subset_of_opened_ids(t)  # ...the strict one is not
    assert not passed and detail.endswith("['2605.07838']")
    in_question = trace("2406.99999 was not found.", question="Abstract of 2406.99999?")
    assert not cited_ids_subset_of_opened_ids(in_question)[0]


@pytest.mark.parametrize(
    "opening",
    [
        details("2606.01781", ok=False),  # failed: not_found, arXiv down...
        details("2606.01781", repeated=True),  # a repetition warning, not an execution
        call("search_arxiv", {"arxiv_id": "2606.01781"}),  # another tool
        details(2606.01781),  # not a string
        call("get_paper_details", {"id": "2606.01781"}),  # another argument
    ],
)
def test_calls_that_do_not_open_a_paper(opening: ToolCallTrace) -> None:
    assert not cited_ids_subset_of_opened_ids(trace("See 2606.01781.", opening))[0]


def test_a_repeat_of_a_successful_opening_still_counts() -> None:
    t = trace("See 2606.01781.", details("2606.01781"), details("2606.01781", repeated=True))
    assert cited_ids_subset_of_opened_ids(t)[0]


def test_cited_ids_count() -> None:
    answer = "1. 2606.01781 2. arXiv:2605.07838 3. 2603.15047v2 (again 2606.01781)"
    assert cited_ids_count(trace(answer), exactly=3)[0]
    passed, detail = cited_ids_count(trace(answer), exactly=2)
    assert not passed and "3 distinct" in detail
    assert cited_ids_count(trace("No papers matched."), exactly=0)[0]


@pytest.mark.parametrize(
    ("n_ids", "bounds", "passed"),
    [
        (5, {"min": 5, "max": 10}, True),
        (10, {"min": 5, "max": 10}, True),
        (4, {"min": 5, "max": 10}, False),
        (11, {"min": 5, "max": 10}, False),
        (7, {"min": 5}, True),
        (2, {"min": 5}, False),
        (0, {"max": 2}, True),
        (3, {"max": 2}, False),
    ],
)
def test_cited_ids_count_bounds(n_ids: int, bounds: dict[str, int], passed: bool) -> None:
    answer = " ".join(f"2601.{i:05d}" for i in range(1, n_ids + 1))
    verdict, detail = cited_ids_count(trace(answer), **bounds)
    assert verdict is passed and f"{n_ids} distinct" in detail
    assert (">= 5" in detail) is ("min" in bounds)


def test_heuristic_answer_matches_is_heuristic_and_reported_apart() -> None:
    assert CHECKS["heuristic_answer_matches"].heuristic
    assert heuristic_answer_matches(trace("Which topic do you mean?"), regex=r"(?i)\?")[0]
    assert not heuristic_answer_matches(trace("Here are papers."), regex=r"\?")[0]
    result = run_check("heuristic_answer_matches", {"regex": "x"}, trace("y"))
    assert result.heuristic and not result.passed
    # The original name still works, with the same verdicts and its own label.
    legacy = run_check("answer_signals_failure", {"regex": "y"}, trace("y"))
    assert legacy.heuristic and legacy.passed and legacy.name == "answer_signals_failure(regex=y)"


# --- counters -------------------------------------------------------------------------------


def test_counters_and_stop_reason() -> None:
    t = trace("x", iterations=4, tool_errors=2, stop="too_many_tool_errors")
    assert max_iterations(t, n=4)[0] and not max_iterations(t, n=3)[0]
    assert max_tool_errors(t, n=2)[0] and not max_tool_errors(t, n=1)[0]
    assert stop_reason_is(t, value="too_many_tool_errors")[0]
    passed, detail = stop_reason_is(t, value="final_answer")
    assert not passed and "expected final_answer" in detail


# --- registry and validation ----------------------------------------------------------------


def test_run_check_builds_a_labelled_result() -> None:
    result = run_check(
        "tool_called", {"tool": "calculator", "min": 1}, trace("", call("calculator"))
    )
    assert result.name == "tool_called(tool=calculator, min=1)"
    assert result.passed and not result.heuristic
    assert run_check("answer_matches", {"regex": "x"}, trace("x"), heuristic=True).heuristic
    assert check_label("no_tool_called", {}) == "no_tool_called()"


def test_checks_are_pure() -> None:
    t = trace("2606.01781", call("search_arxiv", {"max_results": 10}, content=SEARCH_OUT))
    before = repr(t)
    for name, params in [
        ("tool_arg_min", {"tool": "search_arxiv", "arg": "max_results", "value": 6}),
        ("cited_ids_subset_of_tool_ids", {}),
        ("cited_ids_subset_of_opened_ids", {}),
        ("cited_ids_count", {"exactly": 1}),
    ]:
        assert run_check(name, params, t) == run_check(name, params, t)
    assert repr(t) == before


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("nope", {}, "unknown check 'nope'"),
        ("tool_called", {}, "missing param(s) ['tool']"),
        ("tool_called", {"tool": "x", "max": 1}, "unknown param(s) ['max']"),
        ("tool_called", {"tool": "x", "min": "2"}, "param 'min'='2' is not"),
        ("tool_called", {"tool": "x", "min": True}, "param 'min'=True is not"),
        ("answer_matches", {"regex": "("}, "invalid regex"),
        ("cited_ids_subset_of_tool_ids", {"include_question": "yes"}, "is not"),
        ("cited_ids_subset_of_opened_ids", {"tool": "x"}, "unknown param(s) ['tool']"),
        ("cited_ids_count", {}, "give `exactly`, or `min` and/or `max`"),
        ("cited_ids_count", {"exactly": 3, "max": 4}, "cannot be combined"),
        ("cited_ids_count", {"min": 6, "max": 5}, "min 6 > max 5"),
        ("cited_ids_count", {"min": "5"}, "param 'min'='5' is not"),
        ("cited_ids_count", {"max": True}, "param 'max'=True is not"),
        ("tool_arg_covers", {"tool": "t", "arg": "a"}, "missing param(s) ['values']"),
        ("tool_arg_covers", {"tool": "t", "arg": "a", "values": "1706.02216"}, "is not"),
        # Python 3.11+: a global flag must start the pattern (hence "(?i)\?|..." in the suite).
        ("heuristic_answer_matches", {"regex": r"\?|(?i)which"}, "invalid regex"),
    ],
)
def test_validate_check_rejects_bad_specs(name: str, params: dict[str, Any], message: str) -> None:
    with pytest.raises(CheckSpecError, match=re.escape(message)):
        validate_check(name, params)


def test_validate_check_accepts_good_specs() -> None:
    validate_check("tool_arg_min", {"tool": "t", "arg": "a", "value": 6})  # int for a float
    validate_check("tool_arg_equals", {"tool": "t", "arg": "a", "value": {"any": "json"}})
    validate_check("no_tool_called", {})
    validate_check("cited_ids_count", {"exactly": 0})
    validate_check("cited_ids_count", {"min": 5, "max": 10})
    validate_check("cited_ids_count", {"max": 2})
    validate_check("tool_arg_covers", {"tool": "t", "arg": "a", "values": ["x", 1]})
