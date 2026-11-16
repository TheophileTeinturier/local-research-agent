"""The case loader: valid suites load with defaults; invalid ones fail with a precise message."""

import datetime
import json
from pathlib import Path
from typing import Any

import pytest

from evals.suite import (
    CASES_DIR,
    DEFAULT_FROZEN_NOW,
    SuiteError,
    load_suite,
    resolve_suite_path,
    select_cases,
)
from fakes import make_eval_fixtures


def case(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "calc_basic",
        "category": "single_tool",
        "question": "What is 12% of 2450?",
        "checks": [{"name": "tool_called", "params": {"tool": "calculator"}}],
    }
    return {**base, **overrides}


def write_suite(tmp_path: Path, *cases: dict[str, Any]) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"cases": list(cases)}), encoding="utf-8")
    return path


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    return make_eval_fixtures(tmp_path / "fixtures")


def test_valid_case_gets_defaults(tmp_path: Path, fixtures: Path) -> None:
    suite = load_suite(write_suite(tmp_path, case()), fixtures_dir=fixtures)
    [loaded] = suite.cases
    assert loaded.frozen_now == DEFAULT_FROZEN_NOW == datetime.datetime(2026, 11, 9, 10, 18, 18)
    assert loaded.expected_stop_reason == "final_answer"
    assert loaded.arxiv.search is None and loaded.arxiv.failure_mode is None
    # The expected stop reason becomes an ordinary, last check.
    assert [c.name for c in loaded.all_checks()] == ["tool_called", "stop_reason_is"]
    assert loaded.all_checks()[-1].params == {"value": "final_answer"}


def test_full_case(tmp_path: Path, fixtures: Path) -> None:
    full = case(
        arxiv={
            "search": "search.xml",
            "papers": {"2606.01781": "paper.xml"},
            "failure_mode": "timeout",
        },
        frozen_now="2026-01-02T03:04:05",
        expected_stop_reason="too_many_tool_errors",
        checks=[{"name": "answer_signals_failure", "params": {"regex": "x"}, "heuristic": True}],
    )
    [loaded] = load_suite(write_suite(tmp_path, full), fixtures_dir=fixtures).cases
    assert loaded.arxiv.failure_mode == "timeout"
    assert loaded.frozen_now == datetime.datetime(2026, 1, 2, 3, 4, 5)
    assert loaded.checks[0].heuristic is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"checks": [{"name": "nope"}]},
            "case 'calc_basic' (cases.0) checks.0: Value error, unknown check 'nope'",
        ),
        ({"checks": [{"name": "tool_called"}]}, "missing param(s) ['tool']"),
        ({"checks": [{"name": "answer_matches", "params": {"regex": "("}}]}, "invalid regex"),
        ({"checks": []}, "checks: Tuple should have at least 1 item"),
        ({"id": "Bad Id"}, "case 'Bad Id' (cases.0) id: String should match pattern"),
        ({"expected_stop_reason": "done"}, "unknown stop reason 'done'"),
        (
            {"arxiv": {"failure_mode": "http_500"}},
            "arxiv.failure_mode: Input should be 'http_503' or 'timeout'",
        ),
        ({"frozen_now": "yesterday"}, "frozen_now: Input should be a valid datetime"),
        ({"surprise": 1}, "surprise: Extra inputs are not permitted"),
        ({"arxiv": {"search": "missing.xml"}}, "fixture 'missing.xml' not found"),
        (
            {"arxiv": {"default_search": "full"}},
            "arxiv.default_search: Input should be 'empty'",
        ),
        (
            {"arxiv": {"search": "search.xml", "default_search": "empty"}},
            "give either `search` (a fixture) or `default_search`, not both",
        ),
    ],
)
def test_invalid_case_has_a_clear_error(
    tmp_path: Path, fixtures: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(SuiteError) as info:
        load_suite(write_suite(tmp_path, case(**overrides)), fixtures_dir=fixtures)
    assert message in str(info.value)
    assert str(tmp_path / "suite.json") in str(info.value)  # the file is named


def test_default_search(tmp_path: Path, fixtures: Path) -> None:
    suite = load_suite(
        write_suite(tmp_path, case(arxiv={"default_search": "empty"}), case(id="plain")),
        fixtures_dir=fixtures,
    )
    assert [c.arxiv.default_search for c in suite.cases] == ["empty", None]  # None: a miss


def test_not_found_fixture_is_required(tmp_path: Path, fixtures: Path) -> None:
    (fixtures / "not_found.xml").unlink()
    with pytest.raises(SuiteError, match=r"'not_found\.xml' \(served for unknown ids\) not found"):
        load_suite(write_suite(tmp_path, case()), fixtures_dir=fixtures)


def test_duplicate_ids(tmp_path: Path, fixtures: Path) -> None:
    with pytest.raises(SuiteError, match="duplicate case id 'calc_basic'"):
        load_suite(write_suite(tmp_path, case(), case()), fixtures_dir=fixtures)


def test_bad_files(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="suite file not found"):
        load_suite(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text('{"cases": [', encoding="utf-8")
    with pytest.raises(SuiteError, match="invalid JSON at line 1"):
        load_suite(broken)


def test_select_cases(tmp_path: Path, fixtures: Path) -> None:
    suite = load_suite(write_suite(tmp_path, case(), case(id="other")), fixtures_dir=fixtures)
    assert [c.id for c in select_cases(suite, None)] == ["calc_basic", "other"]
    assert [c.id for c in select_cases(suite, ["other"])] == ["other"]
    with pytest.raises(SuiteError, match=r"unknown case id\(s\) \['x'\]"):
        select_cases(suite, ["x"])


def test_resolve_suite_path() -> None:
    assert resolve_suite_path("minimal") == CASES_DIR / "minimal.json"
    assert resolve_suite_path("some/dir/x.json") == Path("some/dir/x.json")


def test_the_shipped_minimal_suite_is_valid() -> None:
    suite = load_suite(resolve_suite_path("minimal"))  # real fixtures dir
    assert len(suite.cases) == 10
    unknown_id = next(c for c in suite.cases if c.id == "unknown_id_wellformed")
    assert unknown_id.arxiv.default_search == "empty"  # a fallback search is not a miss
    assert {c.category for c in suite.cases} == {
        "single_tool",
        "multi_step",
        "no_tool",
        "tool_choice",
        "error",
        "edge",
    }
