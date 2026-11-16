"""`evals.run` end to end with a scripted FakeLLM in place of Ollama: exit codes and warning."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from evals import run
from evals.store import ResultStore
from fakes import FakeLLM, text_response, tool_call_response
from local_research_agent.llm import LLMResponse

OK = text_response("OK")  # the warm-up's answer


@pytest.fixture(autouse=True)
def _reset_package_logger() -> Iterator[None]:
    """run.main configures the package logger (propagate=False): undo it for later tests."""
    yield
    logger = logging.getLogger("local_research_agent")
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def use_fake_llm(monkeypatch: pytest.MonkeyPatch, script: list[LLMResponse]) -> None:
    fake = FakeLLM(script)

    class Context:
        def __enter__(self) -> FakeLLM:
            return fake

        def __exit__(self, *exc: object) -> None:
            return None

    class Factory:
        @staticmethod
        def from_env(model: str | None = None, host: str | None = None) -> Context:
            return Context()

    monkeypatch.setattr(run, "OllamaClient", Factory)
    monkeypatch.setattr(run, "git_commit", lambda: "test")


def write_suite(tmp_path: Path, **arxiv: Any) -> Path:
    case = {
        "id": "unknown_id",
        "category": "error",
        "question": "Give me the abstract of arXiv paper 2406.99999",
        "arxiv": arxiv,
        "checks": [{"name": "tool_called", "params": {"tool": "get_paper_details"}}],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    return path


FALLBACK_SEARCH: list[LLMResponse] = [
    OK,
    tool_call_response(("get_paper_details", {"arxiv_id": "2406.99999"})),
    tool_call_response(("search_arxiv", {"query": "2406.99999"})),
    text_response("No such paper."),
]


def args(tmp_path: Path, suite: Path) -> list[str]:
    db = tmp_path / "evals.sqlite"
    return ["--suite", str(suite), "--label", "t", "--db", str(db), "--log-level", "ERROR"]


def test_too_many_inconclusive_runs_exit_non_zero_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_llm(monkeypatch, FALLBACK_SEARCH)

    code = run.main(args(tmp_path, write_suite(tmp_path)))  # no default_search: a miss

    assert code == run.EXIT_TOO_MANY_INCONCLUSIVE == 3
    captured = capsys.readouterr()
    assert "MISS" in captured.out
    assert "warning: 1/1 runs are inconclusive (fixture_miss), more than 10%" in captured.err
    # The batch is stored anyway: the warning is about trusting it, not about losing it.
    with ResultStore(tmp_path / "evals.sqlite") as store:
        batch = store.latest_batch("t")
        assert batch is not None and len(store.load_runs(batch.id)) == 1


def test_default_search_makes_the_same_run_conclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_llm(monkeypatch, FALLBACK_SEARCH)

    code = run.main(args(tmp_path, write_suite(tmp_path, default_search="empty")))

    assert code == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out and "1/1 pass" in captured.out
    assert "warning" not in captured.err
