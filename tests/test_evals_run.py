"""`evals.run` end to end with a scripted FakeLLM in place of Ollama: exit codes and warning."""

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from evals import run
from evals.provenance import OllamaInfo, tool_schemas_sha256
from evals.store import ResultStore
from evals.suite import suite_hashes
from fakes import FakeLLM, make_git_repo, text_response, tool_call_response
from local_research_agent.tools import build_registry
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
    # FakeLLM has no `.host`, so no request would be sent anyway; made explicit here.
    monkeypatch.setattr(run, "read_ollama_info", lambda host, model: OllamaInfo(None, None))


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


def test_batch_config_records_temperature_and_suite_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake_llm(monkeypatch, FALLBACK_SEARCH)
    suite = write_suite(tmp_path, default_search="empty")

    assert run.main([*args(tmp_path, suite), "--temperature", "0.7"]) == 0

    with ResultStore(tmp_path / "evals.sqlite") as store:
        batch = store.latest_batch("t")
    assert batch is not None
    assert batch.config["temperature"] == 0.7
    assert {k: batch.config[k] for k in suite_hashes(suite)} == suite_hashes(suite)
    assert batch.config["cases_sha256"] == hashlib.sha256(suite.read_bytes()).hexdigest()


def test_batch_config_records_ollama_info_and_the_schema_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_llm(monkeypatch, FALLBACK_SEARCH)
    monkeypatch.setattr(run, "read_ollama_info", lambda host, model: OllamaInfo("0.40.14", None))

    assert run.main(args(tmp_path, write_suite(tmp_path, default_search="empty"))) == 0

    with ResultStore(tmp_path / "evals.sqlite") as store:
        batch = store.latest_batch("t")
    assert batch is not None
    assert batch.config["ollama_version"] == "0.40.14"
    assert batch.config["model_digest"] is None  # unreadable: stored as null, run not failed
    assert batch.config["tool_schemas_sha256"] == tool_schemas_sha256(build_registry().schemas())
    assert "ollama 0.40.14, model digest unknown" in capsys.readouterr().out


def test_schema_hash_config_flags_disagreeing_runs() -> None:
    assert run.schema_hash_config(["a"]) == {"tool_schemas_sha256": "a"}
    assert run.schema_hash_config(["a", "b"]) == {
        "tool_schemas_sha256": None,
        "tool_schemas_sha256_all": ["a", "b"],
    }


def test_require_clean_refuses_a_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_git_repo(tmp_path / "repo")
    (repo / "tracked.txt").write_text("edited\n", encoding="utf-8")
    monkeypatch.setattr(run, "REPO_ROOT", repo)
    use_fake_llm(monkeypatch, [])  # an LLM call would fail the test: none must happen

    code = run.main([*args(tmp_path, write_suite(tmp_path)), "--require-clean"])

    assert code == 2
    err = capsys.readouterr().err
    assert "--require-clean: the working tree has 1 uncommitted change(s)" in err
    assert "tracked.txt" in err
    assert not (tmp_path / "evals.sqlite").exists()  # refused before any batch existed


def test_require_clean_ignores_gitignored_files_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_git_repo(tmp_path / "repo")
    (repo / "ignored.txt").write_text("results, caches...\n", encoding="utf-8")
    monkeypatch.setattr(run, "REPO_ROOT", repo)
    use_fake_llm(monkeypatch, FALLBACK_SEARCH)
    suite = write_suite(tmp_path, default_search="empty")  # outside the repo: not a change

    assert run.main([*args(tmp_path, suite), "--require-clean"]) == 0


def test_require_clean_refuses_when_git_cannot_tell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run, "working_tree_changes", lambda repo=None: None)
    use_fake_llm(monkeypatch, [])
    assert run.main([*args(tmp_path, write_suite(tmp_path)), "--require-clean"]) == 2
    assert "cannot read the git state" in capsys.readouterr().err


def test_negative_temperature_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_llm(monkeypatch, [])
    suite = write_suite(tmp_path)
    assert run.main([*args(tmp_path, suite), "--temperature", "-1"]) == 2
    assert "temperature must be >= 0" in capsys.readouterr().err


def test_default_search_makes_the_same_run_conclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_llm(monkeypatch, FALLBACK_SEARCH)

    code = run.main(args(tmp_path, write_suite(tmp_path, default_search="empty")))

    assert code == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out and "1/1 pass" in captured.out
    assert "warning" not in captured.err
