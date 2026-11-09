import json
import logging

import pytest

from fakes import FakeLLM, text_response, tool_call_response
from local_research_agent import cli
from local_research_agent.observability import configure_logging, log_event


@pytest.fixture(autouse=True)
def _reset_package_logger():
    yield
    logger = logging.getLogger("local_research_agent")
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def test_logs_are_json_on_stderr_only(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    log_event(logging.getLogger("local_research_agent.test"), "llm_call", eval_count=25)

    captured = capsys.readouterr()
    assert captured.out == ""
    entry = json.loads(captured.err.strip())
    assert entry["event"] == "llm_call"
    assert entry["eval_count"] == 25
    assert entry["level"] == "info"


def _patch_client(monkeypatch: pytest.MonkeyPatch, fake: FakeLLM) -> None:
    class FakeContext:
        def __enter__(self) -> FakeLLM:
            return fake

        def __exit__(self, *exc: object) -> None:
            return None

    class FakeClientFactory:
        @staticmethod
        def from_env(model: str | None = None, host: str | None = None) -> FakeContext:
            return FakeContext()

    monkeypatch.setattr(cli, "OllamaClient", FakeClientFactory)


def test_cli_prints_answer_and_trace_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_client(
        monkeypatch,
        FakeLLM(
            [
                tool_call_response(("calculator", {"expression": "17 * 23 + 4"})),
                text_response("The answer is 395."),
            ]
        ),
    )

    assert cli.main(["What is 17 * 23 + 4?"]) == 0

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == "The answer is 395."
    assert lines[2].startswith(
        "trace: stop_reason=final_answer | iterations=2 | tokens in=30 out=13"
    )
    assert "  1. llm  -> tool_calls: calculator | tokens in=20 out=8" in captured.out
    assert 'tool calculator {"expression":"17 * 23 + 4"} -> ok: 17 * 23 + 4 = 395' in captured.out
    assert "  2. llm  -> answer" in captured.out
    # Logs went to stderr, as JSON lines.
    events = [json.loads(line)["event"] for line in captured.err.splitlines()]
    assert events.count("llm_call") == 2
    assert events.count("tool_execution") == 1
    assert events.count("agent_iteration") == 2
    assert events[-1] == "agent_finished"


def test_cli_explains_a_stop_without_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A deliberately looping model: it asks for a new calculation forever.
    looping = [tool_call_response(("calculator", {"expression": f"{i} + 1"})) for i in range(3)]
    _patch_client(monkeypatch, FakeLLM(looping))

    assert cli.main(["loop forever", "--max-iterations", "3"]) == 1

    out = capsys.readouterr().out
    assert out.startswith(
        "No answer: the model still wanted tools after the maximum number of steps "
        "(max_iterations)."
    )
    assert "stop_reason=max_iterations | iterations=3" in out
