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


def test_cli_prints_answer_and_trace_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeLLM(
        [
            tool_call_response(("calculator", {"expression": "17 * 23 + 4"})),
            text_response("The answer is 395."),
        ]
    )

    class FakeClientFactory:
        @staticmethod
        def from_env(model: str | None = None, host: str | None = None) -> "FakeContext":
            return FakeContext()

    class FakeContext:
        def __enter__(self) -> FakeLLM:
            return fake

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(cli, "OllamaClient", FakeClientFactory)

    assert cli.main(["What is 17 * 23 + 4?"]) == 0

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == "The answer is 395."
    assert "llm  -> tool_calls: calculator" in captured.out
    assert 'tool calculator {"expression": "17 * 23 + 4"} -> ok: 17 * 23 + 4 = 395' in captured.out
    assert "llm  -> answer" in captured.out
    # Logs went to stderr, as JSON lines.
    events = [json.loads(line)["event"] for line in captured.err.splitlines()]
    assert events.count("llm_call") == 2
    assert events.count("tool_execution") == 1
