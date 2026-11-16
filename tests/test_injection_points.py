"""The small injection points added for the eval harness: frozen clock, fake arXiv transport,
seed passthrough, prompt variants. Defaults must be unchanged."""

import datetime
import json
import logging
from collections.abc import Iterator

import httpx
import pytest

from fakes import ARXIV_FIXTURES, FakeLLM, text_response
from local_research_agent import cli
from local_research_agent.agent import (
    PROMPT_VARIANTS,
    SYSTEM_PROMPT,
    AgentConfig,
    get_system_prompt,
)
from local_research_agent.llm import ChatOptions, OllamaClient
from local_research_agent.tools import build_default_registry, build_registry
from local_research_agent.tools.arxiv.testing import FakeArxivTransport

FROZEN = datetime.datetime(2026, 11, 9, 10, 18, 18)


def test_build_registry_defaults_match_the_default_registry() -> None:
    assert build_registry().names == build_default_registry().names


def test_frozen_clock_drives_current_date() -> None:
    registry = build_registry(clock=lambda: FROZEN)
    assert registry.execute("current_date", {}).content == "2026-11-09T10:18:18"


def test_arxiv_transport_receives_the_tool_requests() -> None:
    transport = FakeArxivTransport.from_directory(ARXIV_FIXTURES)
    registry = build_registry(arxiv_transport=transport)
    result = registry.execute("get_paper_details", {"arxiv_id": "2606.01781"})
    assert result.ok
    assert len(transport.requests) == 1 and transport.unmatched == []


def test_seed_goes_into_ollama_options() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})

    with OllamaClient(transport=httpx.MockTransport(handler)) as client:
        client.chat([], options=ChatOptions(seed=7))
        client.chat([], options=ChatOptions())
    assert bodies[0]["options"] == {"temperature": 0.2, "num_ctx": 8192, "seed": 7}
    assert "seed" not in bodies[1]["options"]  # type: ignore[operator]


def test_prompt_variants() -> None:
    assert get_system_prompt("baseline") is SYSTEM_PROMPT  # unchanged prompt
    assert AgentConfig().system_prompt is SYSTEM_PROMPT  # default unchanged
    workflow = get_system_prompt("research_workflow")
    assert workflow.startswith(SYSTEM_PROMPT)
    assert "at least 3*N" in workflow and "get_paper_details" in workflow
    assert set(PROMPT_VARIANTS) == {"baseline", "research_workflow"}
    with pytest.raises(ValueError, match="known: baseline, research_workflow"):
        get_system_prompt("nope")


@pytest.fixture
def _reset_package_logger() -> Iterator[None]:
    """cli.main configures the package logger (propagate=False): undo it for later tests."""
    yield
    logger = logging.getLogger("local_research_agent")
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


@pytest.mark.usefixtures("_reset_package_logger")
def test_cli_prompt_variant_reaches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM([text_response("ok")])

    class Factory:
        @staticmethod
        def from_env(model: str | None = None, host: str | None = None) -> object:
            class Context:
                def __enter__(self) -> FakeLLM:
                    return fake

                def __exit__(self, *exc: object) -> None:
                    return None

            return Context()

    monkeypatch.setattr(cli, "OllamaClient", Factory)
    assert cli.main(["hi", "--prompt-variant", "research_workflow", "--log-level", "ERROR"]) == 0
    assert fake.calls[0].messages[0]["content"] == PROMPT_VARIANTS["research_workflow"]
