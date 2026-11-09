"""Hits a real Ollama server. Opt-in: `uv run pytest -m integration`.

Model outputs are not deterministic, so these assert on the behaviour that matters (which
tools were called, does the answer contain the right fact), not on exact wording.
"""

import datetime

import httpx
import pytest

from local_research_agent.agent import AgentConfig, StopReason, run_agent
from local_research_agent.llm import ChatOptions, OllamaClient
from local_research_agent.tools import build_default_registry

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def llm():
    client = OllamaClient.from_env()
    try:
        tags = httpx.get(f"{client.host}/api/tags", timeout=3).json()
    except httpx.HTTPError:
        pytest.skip(f"Ollama is not reachable at {client.host}")
    if client.model not in {m["name"] for m in tags.get("models", [])}:
        pytest.skip(f"Model {client.model} is not pulled (ollama pull {client.model})")
    yield client
    client.close()


CONFIG = AgentConfig(llm_options=ChatOptions(temperature=0.0, think=False))


def test_two_step_question_chains_date_then_calculator(llm: OllamaClient) -> None:
    # The second tool call depends on the first result: the year is unknown to the model.
    result = run_agent(
        llm, build_default_registry(), CONFIG, "What is the current year multiplied by 3?"
    )

    assert result.stop_reason is StopReason.FINAL_ANSWER
    called = [c.name for c in result.state.tool_calls if c.ok]
    assert called[:1] == ["current_date"]
    assert "calculator" in called
    expected = str(datetime.date.today().year * 3)
    assert expected in (result.final_answer or "").replace(",", "")


def test_general_knowledge_question_answers_directly(llm: OllamaClient) -> None:
    result = run_agent(llm, build_default_registry(), CONFIG, "What is the capital of France?")

    assert result.stop_reason is StopReason.FINAL_ANSWER
    assert result.state.tool_calls == []
    assert "Paris" in (result.final_answer or "")
