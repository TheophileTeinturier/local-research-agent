"""Hits a real Ollama server. Opt-in: `uv run pytest -m integration`.

Model outputs are not deterministic, so these assert on the behaviour that matters (was the
tool called, does the answer contain the right fact), not on exact wording.
"""

import httpx
import pytest

from local_research_agent.agent import ToolStep, run_once
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


OPTIONS = ChatOptions(temperature=0.0, think=False)


def test_arithmetic_question_uses_calculator(llm: OllamaClient) -> None:
    result = run_once(llm, build_default_registry(), "What is 17 * 23 + 4?", options=OPTIONS)

    tool_steps = [step for step in result.trace if isinstance(step, ToolStep)]
    assert tool_steps, "expected a calculator call"
    assert tool_steps[0].call.name == "calculator"
    assert tool_steps[0].result.ok
    assert "395" in result.answer


def test_general_knowledge_question_answers_directly(llm: OllamaClient) -> None:
    result = run_once(
        llm, build_default_registry(), "What is the capital of France?", options=OPTIONS
    )

    assert not any(isinstance(step, ToolStep) for step in result.trace)
    assert "Paris" in result.answer
