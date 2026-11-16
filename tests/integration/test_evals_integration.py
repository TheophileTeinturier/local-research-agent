"""One suite case on a real Ollama server. Opt-in: `uv run pytest -m integration`.

No network besides Ollama: the case's arXiv is the fake transport. Asserts that the harness
produces a complete record, not that the model passes (that is what the evaluation measures).
"""

import httpx
import pytest

from evals.runner import EvalConfig, run_case, warm_up
from evals.suite import load_suite, resolve_suite_path, select_cases
from local_research_agent.llm import OllamaClient

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


def test_one_case_end_to_end(llm: OllamaClient) -> None:
    [case] = select_cases(load_suite(resolve_suite_path("minimal")), ["calc_basic"])
    config = EvalConfig(model=llm.model, label="integration")

    assert warm_up(llm, config).ok
    record = run_case(case, config, 0, llm=llm)

    assert record.harness_error is None and record.fixture_misses == ()
    assert record.trace.stop_reason is not None and record.trace.iterations >= 1
    assert record.prompt_tokens > 0 and record.llm_latency_ms > 0
    assert [c.name for c in record.checks][-1] == "stop_reason_is(value=final_answer)"
    assert record.passed in (True, False)
