"""A scripted research run: FakeLLM drives search -> details -> answer on recorded arXiv data."""

import re

from fakes import ARXIV_FIXTURES, FakeClock, FakeLLM, text_response, tool_call_response
from local_research_agent.agent import AgentConfig, StopReason, run_agent
from local_research_agent.tools import build_default_registry
from local_research_agent.tools.arxiv.testing import FakeArxivTransport, offline_arxiv_client

QUESTION = (
    "Find 3 recent papers on graph neural networks for protein-protein interaction "
    "prediction and list them"
)
SEARCH = {
    "query": "graph neural networks protein-protein interaction prediction",
    "sort": "date",
    "max_results": 5,
}
ANSWER = (
    "1. arXiv:2606.01781 - Structure-Guided Adaptive Propagation for Protein-Protein "
    "Interaction Site Prediction (2026)\n2. arXiv:2605.07838 (2026)\n3. arXiv:2603.15047 (2026)"
)

ID_RE = re.compile(r"\b\d{4}\.\d{4,5}\b")


def test_search_then_details_then_answer() -> None:
    transport = FakeArxivTransport.from_directory(ARXIV_FIXTURES)
    registry = build_default_registry(arxiv_client=offline_arxiv_client(transport))
    llm = FakeLLM(
        [
            tool_call_response(("search_arxiv", SEARCH)),
            tool_call_response(("get_paper_details", {"arxiv_id": "2606.01781"})),
            text_response(ANSWER),
        ]
    )

    result = run_agent(llm, registry, AgentConfig(), QUESTION, clock=FakeClock())

    assert result.stop_reason is StopReason.FINAL_ANSWER
    assert [(c.name, c.ok) for c in result.state.tool_calls] == [
        ("search_arxiv", True),
        ("get_paper_details", True),
    ]
    # Both requests were served from the recordings, none fell through to "empty".
    assert transport.unmatched == []
    assert len(transport.requests) == 2

    # The model was offered the arXiv tools, and the system prompt asks it to cite ids.
    offered = [schema["function"]["name"] for schema in llm.calls[0].tools or []]
    assert {"search_arxiv", "get_paper_details"} <= set(offered)
    assert "Cite the arXiv id" in llm.calls[0].messages[0]["content"]

    # The second LLM call saw the search summaries; the third saw the paper details.
    search_observation = llm.calls[1].messages[-1]
    assert search_observation["tool_name"] == "search_arxiv"
    assert search_observation["content"].count("\n    ") == 5  # 5 summaries
    details_observation = llm.calls[2].messages[-1]
    assert details_observation["content"].startswith("arXiv:2606.01781 (v1) | Structure-Guided")

    # Every id in the answer was returned by the search tool (none invented).
    returned_ids = set(ID_RE.findall(search_observation["content"]))
    cited_ids = set(ID_RE.findall(result.final_answer or ""))
    assert cited_ids and cited_ids <= returned_ids


def test_invented_id_gets_an_actionable_error_and_the_run_continues() -> None:
    transport = FakeArxivTransport.from_directory(ARXIV_FIXTURES)
    registry = build_default_registry(arxiv_client=offline_arxiv_client(transport))
    llm = FakeLLM(
        [
            tool_call_response(("get_paper_details", {"arxiv_id": "2301.99999"})),
            text_response("I could not find that paper."),
        ]
    )

    result = run_agent(llm, registry, AgentConfig(), "Summarise 2301.99999", clock=FakeClock())

    assert result.stop_reason is StopReason.FINAL_ANSWER
    [record] = result.state.tool_calls
    assert not record.ok and record.error_type == "not_found"
    assert "do not guess ids" in llm.calls[1].messages[-1]["content"]
