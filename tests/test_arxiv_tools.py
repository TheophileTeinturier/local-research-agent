"""The arXiv tools end to end through the registry, on recorded fixtures (no network).

Every failure must come back as `ToolResult(ok=False)` with an actionable message: the
registry's `execute` is what the agent loop calls, so that is what is tested.
"""

import datetime

import httpx
import pytest

from fakes import ARXIV_FIXTURES, FakeClock, FakeSleep
from local_research_agent.tools import DEFAULT_MAX_OUTPUT_CHARS, ToolRegistry
from local_research_agent.tools.arxiv import ArxivClient, make_arxiv_tools
from local_research_agent.tools.arxiv.models import PaperDetails, PaperSummary
from local_research_agent.tools.arxiv.query import SearchQuery
from local_research_agent.tools.arxiv.testing import FakeArxivTransport, offline_arxiv_client
from local_research_agent.tools.arxiv.tools import (
    SEARCH_OUTPUT_MAX_CHARS,
    format_paper_details,
    format_search_results,
)

SEARCH_ARGS = {
    "query": "graph neural networks protein-protein interaction prediction",
    "sort": "date",
    "max_results": 5,
}


@pytest.fixture
def transport() -> FakeArxivTransport:
    return FakeArxivTransport.from_directory(ARXIV_FIXTURES)


@pytest.fixture
def registry(transport: FakeArxivTransport) -> ToolRegistry:
    return ToolRegistry(make_arxiv_tools(offline_arxiv_client(transport)))


def registry_on(respond: object) -> ToolRegistry:
    clock = FakeClock()
    client = ArxivClient(
        http=httpx.Client(transport=httpx.MockTransport(respond)),  # type: ignore[arg-type]
        clock=clock,
        sleep=FakeSleep(clock),
    )
    return ToolRegistry(make_arxiv_tools(client))


# --- search_arxiv ----------------------------------------------------------------------------


def test_search_returns_compact_summaries(
    registry: ToolRegistry, transport: FakeArxivTransport
) -> None:
    result = registry.execute("search_arxiv", SEARCH_ARGS)

    assert result.ok, result.content
    assert transport.unmatched == []  # the fixture was found: same request as recorded
    lines = result.content.splitlines()
    assert lines[0] == (
        'arXiv search: graph AND neural AND networks AND "protein protein" AND interaction '
        "AND prediction (newest first) | 5 results:"
    )
    assert lines[1].startswith(
        "[1] 2606.01781 | 2026 | cs.AI | Structure-Guided Adaptive Propagation for "
        "Protein-Protein Interaction Site Prediction"
    )
    assert lines[2].startswith("    Enqiang Zhu et al.: Accurate prediction of protein-protein")
    assert lines[-1].startswith("Ids above are the only valid ones.")
    assert "get_paper_details" in lines[-1]
    assert len(lines) == 1 + 2 * 5 + 1
    assert len(result.content) < 2500  # 5 papers: 2088 chars on this fixture
    assert "truncated" not in result.metadata


def test_search_reports_ignored_words(registry: ToolRegistry) -> None:
    result = registry.execute("search_arxiv", {"query": "recent papers on qwxzvkjhq zzqplmnb"})
    assert "ignored common words: recent, papers, on" in result.content


def test_zero_results_is_an_actionable_observation(registry: ToolRegistry) -> None:
    result = registry.execute("search_arxiv", {"query": "qwxzvkjhq zzqplmnb"})
    assert result.ok  # the search worked; it found nothing
    assert "No papers matched all keywords. Try fewer or broader keywords" in result.content


def test_zero_results_with_category_suggests_dropping_it() -> None:
    query = SearchQuery.build("graph", "cs.LG", 5, "relevance")
    assert "drop the category filter" in format_search_results(query, [])


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"query": ""}, "query: String should have at least 1 character"),
        ({"query": "graph", "max_results": 11}, "max_results: Input should be less than"),
        ({"query": "graph", "max_results": 0}, "max_results: Input should be greater than"),
        ({"query": "graph", "sort": "newest"}, "sort: Input should be 'relevance' or 'date'"),
        ({"query": "graph", "category": "cs.LG OR all:x"}, "Invalid category"),
        ({"query": "graph", "limit": 3}, "limit: Extra inputs are not permitted"),
    ],
)
def test_invalid_search_arguments(registry: ToolRegistry, args: dict, message: str) -> None:
    result = registry.execute("search_arxiv", args)
    assert not result.ok
    assert result.metadata["error_type"] == "invalid_arguments"
    assert message in result.content


@pytest.mark.parametrize("raw", [None, "", " ", "null"])
def test_empty_category_means_no_filter(
    registry: ToolRegistry, transport: FakeArxivTransport, raw: object
) -> None:
    registry.execute("search_arxiv", {**SEARCH_ARGS, "category": raw})
    assert "cat%3A" not in str(transport.requests[-1].url)


def test_stopword_only_query_fails_before_any_request(
    registry: ToolRegistry, transport: FakeArxivTransport
) -> None:
    result = registry.execute("search_arxiv", {"query": "the latest papers"})
    assert not result.ok
    assert result.metadata["error_type"] == "invalid_arguments"
    assert "no usable keywords" in result.content
    assert transport.requests == []


def test_search_output_respects_budget_in_the_worst_case() -> None:
    # 10 papers with every field at its cap.
    worst = PaperSummary(
        arxiv_id="hep-th/9901001",
        title="t" * 500,
        first_author="a" * 500,
        et_al=True,
        year=2026,
        primary_category="cond-mat.str-el",
        snippet="w " * 500,
    )
    query = SearchQuery.build("x " * 8, "cond-mat.str-el", 10, "date")
    text = format_search_results(query, [worst] * 10)
    assert len(text) <= SEARCH_OUTPUT_MAX_CHARS < DEFAULT_MAX_OUTPUT_CHARS
    assert "more results not shown" in text
    assert text.endswith("call get_paper_details with its id.")


# --- get_paper_details -----------------------------------------------------------------------


def test_details_returns_the_full_record(registry: ToolRegistry) -> None:
    result = registry.execute("get_paper_details", {"arxiv_id": "2606.01781"})

    assert result.ok, result.content
    lines = result.content.splitlines()
    assert lines[0] == (
        "arXiv:2606.01781 (v1) | Structure-Guided Adaptive Propagation for Protein-Protein "
        "Interaction Site Prediction"
    )
    assert lines[1] == (
        "Authors: Enqiang Zhu, Yizi Liu, Yilong Luo, Yao Chen, Yu Zhang, Baoshan Ma"
    )
    assert lines[2] == "Published: 2026-06-01"
    assert lines[3] == "Categories: cs.AI (primary)"
    assert lines[4] == "PDF: https://arxiv.org/pdf/2606.01781v1"
    assert lines[5].startswith("Abstract: Accurate prediction of protein-protein interaction")
    assert lines[5].endswith("jointly drive these improvements.")  # full, not cut
    assert len(result.content) < 2000


def test_details_accepts_arxiv_prefix(
    registry: ToolRegistry, transport: FakeArxivTransport
) -> None:
    assert registry.execute("get_paper_details", {"arxiv_id": "arXiv:2606.01781"}).ok
    assert transport.unmatched == []


def test_unknown_id_is_not_found(registry: ToolRegistry) -> None:
    result = registry.execute("get_paper_details", {"arxiv_id": "2301.99999"})
    assert not result.ok
    assert result.metadata["error_type"] == "not_found"
    assert result.content == (
        "Error: No arXiv paper has the id '2301.99999'. Use an id exactly as returned by "
        "search_arxiv, or search again; do not guess ids."
    )


@pytest.mark.parametrize(
    "bad_id",
    ["1234.5678 OR all:x", "2301.12345,2302.00001", "all:electron", "2301.1234", "x" * 65],
)
def test_injected_or_malformed_id_never_reaches_arxiv(
    registry: ToolRegistry, transport: FakeArxivTransport, bad_id: str
) -> None:
    result = registry.execute("get_paper_details", {"arxiv_id": bad_id})
    assert not result.ok
    assert result.metadata["error_type"] == "invalid_arguments"
    assert result.content.startswith("Error: Invalid arguments for 'get_paper_details': arxiv_id")
    assert transport.requests == []


def test_details_output_with_many_authors_and_long_abstract() -> None:
    paper = PaperDetails(
        arxiv_id="2301.12345",
        version="v2",
        title="T",
        authors=[f"Author {i}" for i in range(10)],
        authors_total=42,
        published=datetime.date(2023, 1, 2),
        updated=datetime.date(2023, 3, 4),
        primary_category="cs.LG",
        categories=["cs.LG", "q-bio.QM"],
        abstract="word " * 1000,
        pdf_url="https://arxiv.org/pdf/2301.12345v2",
    )
    text = format_paper_details(paper)
    assert "Author 9 (+32 more)" in text
    assert "Published: 2023-01-02 | Updated: 2023-03-04" in text
    assert "Categories: cs.LG (primary), q-bio.QM" in text
    assert len(text) < 2000


# --- transport failures become observations -----------------------------------------------


@pytest.mark.parametrize(
    ("respond", "error_type", "message"),
    [
        (
            lambda r: httpx.Response(503),
            "arxiv_unavailable",
            "arXiv is temporarily unavailable (timeout or server error). Try again once with "
            "the same arguments",
        ),
        (
            lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow")),
            "arxiv_unavailable",
            "Try again once with the same arguments",
        ),
        (
            lambda r: httpx.Response(429, headers={"Retry-After": "30"}),
            "arxiv_rate_limited",
            "arXiv is rate limiting requests (retry possible in about 30 s). Do not retry now",
        ),
        (lambda r: httpx.Response(400), "arxiv_rejected", "rejected the request (HTTP 400)"),
        (lambda r: httpx.Response(200, text="<<"), "arxiv_bad_response", "unreadable response"),
    ],
)
@pytest.mark.parametrize(
    ("tool", "args"),
    [("search_arxiv", {"query": "graph"}), ("get_paper_details", {"arxiv_id": "2301.12345"})],
)
def test_transport_failures_are_observations(
    respond: object, error_type: str, message: str, tool: str, args: dict
) -> None:
    result = registry_on(respond).execute(tool, args)
    assert not result.ok
    assert result.metadata["error_type"] == error_type
    assert result.content.startswith("Error: ")
    assert message in result.content


def test_unmatched_requests_are_recorded(
    registry: ToolRegistry, transport: FakeArxivTransport
) -> None:
    result = registry.execute("search_arxiv", {"query": "never recorded"})
    assert result.ok and "No papers matched" in result.content
    assert len(transport.unmatched) == 1


def test_single_result_header_is_singular() -> None:
    paper = PaperSummary(
        arxiv_id="2301.12345",
        title="T",
        first_author="A",
        et_al=False,
        year=2023,
        primary_category="cs.LG",
        snippet="S",
    )
    text = format_search_results(SearchQuery.build("graph", None, 1, "date"), [paper])
    assert text.splitlines()[0].endswith("| 1 result:")
    assert "[1] 2301.12345 | 2023 | cs.LG | T\n    A: S" in text
