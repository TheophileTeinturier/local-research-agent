"""The per-case fake arXiv, exercised through the real tools (registry.execute), offline."""

from pathlib import Path

import pytest

from evals.arxiv_fake import CaseArxivTransport, single_entry_feed, truncate_feed
from evals.suite import ArxivSpec
from fakes import make_eval_fixtures
from local_research_agent.tools import ToolRegistry, build_registry
from local_research_agent.tools.arxiv.parsing import parse_feed


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    return make_eval_fixtures(tmp_path)


def setup(fixtures: Path, **spec: object) -> tuple[CaseArxivTransport, ToolRegistry]:
    transport = CaseArxivTransport(ArxivSpec.model_validate(spec), fixtures)
    return transport, build_registry(arxiv_transport=transport)


def test_search_is_served_whatever_the_query_and_truncated(fixtures: Path) -> None:
    transport, registry = setup(fixtures, search="search.xml")
    result = registry.execute("search_arxiv", {"query": "anything at all", "max_results": 2})
    assert result.ok
    assert "2 results" in result.content and "2606.01781" in result.content
    full = registry.execute("search_arxiv", {"query": "other words", "max_results": 10})
    assert "5 results" in full.content  # the fixture has 5: never more than recorded
    assert [r.outcome for r in transport.served] == ["fixture", "fixture"]
    assert transport.misses == []


def test_truncate_feed_keeps_the_first_entries() -> None:
    feed = (Path(__file__).parent / "fixtures/arxiv/search_gnn_ppi.xml").read_text("utf-8")
    papers = parse_feed(truncate_feed(feed, 3)).papers
    assert [p.arxiv_id for p in papers] == [p.arxiv_id for p in parse_feed(feed).papers[:3]]
    assert truncate_feed(feed, None) == feed


def test_lookup_from_mapping_from_search_hits_and_not_found(fixtures: Path) -> None:
    feed = (fixtures / "search.xml").read_text("utf-8")
    second_hit = parse_feed(feed).papers[1].arxiv_id
    transport, registry = setup(fixtures, search="search.xml", papers={"2606.01781": "paper.xml"})

    assert registry.execute("get_paper_details", {"arxiv_id": "2606.01781v1"}).ok  # version ignored
    hit = registry.execute("get_paper_details", {"arxiv_id": second_hit})
    assert hit.ok and hit.content.startswith(f"arXiv:{second_hit}")
    missing = registry.execute("get_paper_details", {"arxiv_id": "2406.99999"})
    assert not missing.ok and missing.metadata["error_type"] == "not_found"

    assert [r.outcome for r in transport.served] == ["fixture", "search_hit", "not_found"]
    assert transport.misses == []  # an unknown id is an answer, not a miss


def test_single_entry_feed() -> None:
    feed = (Path(__file__).parent / "fixtures/arxiv/search_gnn_ppi.xml").read_text("utf-8")
    one = single_entry_feed(feed, "2606.01781")
    assert one is not None
    parsed = parse_feed(one)
    assert [p.arxiv_id for p in parsed.papers] == ["2606.01781"] and parsed.total_results == 1
    assert single_entry_feed(feed, "1111.11111") is None


def test_search_without_fixture_is_a_miss(fixtures: Path) -> None:
    transport, registry = setup(fixtures)  # no `search` declared
    result = registry.execute("search_arxiv", {"query": "graph"})
    assert result.ok and "No papers matched" in result.content  # an empty feed was served
    [miss] = transport.misses
    assert "search_query=" in miss and "graph" in miss


def test_default_search_empty_answers_any_search_without_a_miss(fixtures: Path) -> None:
    transport, registry = setup(fixtures, default_search="empty")
    for query in ("2406.99999", "graph neural network"):
        result = registry.execute("search_arxiv", {"query": query, "sort": "date"})
        assert result.ok and "No papers matched" in result.content
    # Lookups are unaffected: an unknown id is still arXiv's "not found".
    missing = registry.execute("get_paper_details", {"arxiv_id": "2406.99999"})
    assert missing.metadata["error_type"] == "not_found"
    outcomes = [r.outcome for r in transport.served]
    assert outcomes == ["default_empty", "default_empty", "not_found"]
    assert transport.misses == []


def test_default_search_does_not_override_a_failure_mode(fixtures: Path) -> None:
    transport, registry = setup(fixtures, default_search="empty", failure_mode="http_503")
    assert not registry.execute("search_arxiv", {"query": "graph"}).ok
    assert [r.outcome for r in transport.served] == ["http_503"]


def test_http_503_failure_mode(fixtures: Path) -> None:
    transport, registry = setup(
        fixtures, papers={"2606.01781": "paper.xml"}, failure_mode="http_503"
    )
    for _ in range(2):
        result = registry.execute("get_paper_details", {"arxiv_id": "2606.01781"})
        assert not result.ok and result.metadata["error_type"] == "arxiv_unavailable"
    assert [r.outcome for r in transport.served] == ["http_503", "http_503"]  # never cached
    assert transport.misses == []


def test_timeout_failure_mode_is_instant(fixtures: Path) -> None:
    transport, registry = setup(fixtures, search="search.xml", failure_mode="timeout")
    result = registry.execute("search_arxiv", {"query": "graph"})
    assert not result.ok and result.metadata["error_type"] == "arxiv_unavailable"
    assert result.metadata["duration_ms"] < 1000
    assert transport.served[0].outcome == "timeout"
