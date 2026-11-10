"""Hits the real arXiv API with a single query. Opt-in: `uv run pytest -m integration`.

One request only (the Terms of Use ask for at most one request every 3 s), and assertions on
structure, not on which papers come back, since the index changes daily.
"""

import pytest

from local_research_agent.tools.arxiv import ArxivClient
from local_research_agent.tools.arxiv.errors import ArxivTimeout, ArxivUnavailable

pytestmark = pytest.mark.integration


def test_real_search_returns_parsed_summaries() -> None:
    with ArxivClient() as client:
        try:
            papers = client.search("graph neural network protein", max_results=3, sort="date")
        except (ArxivTimeout, ArxivUnavailable) as exc:
            pytest.skip(f"arXiv unreachable: {exc!r}")

    assert 1 <= len(papers) <= 3
    for paper in papers:
        assert paper.arxiv_id and "v" not in paper.arxiv_id.split(".")[-1]
        assert paper.title and paper.title != "(untitled)"
        assert paper.year is not None and paper.year >= 2007
        assert len(paper.snippet) <= 200
