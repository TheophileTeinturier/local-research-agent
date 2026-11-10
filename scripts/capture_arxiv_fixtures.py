"""Record real arXiv API responses as fixtures for offline tests and evaluation.

Run manually, never from the tests (it uses the network):

    uv run python scripts/capture_arxiv_fixtures.py [output_dir]   # default tests/fixtures/arxiv

The requests are built by the same code as the tools (`SearchQuery.params`, `lookup_params`)
and sent through `ArxivClient.fetch_text`, so the 3 s rate limit applies.
`index.json` maps each request key to its file; `FakeArxivTransport`
replays them. The paper looked up in scenario 3 is the first hit of scenario 1, so a recorded
"search -> details" run is consistent.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

from local_research_agent.tools.arxiv.client import ArxivClient, request_key
from local_research_agent.tools.arxiv.parsing import parse_feed
from local_research_agent.tools.arxiv.query import SearchQuery, lookup_params, parse_arxiv_id

SEARCH_KEYWORDS = "graph neural networks protein-protein interaction prediction"
NO_MATCH_KEYWORDS = "qwxzvkjhq zzqplmnb"
UNKNOWN_ID = "2301.99999"  # well-formed, but no such paper (checked live)


def capture(client: ArxivClient, out: Path, name: str, params: dict[str, str]) -> str:
    """Fetch once, save `<name>.xml`, return the body. Prints the size and latency."""
    start = time.perf_counter()
    body = client.fetch_text(params)
    elapsed = time.perf_counter() - start
    # newline="": keep the body byte-for-byte (no CRLF translation on Windows).
    (out / f"{name}.xml").write_text(body, encoding="utf-8", newline="")
    print(f"{name}: {len(body)} bytes in {elapsed:.2f} s (includes rate-limit wait)")
    return body


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/arxiv")
    out.mkdir(parents=True, exist_ok=True)
    search = SearchQuery.build(SEARCH_KEYWORDS, None, 5, "date").params()
    no_match = SearchQuery.build(NO_MATCH_KEYWORDS, None, 5, "relevance").params()
    unknown = lookup_params(parse_arxiv_id(UNKNOWN_ID))

    with ArxivClient() as client:
        body = capture(client, out, "search_gnn_ppi", search)
        first_id = parse_feed(body).papers[0].arxiv_id
        lookup = lookup_params(parse_arxiv_id(first_id))
        scenarios = [
            ("search_gnn_ppi", f"search_arxiv {SEARCH_KEYWORDS!r}, sort=date, 5", search),
            ("search_no_results", f"search_arxiv {NO_MATCH_KEYWORDS!r}: zero hits", no_match),
            ("paper_details", f"get_paper_details {first_id!r} (1st hit above)", lookup),
            ("unknown_id", f"get_paper_details {UNKNOWN_ID!r}: no such paper", unknown),
        ]
        for name, _description, params in scenarios[1:]:
            capture(client, out, name, params)

    index = {
        "captured_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source": "https://export.arxiv.org/api/query",
        "fixtures": [
            {"key": request_key(params), "file": f"{name}.xml", "description": description}
            for name, description, params in scenarios
        ],
    }
    index_text = json.dumps(index, indent=2) + "\n"
    (out / "index.json").write_text(index_text, encoding="utf-8", newline="")
    print(f"wrote {out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
