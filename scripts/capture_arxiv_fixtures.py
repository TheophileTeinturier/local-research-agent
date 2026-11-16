"""Record real arXiv API responses as fixtures for offline tests and evaluation.

Run manually, never from the tests (it uses the network):

    uv run python scripts/capture_arxiv_fixtures.py [output_dir]          # tests/fixtures/arxiv
    uv run python scripts/capture_arxiv_fixtures.py --evals [output_dir]  # evals/fixtures/arxiv

The requests are built by the same code as the tools (`SearchQuery.params`, `lookup_params`)
and sent through `ArxivClient.fetch_text`, so the 3 s rate limit applies. No custom
User-Agent is set (httpx's default is sent).

Default mode (4 requests): `index.json` maps each request key to its file; `FakeArxivTransport`
replays them. The paper looked up in scenario 3 is the first hit of scenario 1, so a recorded
"search -> details" run is consistent. Re-running it changes the papers the tests name.

`--evals` mode (3 requests, the default-mode fixtures are left untouched): the files named by
`evals/cases/minimal.json`, served by rule by `evals/arxiv_fake.py`, plus `MANIFEST.json`
(what each file is and the exact request). The zero-hit search is copied from
tests/fixtures/arxiv (no request): a real "no match" response serves any empty-results case.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import time
from pathlib import Path

import httpx

from local_research_agent.tools.arxiv.client import API_URL, ArxivClient, request_key
from local_research_agent.tools.arxiv.errors import ArxivError
from local_research_agent.tools.arxiv.parsing import parse_feed
from local_research_agent.tools.arxiv.query import SearchQuery, lookup_params, parse_arxiv_id

SEARCH_KEYWORDS = "graph neural networks protein-protein interaction prediction"
NO_MATCH_KEYWORDS = "qwxzvkjhq zzqplmnb"
UNKNOWN_ID = "2301.99999"  # well-formed, but no such paper (checked live)

EVAL_SEARCH_RESULTS = 10  # the tool's maximum: research_synthesis wants >= 6 candidates
EVAL_KNOWN_ID = "1706.02216"  # GraphSAGE (Hamilton et al., 2017)
EVAL_UNKNOWN_ID = "2406.99999"  # well-formed; June 2024 had far fewer than 99,999 papers

TESTS_DIR = Path("tests/fixtures/arxiv")
EVALS_DIR = Path("evals/fixtures/arxiv")

# Longer than the client's default 7 s read timeout (sized to stay under the agent's 10 s
# tool timeout). A query arXiv has not cached can take longer than that (observed
# 2026-11-11 for the 10-result search); a manual capture can afford to wait.
CAPTURE_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def capture_client(http: httpx.Client) -> ArxivClient:
    """An `ArxivClient` (same rate limiter) on `http`, which the caller owns and closes."""
    return ArxivClient(http=http)


def capture_http() -> httpx.Client:
    """One connection at most, as the Terms of Use ask; patient timeouts."""
    return httpx.Client(timeout=CAPTURE_TIMEOUT, limits=httpx.Limits(max_connections=1))


def capture(client: ArxivClient, out: Path, name: str, params: dict[str, str]) -> str:
    """Fetch once, save `<name>.xml`, return the body. Prints the size and latency."""
    start = time.perf_counter()
    body = client.fetch_text(params)
    elapsed = time.perf_counter() - start
    # newline="": keep the body byte-for-byte (no CRLF translation on Windows).
    (out / f"{name}.xml").write_text(body, encoding="utf-8", newline="")
    print(f"{name}: {len(body)} bytes in {elapsed:.2f} s (includes rate-limit wait)")
    return body


def capture_tests(out: Path) -> int:
    """The 4 fixtures of the unit tests, and their `index.json`."""
    search = SearchQuery.build(SEARCH_KEYWORDS, None, 5, "date").params()
    no_match = SearchQuery.build(NO_MATCH_KEYWORDS, None, 5, "relevance").params()
    unknown = lookup_params(parse_arxiv_id(UNKNOWN_ID))

    with capture_http() as http, capture_client(http) as client:
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
        "captured_at": _now(),
        "source": API_URL,
        "fixtures": [
            {"key": request_key(params), "file": f"{name}.xml", "description": description}
            for name, description, params in scenarios
        ],
    }
    _write_json(out / "index.json", index)
    return 0


def capture_evals(out: Path) -> int:
    """The evaluation fixtures (3 requests), with sanity checks on what came back."""
    search = SearchQuery.build(SEARCH_KEYWORDS, None, EVAL_SEARCH_RESULTS, "date").params()
    known = lookup_params(parse_arxiv_id(EVAL_KNOWN_ID))
    unknown = lookup_params(parse_arxiv_id(EVAL_UNKNOWN_ID))
    problems = []

    with capture_http() as http, capture_client(http) as client:
        papers = parse_feed(capture(client, out, "search_gnn_ppi_10", search)).papers
        if len(papers) != EVAL_SEARCH_RESULTS:
            problems.append(f"search returned {len(papers)} papers, wanted {EVAL_SEARCH_RESULTS}")
        found = parse_feed(capture(client, out, f"paper_{EVAL_KNOWN_ID}", known)).papers
        if [p.arxiv_id for p in found] != [EVAL_KNOWN_ID]:
            problems.append(f"lookup {EVAL_KNOWN_ID} returned {[p.arxiv_id for p in found]}")
        missing = parse_feed(capture(client, out, "not_found", unknown)).papers
        if missing:
            problems.append(f"{EVAL_UNKNOWN_ID} exists on arXiv: pick another unknown id")

    shutil.copyfile(TESTS_DIR / "search_no_results.xml", out / "search_no_results.xml")
    print("search_no_results: copied from tests/fixtures/arxiv (no request)")

    files = [
        ("search_gnn_ppi_10.xml", f"search {SEARCH_KEYWORDS!r}, sort=date, 10", search),
        (f"paper_{EVAL_KNOWN_ID}.xml", f"lookup {EVAL_KNOWN_ID}", known),
        ("not_found.xml", f"lookup {EVAL_UNKNOWN_ID}: no such paper", unknown),
    ]
    manifest = {
        "captured_at": _now(),
        "source": API_URL,
        "files": [
            {"file": name, "description": description, "request": request_key(params)}
            for name, description, params in files
        ]
        + [
            {
                "file": "search_no_results.xml",
                "description": f"copy of tests/fixtures/arxiv (search {NO_MATCH_KEYWORDS!r})",
                "request": None,
            }
        ],
        "search_hits": [p.arxiv_id for p in papers],
    }
    _write_json(out / "MANIFEST.json", manifest)
    for problem in problems:
        print(f"WARNING: {problem}")
    return 1 if problems else 0


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="")
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evals", action="store_true", help="record the evaluation fixtures")
    parser.add_argument("output_dir", nargs="?", type=Path)
    args = parser.parse_args()
    out = args.output_dir or (EVALS_DIR if args.evals else TESTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    try:
        return capture_evals(out) if args.evals else capture_tests(out)
    except ArxivError as exc:
        # Files written before the failure are kept; re-run once arXiv answers.
        print(f"error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
