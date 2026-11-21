"""The fake arXiv of one case: an httpx transport serving recorded feeds by RULE, not by key.

`tools/arxiv/testing.py::FakeArxivTransport` replays a response only for the exact recorded
request. An evaluated model phrases its query freely, so this transport serves by kind:

- search (`search_query` param): the case's `search` feed, whatever the query text and sort,
  cut to the requested `max_results` entries. No `search` fixture: with
  `default_search: "empty"`, a zero-result Atom feed (a normal answer, outcome
  "default_empty"); otherwise a FIXTURE MISS (an empty feed is served too, but the request is
  recorded in `misses` and the run is inconclusive).
- lookup (`id_list` param, version ignored): the case's `papers[id]` file; else the entry
  with that id in the case's `search` feed (the search response already holds the full
  record, so a "search -> details" run stays consistent); else arXiv's real "not found"
  response (an empty feed). An unknown id is a normal answer, never a miss.
- anything else: a miss.
- `failure_mode` "http_503": every request gets HTTP 503 (no Retry-After). "timeout": every
  request raises `httpx.ReadTimeout` at once (no real waiting). Never a miss.

The client turns these into the same typed errors and observations as in production.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from local_research_agent.tools.arxiv.testing import EMPTY_FEED

from .suite import FIXTURES_DIR, NOT_FOUND_FILE, ArxivSpec

_ENTRY_RE = re.compile(r"<entry>.*?</entry>\s*", re.DOTALL)
_ENTRY_ID_RE = re.compile(r"<id>\s*https?://arxiv\.org/abs/(?P<id>[^<\s]+?)(?:v\d+)?\s*</id>")
_TOTAL_RE = re.compile(r"<opensearch:totalResults>\d+</opensearch:totalResults>")
_VERSION_RE = re.compile(r"v\d+$")

_HEADERS = {"Content-Type": "application/atom+xml; charset=utf-8"}


@dataclass(frozen=True)
class ServedRequest:
    """How one request was answered: kind (search/lookup/other) and outcome."""

    kind: str
    key: str  # the query string as sent
    # "fixture" | "search_hit" | "not_found" | "default_empty" | "miss" | "http_503" | "timeout"
    outcome: str


class CaseArxivTransport(httpx.BaseTransport):
    """Serves one case's arXiv fixtures (see the module docstring for the rules)."""

    def __init__(self, spec: ArxivSpec, fixtures_dir: Path = FIXTURES_DIR) -> None:
        self._spec = spec
        self._search = _read(fixtures_dir, spec.search) if spec.search else None
        self._papers = {_base_id(k): _read(fixtures_dir, v) for k, v in spec.papers.items()}
        self._not_found = _read(fixtures_dir, NOT_FOUND_FILE)
        self.served: list[ServedRequest] = []

    @property
    def misses(self) -> list[str]:
        """Query strings of the requests no fixture covered."""
        return [r.key for r in self.served if r.outcome == "miss"]

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Answer one arXiv API request from the case's fixtures and record how it was served."""
        params = dict(request.url.params)
        key = request.url.query.decode("ascii", errors="replace")
        kind = (
            "lookup" if "id_list" in params else "search" if "search_query" in params else "other"
        )

        failure = self._spec.failure_mode
        if failure is not None:
            self.served.append(ServedRequest(kind, key, failure))
            if failure == "timeout":
                raise httpx.ReadTimeout("simulated arXiv timeout", request=request)
            return httpx.Response(503, text="Service Unavailable", request=request)

        if kind == "search" and self._search is not None:
            body, outcome = truncate_feed(self._search, _int(params.get("max_results"))), "fixture"
        elif kind == "search" and self._spec.default_search == "empty":
            body, outcome = EMPTY_FEED, "default_empty"
        elif kind == "lookup":
            body, outcome = self._lookup(_base_id(params["id_list"]))
        else:
            body, outcome = self._not_found, "miss"
        self.served.append(ServedRequest(kind, key, outcome))
        return httpx.Response(200, text=body, headers=_HEADERS, request=request)

    def _lookup(self, arxiv_id: str) -> tuple[str, str]:
        if arxiv_id in self._papers:
            return self._papers[arxiv_id], "fixture"
        if self._search is not None:
            feed = single_entry_feed(self._search, arxiv_id)
            if feed is not None:
                return feed, "search_hit"
        return self._not_found, "not_found"


def truncate_feed(feed: str, max_results: int | None) -> str:
    """The feed with only its first `max_results` entries (all of them if None)."""
    if max_results is None:
        return feed
    kept = 0

    def keep_first(match: re.Match[str]) -> str:
        nonlocal kept
        kept += 1
        return match.group(0) if kept <= max_results else ""

    return _ENTRY_RE.sub(keep_first, feed)


def single_entry_feed(feed: str, arxiv_id: str) -> str | None:
    """A lookup-style feed holding only the entry `arxiv_id` of `feed`, or None if absent."""
    for match in _ENTRY_RE.finditer(feed):
        id_match = _ENTRY_ID_RE.search(match.group(0))
        if id_match is not None and id_match["id"] == arxiv_id:
            first = _ENTRY_RE.search(feed)
            head = feed[: first.start()] if first else feed
            head = _TOTAL_RE.sub("<opensearch:totalResults>1</opensearch:totalResults>", head)
            return f"{head}{match.group(0)}</feed>\n"
    return None


def _read(directory: Path, name: str) -> str:
    return (directory / name).read_text(encoding="utf-8")


def _base_id(raw: str) -> str:
    return _VERSION_RE.sub("", raw.strip())


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
