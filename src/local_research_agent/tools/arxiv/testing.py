"""Offline replay of recorded arXiv responses, for tests and the evaluation harness.

    transport = FakeArxivTransport.from_directory("tests/fixtures/arxiv")
    client = offline_arxiv_client(transport)
    registry = build_default_registry(arxiv_client=client)

Fixtures are recorded by `scripts/capture_arxiv_fixtures.py`. `index.json` maps the canonical
request key (`client.request_key`, the same key as the cache) to an XML file. A request
that was never recorded gets an empty feed (what arXiv answers for "no match") and is
logged in `unmatched`, so a harness can tell "no results" apart from "not recorded".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import httpx

from .client import ArxivClient, request_key

EMPTY_FEED = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<feed xmlns="http://www.w3.org/2005/Atom" '
    'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">\n'
    "  <opensearch:totalResults>0</opensearch:totalResults>\n"
    "</feed>\n"
)


class FakeArxivTransport(httpx.BaseTransport):
    """httpx transport answering arXiv API requests from recorded bodies. No network."""

    def __init__(self, responses: Mapping[str, str | bytes]) -> None:
        self._responses = dict(responses)
        self.requests: list[httpx.Request] = []
        self.unmatched: list[str] = []

    @classmethod
    def from_directory(cls, directory: str | Path) -> FakeArxivTransport:
        """Load `index.json` ({"fixtures": [{"key", "file", ...}]}) and the XML files."""
        root = Path(directory)
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        return cls({item["key"]: (root / item["file"]).read_bytes() for item in index["fixtures"]})

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = request_key(dict(request.url.params))
        body = self._responses.get(key)
        if body is None:
            self.unmatched.append(key)
            body = EMPTY_FEED
        return httpx.Response(
            200, content=body, headers={"Content-Type": "application/atom+xml; charset=utf-8"}
        )


def offline_arxiv_client(transport: httpx.BaseTransport) -> ArxivClient:
    """An `ArxivClient` on `transport` that never sleeps (no real API, so no rate limit)."""
    return ArxivClient(http=httpx.Client(transport=transport), sleep=lambda _seconds: None)
