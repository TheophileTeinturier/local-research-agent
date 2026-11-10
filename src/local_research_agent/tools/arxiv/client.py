"""`ArxivClient`: the arXiv API over HTTP, with a rate limiter and a small TTL cache.

Usage rules (Terms of Use, info.arxiv.org/help/api/tou.html, checked 2026-11-09):
"make no more than one request every three seconds, and limit requests to a single connection
at a time", applying "to all of the machines under your control as a whole". The user manual
also asks clients to cache results ("no need to call the API more than once in a day for the
same query"). Neither page asks anything of the User-Agent, so none is set: httpx sends its
default one ("python-httpx/<version>").

How this client complies:
- `RateLimiter`: request starts are spaced by at least 3 s (injectable clock and sleep, so the
  tests never sleep). A 429 (or a 5xx carrying Retry-After) also blocks requests for that
  long: during that window a call fails at once, without contacting arXiv.
- One lock around "wait for the limiter, then send": at most one request in flight, even when
  the agent abandons a timed-out tool thread that is still waiting for its response.
- `httpx.Limits(max_connections=1)` on the client this class creates.
- `TTLCache`: an identical request (same parameters) is answered from memory for an hour.

Errors are typed (`errors.py`) and never raised as httpx exceptions. Turning them into messages
for the LLM is the tool layer's job.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlencode

import httpx

from .errors import (
    ArxivHTTPError,
    ArxivRateLimited,
    ArxivTimeout,
    ArxivUnavailable,
)
from .models import PaperDetails, PaperSummary
from .parsing import Feed, parse_feed
from .query import SearchQuery, Sort, lookup_params, parse_arxiv_id

API_URL = "https://export.arxiv.org/api/query"

MIN_REQUEST_INTERVAL_S = 3.0  # Terms of Use: one request every three seconds
DEFAULT_CACHE_TTL_S = 3600.0
DEFAULT_CACHE_ENTRIES = 128
# Below the agent's default 10 s tool timeout, so a slow API normally surfaces as a clean
# "timed out" observation from this client rather than an abandoned thread.
DEFAULT_TIMEOUT = httpx.Timeout(7.0, connect=3.0)
DEFAULT_BACKOFF_S = 10.0  # 429/503 without a usable Retry-After header
MAX_BACKOFF_S = 300.0

Clock = Callable[[], float]
Sleep = Callable[[float], None]


def request_key(params: Mapping[str, Any]) -> str:
    """Canonical form of a request's parameters: the cache key and the fixture lookup key."""
    return urlencode(sorted((str(k), str(v)) for k, v in params.items()))


class RateLimiter:
    """Spaces request starts by `min_interval_s` and honours server back-off requests.

    Not thread-safe on its own: `ArxivClient` calls it under its request lock.
    """

    def __init__(self, min_interval_s: float, *, clock: Clock, sleep: Sleep) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = -math.inf
        self._blocked_until = -math.inf

    def acquire(self) -> None:
        """Return when a request may start (sleeping if needed). Raises `ArxivRateLimited`
        without sleeping if arXiv asked us to back off and the back-off has not expired."""
        now = self._clock()
        if now < self._blocked_until:
            remaining = self._blocked_until - now
            raise ArxivRateLimited(
                f"arXiv asked clients to slow down; no request sent for {remaining:.0f} s",
                retry_after_s=remaining,
            )
        wait = self._next_allowed - now
        if wait > 0:
            self._sleep(wait)
            now = self._clock()
        self._next_allowed = now + self.min_interval_s

    def back_off(self, seconds: float) -> None:
        """Refuse every request for `seconds` (capped at MAX_BACKOFF_S)."""
        until = self._clock() + min(max(seconds, 0.0), MAX_BACKOFF_S)
        self._blocked_until = max(self._blocked_until, until)


class TTLCache:
    """Small in-memory cache: entries expire after `ttl_s`; the oldest is evicted when full."""

    def __init__(self, ttl_s: float, *, clock: Clock, max_entries: int = DEFAULT_CACHE_ENTRIES):
        self.ttl_s = ttl_s
        self._clock = clock
        self._max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, Feed]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Feed | None:
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return None
            expires_at, value = item
            if self._clock() >= expires_at:
                del self._entries[key]
                return None
            return value

    def set(self, key: str, value: Feed) -> None:
        with self._lock:
            self._entries[key] = (self._clock() + self.ttl_s, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


class ArxivClient:
    """Search and look up arXiv papers. Every public method raises only `ArxivError`s."""

    def __init__(
        self,
        *,
        http: httpx.Client | None = None,
        base_url: str = API_URL,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
        min_interval_s: float = MIN_REQUEST_INTERVAL_S,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
    ) -> None:
        # Without an injected client, one is created on the first request: building a
        # registry (as every test does) opens no connection.
        self._http = http
        self._owns_http = http is None
        self._base_url = base_url
        self._limiter = RateLimiter(min_interval_s, clock=clock, sleep=sleep)
        self._cache = TTLCache(cache_ttl_s, clock=clock)
        self._request_lock = threading.Lock()

    # --- public API -------------------------------------------------------------------------

    def search(
        self,
        keywords: str,
        category: str | None = None,
        max_results: int = 5,
        sort: Sort = "relevance",
    ) -> list[PaperSummary]:
        """Papers matching every keyword. Raises `InvalidQueryError` before any request."""
        return self.run_search(SearchQuery.build(keywords, category, max_results, sort))

    def run_search(self, query: SearchQuery) -> list[PaperSummary]:
        """Run an already validated `SearchQuery`."""
        feed = self.fetch_feed(query.params())
        return [paper.to_summary() for paper in feed.papers]

    def get(self, arxiv_id: str) -> PaperDetails | None:
        """One paper, or None if arXiv has no paper with this id.

        The id is validated (`InvalidArxivIdError`) before any request is sent.
        """
        parsed = parse_arxiv_id(arxiv_id)
        feed = self.fetch_feed(lookup_params(parsed))
        # Defensive: only accept the paper that was asked for.
        return next((p for p in feed.papers if p.arxiv_id == parsed.base), None)

    # --- transport ----------------------------------------------------------------------------

    def fetch_feed(self, params: Mapping[str, str]) -> Feed:
        """Parsed response, from the cache when the identical request was made recently.

        Only successfully parsed responses are cached; errors are never cached.
        """
        key = request_key(params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        feed = parse_feed(self.fetch_text(params))
        self._cache.set(key, feed)
        return feed

    def fetch_text(self, params: Mapping[str, str]) -> str:
        """One rate-limited GET (no cache). Raises a typed `ArxivError` on any failure."""
        with self._request_lock:
            self._limiter.acquire()
            try:
                response = self._client().get(self._base_url, params=dict(params))
            except httpx.TimeoutException as exc:
                raise ArxivTimeout(f"arXiv did not answer in time ({exc!r})") from exc
            except httpx.HTTPError as exc:
                raise ArxivUnavailable(f"cannot reach arXiv ({exc!r})") from exc
            self._check_status(response)
            return response.text

    def _check_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status == 200:
            return
        retry_header = response.headers.get("Retry-After")
        if status == 429:
            retry_after = _retry_after_s(retry_header)
            self._limiter.back_off(retry_after)
            raise ArxivRateLimited(
                f"arXiv returned HTTP 429 (retry after {retry_after:.0f} s)",
                retry_after_s=retry_after,
            )
        if status >= 500:
            # A 503 with Retry-After is an explicit "come back later": honour it. Without the
            # header, the normal 3 s spacing applies and one retry is reasonable.
            if retry_header is not None:
                self._limiter.back_off(_retry_after_s(retry_header))
            raise ArxivUnavailable(f"arXiv returned HTTP {status}", status_code=status)
        raise ArxivHTTPError(
            f"arXiv rejected the request: HTTP {status} {response.text[:200]!r}",
            status_code=status,
        )

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                timeout=DEFAULT_TIMEOUT, limits=httpx.Limits(max_connections=1)
            )
        return self._http

    def close(self) -> None:
        """Close the HTTP client if this object created it (an injected one is left open)."""
        if self._owns_http and self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _retry_after_s(header: str | None) -> float:
    """Seconds from a Retry-After header (delta-seconds form only), else the default."""
    try:
        seconds = float(header) if header is not None else DEFAULT_BACKOFF_S
    except ValueError:
        return DEFAULT_BACKOFF_S  # an HTTP-date: not worth parsing for a back-off hint
    return seconds if math.isfinite(seconds) and seconds >= 0 else DEFAULT_BACKOFF_S
