"""ArxivClient with httpx.MockTransport, a FakeClock and a FakeSleep: no network, no sleeping."""

from collections.abc import Callable

import httpx
import pytest

from fakes import ARXIV_FIXTURES, FakeClock, FakeSleep
from local_research_agent.tools.arxiv.client import (
    API_URL,
    DEFAULT_BACKOFF_S,
    MIN_REQUEST_INTERVAL_S,
    ArxivClient,
    RateLimiter,
    TTLCache,
    request_key,
)
from local_research_agent.tools.arxiv.errors import (
    ArxivHTTPError,
    ArxivMalformedResponse,
    ArxivRateLimited,
    ArxivTimeout,
    ArxivUnavailable,
    InvalidArxivIdError,
    InvalidQueryError,
)
from local_research_agent.tools.arxiv.parsing import Feed

SEARCH_XML = (ARXIV_FIXTURES / "search_gnn_ppi.xml").read_bytes()
DETAILS_XML = (ARXIV_FIXTURES / "paper_details.xml").read_bytes()
EMPTY_XML = (ARXIV_FIXTURES / "unknown_id.xml").read_bytes()

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """MockTransport handler: answers with `respond(request)` and records the requests."""

    def __init__(self, respond: Handler) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)


def make_client(
    respond: Handler, clock: FakeClock | None = None, **kwargs: object
) -> tuple[ArxivClient, Recorder, FakeSleep]:
    clock = clock or FakeClock()
    sleep = FakeSleep(clock)
    recorder = Recorder(respond)
    client = ArxivClient(
        http=httpx.Client(transport=httpx.MockTransport(recorder)),
        clock=clock,
        sleep=sleep,
        **kwargs,  # type: ignore[arg-type]
    )
    return client, recorder, sleep


def ok(body: bytes) -> Handler:
    return lambda request: httpx.Response(200, content=body)


# --- requests ------------------------------------------------------------------------------


def test_search_sends_the_built_query() -> None:
    client, recorder, _ = make_client(ok(SEARCH_XML))

    papers = client.search("graph neural networks", category="cs.LG", max_results=5, sort="date")

    assert len(papers) == 5
    [request] = recorder.requests
    assert str(request.url).startswith(API_URL)
    assert dict(request.url.params) == {
        "search_query": "all:graph AND all:neural AND all:networks AND cat:cs.LG",
        "start": "0",
        "max_results": "5",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    # No custom User-Agent: httpx's default one is sent.
    assert request.headers["User-Agent"].startswith("python-httpx/")


def test_get_returns_the_paper() -> None:
    client, recorder, _ = make_client(ok(DETAILS_XML))
    paper = client.get("arXiv:2606.01781")
    assert paper is not None and paper.arxiv_id == "2606.01781"
    assert dict(recorder.requests[0].url.params) == {"id_list": "2606.01781", "max_results": "1"}


def test_get_unknown_id_returns_none() -> None:
    client, _, _ = make_client(ok(EMPTY_XML))
    assert client.get("2301.99999") is None


def test_get_ignores_a_different_paper_than_requested() -> None:
    client, _, _ = make_client(ok(DETAILS_XML))  # contains 2606.01781
    assert client.get("2301.12345") is None


def test_invalid_input_is_rejected_before_any_request() -> None:
    client, recorder, _ = make_client(ok(SEARCH_XML))
    with pytest.raises(InvalidArxivIdError):
        client.get("1234.5678 OR all:x")
    with pytest.raises(InvalidQueryError):
        client.search("graph", category="cs.LG OR all:x")
    with pytest.raises(InvalidQueryError):
        client.search("the and of")
    assert recorder.requests == []


def test_no_connection_is_opened_until_the_first_request() -> None:
    client = ArxivClient()
    assert client._http is None  # building a registry costs nothing
    client.close()  # closing an unused client is fine


def test_request_key_is_order_independent() -> None:
    assert request_key({"b": "2", "a": "x y"}) == request_key({"a": "x y", "b": 2})
    assert request_key({"a": "x y"}) == "a=x+y"


# --- rate limiting -------------------------------------------------------------------------


def test_rate_limiter_spaces_requests_without_real_sleep() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    limiter = RateLimiter(3.0, clock=clock, sleep=sleep)

    limiter.acquire()  # first request: immediate
    clock.advance(1.0)
    limiter.acquire()  # 1 s later: waits the remaining 2 s
    limiter.acquire()  # right after: waits a full 3 s
    clock.advance(10.0)
    limiter.acquire()  # long after: immediate

    assert sleep.calls == [pytest.approx(2.0), pytest.approx(3.0)]


def test_client_waits_between_consecutive_requests() -> None:
    client, recorder, sleep = make_client(ok(EMPTY_XML))
    client.get("2301.99999")
    client.get("2301.99998")
    client.get("2301.99997")
    assert len(recorder.requests) == 3
    assert sleep.calls == [pytest.approx(MIN_REQUEST_INTERVAL_S)] * 2


def test_back_off_refuses_requests_until_it_expires() -> None:
    clock = FakeClock()
    limiter = RateLimiter(3.0, clock=clock, sleep=FakeSleep(clock))
    limiter.back_off(30)
    clock.advance(10)
    with pytest.raises(ArxivRateLimited) as excinfo:
        limiter.acquire()
    assert excinfo.value.retry_after_s == pytest.approx(20)
    clock.advance(20)
    limiter.acquire()  # expired: allowed again


# --- cache ---------------------------------------------------------------------------------


def test_identical_request_is_served_from_cache() -> None:
    client, recorder, sleep = make_client(ok(SEARCH_XML))
    first = client.search("graph neural networks")
    second = client.search("  Graph neural NETWORKS ")  # same normalized request
    assert first == second
    assert len(recorder.requests) == 1
    assert sleep.calls == []  # a cache hit does not even wait for the rate limiter


def test_different_request_is_not_served_from_cache() -> None:
    client, recorder, _ = make_client(ok(SEARCH_XML))
    client.search("graph neural networks", max_results=5)
    client.search("graph neural networks", max_results=3)
    assert len(recorder.requests) == 2


def test_cache_entries_expire() -> None:
    clock = FakeClock()
    client, recorder, _ = make_client(ok(EMPTY_XML), clock=clock, cache_ttl_s=60)
    client.get("2301.99999")
    clock.advance(59)
    client.get("2301.99999")
    clock.advance(2)
    client.get("2301.99999")
    assert len(recorder.requests) == 2


def test_cache_evicts_oldest_entry_when_full() -> None:
    clock = FakeClock()
    cache = TTLCache(60, clock=clock, max_entries=2)
    empty = Feed(total_results=0, papers=[])
    cache.set("a", empty)
    cache.set("b", empty)
    cache.set("c", empty)
    assert cache.get("a") is None
    assert cache.get("b") is empty and cache.get("c") is empty


def test_failures_are_not_cached() -> None:
    responses = iter([httpx.Response(503), httpx.Response(200, content=EMPTY_XML)])
    client, recorder, _ = make_client(lambda request: next(responses))
    with pytest.raises(ArxivUnavailable):
        client.get("2301.99999")
    assert client.get("2301.99999") is None  # retried for real, not a cached failure
    assert len(recorder.requests) == 2


# --- errors are typed ------------------------------------------------------------------------


def raise_(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


@pytest.mark.parametrize(
    ("respond", "error"),
    [
        (raise_(httpx.ReadTimeout("slow")), ArxivTimeout),
        (raise_(httpx.ConnectTimeout("slow")), ArxivTimeout),
        (raise_(httpx.ConnectError("dns")), ArxivUnavailable),
        (lambda r: httpx.Response(503), ArxivUnavailable),
        (lambda r: httpx.Response(500), ArxivUnavailable),
        (lambda r: httpx.Response(429), ArxivRateLimited),
        (lambda r: httpx.Response(400, text="bad"), ArxivHTTPError),
        (lambda r: httpx.Response(200, text="<html>oops</html>"), ArxivMalformedResponse),
    ],
)
def test_failures_become_typed_errors(respond: Handler, error: type[Exception]) -> None:
    client, _, _ = make_client(respond)
    with pytest.raises(error):
        client.search("graph")


def test_429_blocks_the_next_request_without_contacting_arxiv() -> None:
    clock = FakeClock()
    client, recorder, _ = make_client(
        lambda r: httpx.Response(429, headers={"Retry-After": "20"}), clock=clock
    )
    with pytest.raises(ArxivRateLimited) as first:
        client.search("graph")
    assert first.value.retry_after_s == 20
    clock.advance(5)
    with pytest.raises(ArxivRateLimited):
        client.search("graph")
    assert len(recorder.requests) == 1  # the second call never reached arXiv


def test_429_without_retry_after_uses_the_default_back_off() -> None:
    client, _, _ = make_client(lambda r: httpx.Response(429))
    with pytest.raises(ArxivRateLimited) as excinfo:
        client.search("graph")
    assert excinfo.value.retry_after_s == DEFAULT_BACKOFF_S


def test_503_with_retry_after_backs_off_and_without_does_not() -> None:
    clock = FakeClock()
    client, recorder, _ = make_client(
        lambda r: httpx.Response(503, headers={"Retry-After": "30"}), clock=clock
    )
    with pytest.raises(ArxivUnavailable):
        client.search("graph")
    with pytest.raises(ArxivRateLimited):
        client.search("graph")
    assert len(recorder.requests) == 1

    client, recorder, _ = make_client(lambda r: httpx.Response(503))
    for _ in range(2):
        with pytest.raises(ArxivUnavailable):
            client.search("graph")
    assert len(recorder.requests) == 2
