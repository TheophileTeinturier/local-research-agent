"""Typed failures of the arXiv layer.

The client and the parser raise these; only the tool layer (`tools.py`) turns them into
messages for the LLM. Input errors (`InvalidQueryError`, `InvalidArxivIdError`) are raised
before any request is sent.
"""

from __future__ import annotations


class ArxivError(Exception):
    """Base class of every failure the arXiv tools report to the model."""


class InvalidQueryError(ArxivError, ValueError):
    """The keywords, category or limits cannot form a valid search (no request was sent)."""


class InvalidArxivIdError(ArxivError, ValueError):
    """The string is not an arXiv identifier (no request was sent)."""


class ArxivTimeout(ArxivError):
    """The API did not answer within the client's timeout."""


class ArxivUnavailable(ArxivError):
    """Network failure or HTTP 5xx: transient, retrying later may work."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ArxivRateLimited(ArxivError):
    """HTTP 429, or a request refused locally because arXiv asked us to back off."""

    def __init__(self, message: str, *, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ArxivHTTPError(ArxivError):
    """Any other non-200 status (4xx): the request itself was rejected."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ArxivMalformedResponse(ArxivError):
    """The body is not a parseable Atom feed, or the feed reports an API error."""
