"""Atom feed (arXiv API response) -> `PaperDetails`. Pure: text in, models out.

Why defusedxml rather than `xml.etree` alone: the body comes from the network, and the Python
docs advise against parsing untrusted XML with the standard library (entity-expansion attacks
such as "billion laughs"; recent expat versions mitigate them, older builds do not).
`defusedxml.ElementTree.fromstring(..., forbid_dtd=True)` refuses any DTD outright. An Atom feed
never has one, so this costs nothing and removes the whole class of attacks. It returns plain
`xml.etree` elements, so the rest of the code is ordinary standard-library XML.

Structure (user manual section 3.3, checked against live responses on 2026-11-09): a `feed`
with `opensearch:totalResults` and one `entry` per paper. In each entry, `id` is
"http://arxiv.org/abs/<id>v<N>", plus title, summary, published, updated, author/name,
category@term, arxiv:primary_category@term and link[@title="pdf"]@href. Missing fields get a
placeholder; an entry without a usable id is skipped (there is nothing to cite).
The manual says errors come back as an entry whose id is "http://arxiv.org/api/errors#...". The
live API answered HTTP 200 with an empty feed for malformed ids instead, but both are handled.
"""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from xml.etree.ElementTree import Element, ParseError

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import fromstring

from ...observability import log_event
from .errors import ArxivMalformedResponse
from .models import MAX_AUTHORS, PaperDetails

logger = logging.getLogger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

_ABS_ID_RE = re.compile(r"arxiv\.org/abs/(?P<base>.+?)(?P<version>v\d+)?$")
_ERROR_ID_MARKER = "arxiv.org/api/errors"


@dataclass(frozen=True)
class Feed:
    """A parsed response. `total_results` is how many papers match, not how many were sent."""

    total_results: int | None
    papers: list[PaperDetails]


def parse_feed(xml_text: str | bytes) -> Feed:
    """Parse an API response body. Raises `ArxivMalformedResponse`, never anything else."""
    try:
        root = fromstring(xml_text, forbid_dtd=True)
    except (ParseError, DefusedXmlException, ValueError) as exc:
        raise ArxivMalformedResponse(f"arXiv returned invalid XML: {exc}") from exc
    if root.tag != f"{ATOM}feed":
        raise ArxivMalformedResponse(f"arXiv returned an unexpected document: <{root.tag}>")

    papers: list[PaperDetails] = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = _text(entry, f"{ATOM}id") or ""
        if _ERROR_ID_MARKER in entry_id:
            message = _text(entry, f"{ATOM}summary") or "unknown error"
            raise ArxivMalformedResponse(f"arXiv reported an error: {message}")
        paper = _parse_entry(entry, entry_id)
        if paper is None:
            log_event(logger, "arxiv_entry_skipped", level=logging.WARNING, entry_id=entry_id)
        else:
            papers.append(paper)
    return Feed(total_results=_int(_text(root, f"{OPENSEARCH}totalResults")), papers=papers)


def _parse_entry(entry: Element, entry_id: str) -> PaperDetails | None:
    match = _ABS_ID_RE.search(entry_id)
    if match is None:
        return None
    base, version = match["base"], match["version"]

    authors = [
        name for author in entry.findall(f"{ATOM}author") if (name := _text(author, f"{ATOM}name"))
    ]
    # dict.fromkeys: drop duplicates, keep order.
    categories = list(
        dict.fromkeys(
            term
            for c in entry.findall(f"{ATOM}category")
            if (term := (c.get("term") or "").strip())
        )
    )
    primary_element = entry.find(f"{ARXIV}primary_category")
    primary = (primary_element.get("term") if primary_element is not None else None) or (
        categories[0] if categories else None
    )
    if primary is not None:
        categories = [primary, *(c for c in categories if c != primary)]

    pdf_url = (
        next(
            (
                link.get("href")
                for link in entry.findall(f"{ATOM}link")
                if link.get("title") == "pdf" and link.get("href")
            ),
            None,
        )
        or f"https://arxiv.org/pdf/{base}{version or ''}"
    )

    return PaperDetails(
        arxiv_id=base,
        version=version,
        title=_text(entry, f"{ATOM}title") or "(untitled)",
        authors=authors[:MAX_AUTHORS],
        authors_total=len(authors),
        published=_date(_text(entry, f"{ATOM}published")),
        updated=_date(_text(entry, f"{ATOM}updated")),
        primary_category=primary,
        categories=categories,
        abstract=_text(entry, f"{ATOM}summary") or "(no abstract)",
        pdf_url=pdf_url,
    )


def _text(element: Element, tag: str) -> str | None:
    """Text of a child element with whitespace collapsed (titles contain line breaks)."""
    value = element.findtext(tag)
    if value is None:
        return None
    return " ".join(value.split()) or None


def _date(value: str | None) -> datetime.date | None:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
