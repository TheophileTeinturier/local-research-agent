"""Pure functions that turn LLM input into arXiv API parameters. No I/O.

This module is the injection barrier. The model never writes arXiv query syntax:
- keywords are reduced to lowercase letters and digits before being quoted into `all:` terms,
  so `AND`, `OR`, `ANDNOT`, quotes, parentheses and `field:` prefixes cannot survive;
- a category must match a strict pattern before it becomes a `cat:` clause;
- an identifier must match the new-style or old-style id grammar before it reaches `id_list`.

API facts used here (user manual, info.arxiv.org/help/api/user-manual.html, checked
2026-11-09, plus live probes the same day):
- `search_query` uses `prefix:term` with prefixes such as `all`, `ti`, `abs`, `cat`; boolean
  operators AND / OR / ANDNOT are upper case; phrases are double-quoted; parentheses group.
- `sortBy` is relevance | lastUpdatedDate | submittedDate; `sortOrder` ascending | descending.
- Probed: ANDing a stopword collapses the results (`all:graph AND all:neural` = 21,227 hits,
  `... AND all:for` = 1), hence `STOPWORDS`. Lower-case `and` is not an operator.
- Probed: `cat:cs` (archive only) matches nothing while `cat:cs.*` works, and
  `(cat:hep-th OR cat:hep-th.*)` works for archives without subject classes.
- Probed: `id_list=arXiv:1909.10086` and `id_list=math.GT/0309136` match nothing; the API wants
  `1909.10086` and `math/0309136`, hence the normalisation in `parse_arxiv_id`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .errors import InvalidArxivIdError, InvalidQueryError

Sort = Literal["relevance", "date"]

MAX_QUERY_CHARS = 200
MAX_TERMS = 8
MAX_RESULTS_LIMIT = 10

# Lucene's default English stopwords, the words that are operators in upper case, and request
# words that describe the search rather than its topic ("recent papers on X").
_STOPWORDS_TEXT = (
    "a an and are as at be but by for if in into is it no not of on or such that the their "
    "then there these they this to was will with andnot "
    "paper papers article articles preprint preprints recent latest arxiv"
)
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())

# New style (April 2007 onwards): YYMM.NNNN until 1412, YYMM.NNNNN from 1501.
# Old style (until March 2007): archive[.SUBJECT]/YYMMNNN, e.g. hep-th/9901001, math.GT/0309136.
_NEW_ID = r"(?P<yymm>\d{2}(?:0[1-9]|1[0-2]))\.(?P<number>\d{4,5})"
_OLD_ID = r"(?P<archive>[a-z]+(?:-[a-z]+)?)(?:\.[A-Z]{2})?/(?P<old_number>\d{7})"
_ID_RE = re.compile(rf"(?:{_NEW_ID}|{_OLD_ID})(?P<version>v[1-9]\d{{0,2}})?")
_ID_PREFIX_RE = re.compile(r"^arxiv:\s*", re.IGNORECASE)

# Archive, optionally "." + subject class: cs.LG, q-bio.QM, physics.optics, cond-mat.str-el,
# hep-th. Checked for format only; an unknown category simply returns no results.
CATEGORY_PATTERN = r"[a-z]{1,8}(?:-[a-z]{2,4})?(?:\.[A-Za-z]{2,10}(?:-[a-z]{2,4})?)?"
_CATEGORY_RE = re.compile(CATEGORY_PATTERN)

_TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')  # a quoted phrase, or a run of non-spaces
_WORD_RE = re.compile(r"[^\W_]+")  # letters and digits only (Unicode-aware)


@dataclass(frozen=True)
class ArxivId:
    """A validated identifier: `base` without version, `version` like "v2" or None."""

    base: str
    version: str | None = None

    def __str__(self) -> str:
        return self.base + (self.version or "")


def parse_arxiv_id(raw: str) -> ArxivId:
    """Validate and normalise an identifier, or raise `InvalidArxivIdError`.

    Accepts an optional "arXiv:" prefix. Old-style subject classes (math.GT/0309136) are
    dropped, because the API only knows math/0309136.
    """
    text = _ID_PREFIX_RE.sub("", raw.strip())
    match = _ID_RE.fullmatch(text)
    if match is None:
        raise InvalidArxivIdError(
            f"{raw[:60]!r} is not an arXiv identifier. Expected '2301.12345' (optionally with "
            "a version, '2301.12345v2') or an old-style id such as 'hep-th/9901001'. "
            "Copy the id exactly from search_arxiv results."
        )
    version = match["version"]
    if match["yymm"] is not None:
        digits = 4 if int(match["yymm"]) < 1501 else 5
        if len(match["number"]) != digits:
            raise InvalidArxivIdError(
                f"{raw[:60]!r} is not a valid arXiv identifier: ids from {match['yymm']} have "
                f"{digits} digits after the dot. Copy the id exactly from search_arxiv results."
            )
        return ArxivId(f"{match['yymm']}.{match['number']}", version)
    return ArxivId(f"{match['archive']}/{match['old_number']}", version)


def validate_category(raw: str) -> str:
    """Return the category unchanged if it has the arXiv format, else raise."""
    category = raw.strip()
    if not _CATEGORY_RE.fullmatch(category):
        raise InvalidQueryError(
            f"Invalid category {raw[:40]!r}: use an arXiv category code such as 'cs.LG', "
            "'q-bio.QM', 'stat.ML' or 'hep-th', or omit the category."
        )
    return category


@dataclass(frozen=True)
class SearchQuery:
    """A validated search, ready to be sent. Build it with `SearchQuery.build`."""

    terms: tuple[str, ...]  # lowercase words; a multi-word term is searched as a phrase
    ignored: tuple[str, ...]  # stopwords removed from the input, reported to the model
    category: str | None
    max_results: int
    sort: Sort

    @classmethod
    def build(
        cls, keywords: str, category: str | None, max_results: int, sort: Sort
    ) -> SearchQuery:
        """Validate the raw tool input. Raises `InvalidQueryError` with an actionable message."""
        keywords = keywords.strip()
        if not keywords:
            raise InvalidQueryError("The query is empty: give 2 to 6 specific keywords.")
        if len(keywords) > MAX_QUERY_CHARS:
            raise InvalidQueryError(
                f"The query is too long ({len(keywords)} characters, maximum "
                f"{MAX_QUERY_CHARS}): use 2 to 6 short keywords, not a sentence."
            )
        if not 1 <= max_results <= MAX_RESULTS_LIMIT:
            raise InvalidQueryError(f"max_results must be between 1 and {MAX_RESULTS_LIMIT}.")
        if sort not in ("relevance", "date"):
            raise InvalidQueryError("sort must be 'relevance' or 'date'.")
        terms, ignored = split_keywords(keywords)
        if not terms:
            raise InvalidQueryError(
                "The query has no usable keywords (only stopwords or punctuation): "
                "give specific topic words, e.g. 'graph neural network protein'."
            )
        if len(terms) > MAX_TERMS:
            raise InvalidQueryError(
                f"Too many keywords ({len(terms)}, maximum {MAX_TERMS}). Every keyword must "
                "match, so keep only the 2 to 6 most specific ones."
            )
        return cls(
            terms=terms,
            ignored=ignored,
            category=validate_category(category) if category else None,
            max_results=max_results,
            sort=sort,
        )

    def search_query(self) -> str:
        """The arXiv `search_query` value: every term ANDed over all fields."""
        clauses = [f'all:"{term}"' if " " in term else f"all:{term}" for term in self.terms]
        if self.category is not None:
            clauses.append(category_clause(self.category))
        return " AND ".join(clauses)

    def params(self) -> dict[str, str]:
        """Query-string parameters of the request (values as strings, as sent)."""
        return {
            "search_query": self.search_query(),
            "start": "0",
            "max_results": str(self.max_results),
            "sortBy": "relevance" if self.sort == "relevance" else "submittedDate",
            "sortOrder": "descending",
        }

    def describe(self) -> str:
        """Human/LLM-readable form of what is searched, e.g. 'graph AND "protein protein"'."""
        text = " AND ".join(f'"{t}"' if " " in t else t for t in self.terms)
        return text + (f" in {self.category}" if self.category else "")


def split_keywords(keywords: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Plain keywords -> (terms, ignored stopwords).

    A quoted phrase stays one term. A token with inner punctuation ("protein-protein") becomes a
    phrase of its words, which is how the index tokenises it. Everything is lowercased, so
    nothing can be read as an operator. Duplicates are dropped, order is kept.
    """
    terms: list[str] = []
    ignored: list[str] = []
    for match in _TOKEN_RE.finditer(keywords):
        words = _WORD_RE.findall((match[1] if match[1] is not None else match[2]).lower())
        kept = [word for word in words if word not in STOPWORDS]
        for word in words:
            if word in STOPWORDS and word not in ignored:
                ignored.append(word)
        term = " ".join(kept)
        if term and term not in terms:
            terms.append(term)
    return tuple(terms), tuple(ignored)


def category_clause(category: str) -> str:
    """`cat:cs.LG`; an archive alone also matches its subject classes (`cat:cs` alone does not)."""
    if "." in category:
        return f"cat:{category}"
    return f"(cat:{category} OR cat:{category}.*)"


def lookup_params(arxiv_id: ArxivId) -> dict[str, str]:
    """Parameters to fetch one paper by id (`id_list`, never `search_query`)."""
    return {"id_list": str(arxiv_id), "max_results": "1"}
