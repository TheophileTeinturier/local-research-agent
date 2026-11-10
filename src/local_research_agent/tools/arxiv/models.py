"""Paper models, shaped for the LLM: every text field is capped when the model is built.

The caps live in the field types (`Annotated[..., AfterValidator]`), so no code path can
build an uncapped `PaperSummary` or `PaperDetails`, whoever constructs it.
"""

from __future__ import annotations

import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

TITLE_MAX_CHARS = 200
SNIPPET_MAX_CHARS = 200
ABSTRACT_MAX_CHARS = 1500
AUTHOR_MAX_CHARS = 80
MAX_AUTHORS = 10

ELLIPSIS = "..."


def truncate_words(text: str, limit: int) -> str:
    """Collapse whitespace, then cut to at most `limit` chars on a word boundary + "...".

    A single word longer than the limit is cut mid-word (there is no boundary to use).
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    budget = limit - len(ELLIPSIS)
    # One extra character: if it is a space, the word before it is complete and is kept.
    head = text[: budget + 1]
    boundary = head.rfind(" ")
    head = head[:boundary] if boundary > 0 else head[:budget]
    return head.rstrip(" ,;:.-") + ELLIPSIS


def _capped(limit: int) -> AfterValidator:
    return AfterValidator(lambda text: truncate_words(text, limit))


Title = Annotated[str, _capped(TITLE_MAX_CHARS)]
AuthorName = Annotated[str, _capped(AUTHOR_MAX_CHARS)]


class PaperSummary(BaseModel):
    """One search hit: just enough for the model to decide whether to open the paper."""

    model_config = ConfigDict(frozen=True)

    arxiv_id: str  # without version, e.g. "2301.12345" or "hep-th/9901001"
    title: Title
    first_author: AuthorName
    et_al: bool  # True when the paper has more than one author
    year: int | None
    primary_category: str | None
    snippet: Annotated[str, _capped(SNIPPET_MAX_CHARS)]  # start of the abstract


class PaperDetails(BaseModel):
    """The full (capped) record of one paper, returned by `get_paper_details`."""

    model_config = ConfigDict(frozen=True)

    arxiv_id: str  # without version
    version: str | None  # e.g. "v3"; the API returns the latest unless one was requested
    title: Title
    authors: list[AuthorName] = Field(max_length=MAX_AUTHORS)  # the first MAX_AUTHORS
    authors_total: int  # real number of authors, so the output can say "+N more"
    published: datetime.date | None  # first version
    updated: datetime.date | None  # this version
    primary_category: str | None
    categories: list[str]  # primary first
    abstract: Annotated[str, _capped(ABSTRACT_MAX_CHARS)]
    pdf_url: str

    def to_summary(self) -> PaperSummary:
        """The search-result view of this paper (the snippet cap applies here)."""
        return PaperSummary(
            arxiv_id=self.arxiv_id,
            title=self.title,
            first_author=self.authors[0] if self.authors else "Unknown author",
            et_al=self.authors_total > 1,
            year=self.published.year if self.published else None,
            primary_category=self.primary_category,
            snippet=self.abstract,
        )
