"""The two arXiv tools: `search_arxiv` (compact summaries) and `get_paper_details` (drill-down).

Each tool is defined once: a Pydantic args model, a function taking (args, client), a name
and a description. `make_arxiv_tools(client)` binds the functions to a client with
`functools.partial`, so the functions themselves stay plain and testable.

Every `ArxivError` is converted here into a `ToolError` whose message tells the model what to
do next. The registry turns it into `ToolResult(ok=False, ...)`, with `error_type` in the
metadata.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..base import Tool, ToolError
from .client import ArxivClient
from .errors import (
    ArxivError,
    ArxivHTTPError,
    ArxivMalformedResponse,
    ArxivRateLimited,
    ArxivTimeout,
    ArxivUnavailable,
    InvalidArxivIdError,
    InvalidQueryError,
)
from .models import PaperDetails, PaperSummary
from .query import (
    MAX_QUERY_CHARS,
    MAX_RESULTS_LIMIT,
    SearchQuery,
    parse_arxiv_id,
    validate_category,
)

# Well below the registry's 4000-char cap, so the registry never has to cut a result
# mid-paper. 10 papers with typical titles fit; an extreme case drops the last papers and
# says so (test_search_output_respects_budget_in_the_worst_case).
SEARCH_OUTPUT_MAX_CHARS = 3500

SEARCH_DESCRIPTION = (
    "Search arXiv for scientific papers and return SUMMARIES only: for each paper its arXiv "
    "id, year, primary category, title, first author and a short abstract snippet. Use it "
    "when the user asks for papers, preprints or research on a topic. "
    "`query` is plain keywords, not a sentence and not arXiv query syntax: 2 to 6 short, "
    "specific terms. Every keyword must appear somewhere in the paper, so keep the user's "
    "precise terms and put multi-word concepts in double quotes, which match them as an "
    'exact phrase, e.g. \'"graph neural network" "protein-protein interaction"\'. '
    "Use sort='date' when the user wants recent, latest or new papers, otherwise "
    "'relevance'. Omit `category` unless the user names a field: a category filter hides "
    "papers filed elsewhere. To read the full abstract, all authors and the PDF link of a "
    "promising paper, call get_paper_details with its id."
)

DETAILS_DESCRIPTION = (
    "Get the full record of ONE arXiv paper: title, authors, dates, categories, the full "
    "abstract and the PDF URL. Call it after search_arxiv, only for the most relevant "
    "papers, with an id copied exactly from the search results (e.g. '2301.12345' or "
    "'hep-th/9901001'). Never guess or construct an id."
)


# --- search_arxiv ---------------------------------------------------------------------------


class SearchArxivArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description='Keywords; quote multi-word concepts, e.g. \'"graph neural network" '
        '"protein-protein interaction"\'.',
    )
    category: str | None = Field(
        default=None,
        description="Only if the user names a field: an arXiv category such as 'cs.LG', "
        "'q-bio.QM' or 'hep-th'. Default: no filter.",
    )
    max_results: int = Field(
        default=5, ge=1, le=MAX_RESULTS_LIMIT, description="Number of papers (1 to 10)."
    )
    sort: Literal["relevance", "date"] = Field(
        default="relevance", description="'date' for recent/latest papers, else 'relevance'."
    )

    @field_validator("category", mode="before")
    @classmethod
    def _check_category(cls, value: Any) -> Any:
        # Small models often send "" or "null" for "no category".
        if value is None or (isinstance(value, str) and value.strip() in ("", "null", "none")):
            return None
        if isinstance(value, str):
            return validate_category(value)  # InvalidQueryError is a ValueError: Pydantic error
        return value


def search_arxiv(args: SearchArxivArgs, client: ArxivClient) -> str:
    """Run the search and format the summaries as a compact text block."""
    try:
        query = SearchQuery.build(args.query, args.category, args.max_results, args.sort)
        papers = client.run_search(query)
    except ArxivError as exc:
        raise _to_tool_error(exc) from exc
    return format_search_results(query, papers)


def format_search_results(query: SearchQuery, papers: list[PaperSummary]) -> str:
    """Header, then two lines per paper, then the next-step hint. Bounded by the budget."""
    order = "newest first" if query.sort == "date" else "by relevance"
    header = f"arXiv search: {query.describe()} ({order})"
    if query.ignored:
        header += f" | ignored common words: {', '.join(query.ignored)}"
    if not papers:
        hint = "drop the category filter, " if query.category else ""
        return (
            f"{header}\nNo papers matched all keywords. Try fewer or broader keywords, "
            f"{hint}or synonyms (each keyword must appear in the paper)."
        )

    footer = (
        "Ids above are the only valid ones. For the full abstract of a promising paper, "
        "call get_paper_details with its id."
    )
    lines = [f"{header} | {len(papers)} result{'s' if len(papers) > 1 else ''}:"]
    used = len(lines[0]) + len(footer) + 2
    for index, paper in enumerate(papers, start=1):
        block = _format_summary(index, paper)
        if used + len(block) + 1 > SEARCH_OUTPUT_MAX_CHARS:
            lines.append(f"({len(papers) - index + 1} more results not shown: output budget.)")
            break
        lines.append(block)
        used += len(block) + 1
    lines.append(footer)
    return "\n".join(lines)


def _format_summary(index: int, paper: PaperSummary) -> str:
    year = paper.year if paper.year is not None else "n.d."
    category = paper.primary_category or "?"
    author = paper.first_author + (" et al." if paper.et_al else "")
    return (
        f"[{index}] {paper.arxiv_id} | {year} | {category} | {paper.title}\n"
        f"    {author}: {paper.snippet}"
    )


# --- get_paper_details ----------------------------------------------------------------------


class GetPaperDetailsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str = Field(
        min_length=1,
        max_length=64,
        description="arXiv id copied from search_arxiv results, e.g. '2301.12345'.",
    )

    @field_validator("arxiv_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        # Validated here, before the tool function runs: a malformed id never reaches the
        # client (which validates again, for direct callers).
        return str(parse_arxiv_id(value))


def get_paper_details(args: GetPaperDetailsArgs, client: ArxivClient) -> str:
    """Fetch one paper and format its full (capped) record."""
    try:
        paper = client.get(args.arxiv_id)
    except ArxivError as exc:
        raise _to_tool_error(exc) from exc
    if paper is None:
        raise ToolError(
            f"No arXiv paper has the id {args.arxiv_id!r}. Use an id exactly as returned by "
            "search_arxiv, or search again; do not guess ids.",
            error_type="not_found",
        )
    return format_paper_details(paper)


def format_paper_details(paper: PaperDetails) -> str:
    """Labelled lines; the abstract last, because it is the longest."""
    authors = ", ".join(paper.authors) or "Unknown"
    if paper.authors_total > len(paper.authors):
        authors += f" (+{paper.authors_total - len(paper.authors)} more)"
    published = paper.published.isoformat() if paper.published else "unknown"
    dates = f"Published: {published}"
    if paper.updated and paper.updated != paper.published:
        dates += f" | Updated: {paper.updated.isoformat()}"
    categories = ", ".join(
        f"{c} (primary)" if c == paper.primary_category else c for c in paper.categories
    )
    version = f" ({paper.version})" if paper.version else ""
    return "\n".join(
        [
            f"arXiv:{paper.arxiv_id}{version} | {paper.title}",
            f"Authors: {authors}",
            dates,
            f"Categories: {categories or 'unknown'}",
            f"PDF: {paper.pdf_url}",
            f"Abstract: {paper.abstract}",
        ]
    )


# --- errors and wiring ----------------------------------------------------------------------


def _to_tool_error(exc: ArxivError) -> ToolError:
    """Typed client error -> message telling the model what to do next."""
    if isinstance(exc, InvalidQueryError | InvalidArxivIdError):
        return ToolError(str(exc), error_type="invalid_arguments")
    if isinstance(exc, ArxivRateLimited):
        return ToolError(
            f"arXiv is rate limiting requests (retry possible in about {exc.retry_after_s:.0f} "
            "s). Do not retry now: answer with the results you already have, or tell the user "
            "to try again later.",
            error_type="arxiv_rate_limited",
        )
    if isinstance(exc, ArxivTimeout | ArxivUnavailable):
        return ToolError(
            "arXiv is temporarily unavailable (timeout or server error). Try again once with "
            "the same arguments; if it fails again, tell the user arXiv is unreachable.",
            error_type="arxiv_unavailable",
        )
    if isinstance(exc, ArxivHTTPError):
        return ToolError(
            f"arXiv rejected the request (HTTP {exc.status_code}). Retry with simpler "
            "keywords and no category.",
            error_type="arxiv_rejected",
        )
    if isinstance(exc, ArxivMalformedResponse):
        return ToolError(
            "arXiv returned an unreadable response. Try again once; if it fails again, tell "
            "the user arXiv returned an error.",
            error_type="arxiv_bad_response",
        )
    return ToolError(f"arXiv request failed: {exc}", error_type="arxiv_error")


def make_arxiv_tools(client: ArxivClient) -> list[Tool[Any]]:
    """The two arXiv tools, bound to `client` (shared: one rate limiter, one cache)."""
    return [
        Tool(
            name="search_arxiv",
            description=SEARCH_DESCRIPTION,
            args_model=SearchArxivArgs,
            function=partial(search_arxiv, client=client),
        ),
        Tool(
            name="get_paper_details",
            description=DETAILS_DESCRIPTION,
            args_model=GetPaperDetailsArgs,
            function=partial(get_paper_details, client=client),
        ),
    ]
