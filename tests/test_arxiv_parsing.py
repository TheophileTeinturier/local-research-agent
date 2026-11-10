"""Atom parsing and model caps: recorded real responses plus hand-written edge cases."""

import datetime

import pytest

from fakes import ARXIV_FIXTURES
from local_research_agent.tools.arxiv.errors import ArxivMalformedResponse
from local_research_agent.tools.arxiv.models import (
    ABSTRACT_MAX_CHARS,
    MAX_AUTHORS,
    SNIPPET_MAX_CHARS,
    TITLE_MAX_CHARS,
    PaperDetails,
    truncate_words,
)
from local_research_agent.tools.arxiv.parsing import parse_feed

NS = (
    'xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" '
    'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"'
)


def feed(*entries: str, total: int | None = None) -> str:
    total_xml = (
        "" if total is None else f"<opensearch:totalResults>{total}</opensearch:totalResults>"
    )
    return f"<?xml version='1.0' encoding='UTF-8'?><feed {NS}>{total_xml}{''.join(entries)}</feed>"


# --- recorded responses ----------------------------------------------------------------------


def test_real_search_response() -> None:
    parsed = parse_feed((ARXIV_FIXTURES / "search_gnn_ppi.xml").read_bytes())
    assert parsed.total_results is not None and parsed.total_results >= 5
    assert len(parsed.papers) == 5
    first = parsed.papers[0]
    assert first.arxiv_id == "2606.01781"
    assert first.version == "v1"
    assert first.title.startswith("Structure-Guided Adaptive Propagation")
    assert first.published == datetime.date(2026, 6, 1)
    assert first.primary_category == "cs.AI"
    assert first.pdf_url == "https://arxiv.org/pdf/2606.01781v1"
    assert first.authors_total == 6
    # sort=date: newest first.
    dates = [p.published for p in parsed.papers]
    assert dates == sorted(dates, reverse=True)  # type: ignore[type-var]
    for paper in parsed.papers:
        assert "\n" not in paper.title and "  " not in paper.abstract  # whitespace collapsed


def test_real_details_response() -> None:
    [paper] = parse_feed((ARXIV_FIXTURES / "paper_details.xml").read_bytes()).papers
    assert paper.authors[:2] == ["Enqiang Zhu", "Yizi Liu"]
    assert paper.categories == ["cs.AI"]


@pytest.mark.parametrize("name", ["search_no_results.xml", "unknown_id.xml"])
def test_real_empty_responses(name: str) -> None:
    parsed = parse_feed((ARXIV_FIXTURES / name).read_bytes())
    assert parsed.papers == []
    assert parsed.total_results == 0


# --- edge cases ------------------------------------------------------------------------------


def test_entry_with_only_an_id_gets_placeholders() -> None:
    [paper] = parse_feed(feed("<entry><id>http://arxiv.org/abs/2301.12345v2</id></entry>")).papers
    assert paper == PaperDetails(
        arxiv_id="2301.12345",
        version="v2",
        title="(untitled)",
        authors=[],
        authors_total=0,
        published=None,
        updated=None,
        primary_category=None,
        categories=[],
        abstract="(no abstract)",
        pdf_url="https://arxiv.org/pdf/2301.12345v2",
    )
    summary = paper.to_summary()
    assert summary.first_author == "Unknown author"
    assert summary.et_al is False
    assert summary.year is None


def test_old_style_id_and_primary_category_fallback() -> None:
    entry = """<entry><id>http://arxiv.org/abs/hep-th/9901001v3</id>
      <category term="hep-th"/><category term="hep-ph"/><category term="hep-th"/>
      <published>not a date</published></entry>"""
    [paper] = parse_feed(feed(entry)).papers
    assert (paper.arxiv_id, paper.version) == ("hep-th/9901001", "v3")
    assert paper.primary_category == "hep-th"  # no arxiv:primary_category: first category
    assert paper.categories == ["hep-th", "hep-ph"]  # deduplicated
    assert paper.published is None  # unparseable date


def test_primary_category_is_listed_first() -> None:
    entry = """<entry><id>http://arxiv.org/abs/2301.12345v1</id>
      <category term="cs.LG"/><category term="q-bio.QM"/>
      <arxiv:primary_category term="q-bio.QM"/></entry>"""
    [paper] = parse_feed(feed(entry)).papers
    assert paper.categories == ["q-bio.QM", "cs.LG"]


def test_entry_without_usable_id_is_skipped() -> None:
    good = "<entry><id>http://arxiv.org/abs/2301.12345v1</id></entry>"
    parsed = parse_feed(feed("<entry><title>no id</title></entry>", good, total=2))
    assert [p.arxiv_id for p in parsed.papers] == ["2301.12345"]
    assert parsed.total_results == 2


def test_api_error_entry_raises() -> None:
    entry = """<entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_1234.12345</id>
      <summary>incorrect id format for 1234.12345</summary></entry>"""
    with pytest.raises(ArxivMalformedResponse, match="incorrect id format"):
        parse_feed(feed(entry))


@pytest.mark.parametrize(
    "body",
    [
        "",
        "<feed",
        "<html><body>Service Unavailable</body></html>",  # wrong root element
        "<feed>not atom</feed>",  # no Atom namespace
        # A DTD (entity expansion attack vector) is refused outright by defusedxml.
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
        f"<feed {NS}><title>&lol;</title></feed>",
    ],
)
def test_malformed_bodies_raise_a_typed_error(body: str) -> None:
    with pytest.raises(ArxivMalformedResponse):
        parse_feed(body)


def test_author_list_is_capped_but_counted() -> None:
    authors = "".join(f"<author><name>Author {i}</name></author>" for i in range(25))
    entry = f"<entry><id>http://arxiv.org/abs/2301.12345v1</id>{authors}</entry>"
    [paper] = parse_feed(feed(entry)).papers
    assert len(paper.authors) == MAX_AUTHORS
    assert paper.authors_total == 25
    assert paper.to_summary().et_al is True


def test_long_fields_are_capped() -> None:
    long_text = " ".join(["word"] * 1000)
    entry = (
        f"<entry><id>http://arxiv.org/abs/2301.12345v1</id><title>{long_text}</title>"
        f"<summary>{long_text}</summary></entry>"
    )
    [paper] = parse_feed(feed(entry)).papers
    assert len(paper.title) <= TITLE_MAX_CHARS
    assert len(paper.abstract) <= ABSTRACT_MAX_CHARS
    assert len(paper.to_summary().snippet) <= SNIPPET_MAX_CHARS


# --- truncate_words --------------------------------------------------------------------------


def test_short_text_is_unchanged_except_whitespace() -> None:
    assert truncate_words("  a\n  b  ", 10) == "a b"


def test_cut_on_word_boundary() -> None:
    text = "Graph neural networks predict protein interactions"
    result = truncate_words(text, 30)
    assert result == "Graph neural networks..."
    assert len(result) <= 30


def test_word_ending_exactly_at_the_budget_is_kept() -> None:
    # budget = 13 - 3 = 10 chars: "aaaa bbbbb" fits exactly, then a space.
    assert truncate_words("aaaa bbbbb ccc", 13) == "aaaa bbbbb..."


def test_trailing_punctuation_is_removed_before_the_ellipsis() -> None:
    assert truncate_words("one, two, three, four", 12) == "one, two..."


def test_single_long_word_is_hard_cut() -> None:
    assert truncate_words("x" * 50, 10) == "xxxxxxx..."


@pytest.mark.parametrize("limit", [10, 50, 200])
def test_truncation_never_exceeds_limit(limit: int) -> None:
    text = "lorem ipsum dolor sit amet consectetur " * 20
    assert len(truncate_words(text, limit)) <= limit
