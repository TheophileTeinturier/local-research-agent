"""Pure input handling: id validation, category format, keyword -> arXiv query (no I/O)."""

import pytest

from local_research_agent.tools.arxiv.errors import InvalidArxivIdError, InvalidQueryError
from local_research_agent.tools.arxiv.query import (
    ArxivId,
    SearchQuery,
    category_clause,
    lookup_params,
    parse_arxiv_id,
    split_keywords,
    validate_category,
)

# --- identifiers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2301.12345", ArxivId("2301.12345")),
        ("2301.12345v2", ArxivId("2301.12345", "v2")),
        ("0704.0001", ArxivId("0704.0001")),  # first new-style month, 4 digits
        ("1412.9999v10", ArxivId("1412.9999", "v10")),  # last 4-digit month
        ("1501.00001", ArxivId("1501.00001")),  # 5 digits from 2015
        ("hep-th/9901001", ArxivId("hep-th/9901001")),
        ("hep-th/9901001v3", ArxivId("hep-th/9901001", "v3")),
        ("cond-mat/0011267", ArxivId("cond-mat/0011267")),
        ("math.GT/0309136", ArxivId("math/0309136")),  # subject class dropped (API needs it)
        ("arXiv:2301.12345", ArxivId("2301.12345")),
        ("  ARXIV: 2301.12345v1 ", ArxivId("2301.12345", "v1")),
    ],
)
def test_valid_ids(raw: str, expected: ArxivId) -> None:
    assert parse_arxiv_id(raw) == expected


def test_id_str_includes_version() -> None:
    assert str(ArxivId("2301.12345", "v2")) == "2301.12345v2"
    assert str(ArxivId("hep-th/9901001")) == "hep-th/9901001"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "2301",
        "2301.123",  # too short
        "2301.123456",  # too long
        "2313.12345",  # month 13
        "2301.1234",  # 2015+ ids have 5 digits
        "1412.12345",  # pre-2015 ids have 4 digits
        "2301.12345v0",  # versions start at v1
        "2301.12345v",
        "hep-th/990100",  # old style needs 7 digits
        "HEP-TH/9901001",
        "https://arxiv.org/abs/2301.12345",
        "10.48550/arXiv.2301.12345",  # a DOI, not an id
        # Injection attempts: anything beyond the id grammar is rejected, never forwarded.
        "1234.5678 OR all:x",
        "2301.12345 OR all:x",
        "2301.12345,2302.00001",  # id_list is comma separated: no smuggling a second id
        "2301.12345&max_results=100",
        "all:electron",
        '2301.12345" OR "x',
    ],
)
def test_invalid_ids_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidArxivIdError, match=r"not a valid arXiv identifier|not an arXiv"):
        parse_arxiv_id(raw)


def test_digit_count_error_explains_the_rule() -> None:
    with pytest.raises(InvalidArxivIdError, match="ids from 2301 have 5 digits"):
        parse_arxiv_id("2301.1234")


def test_lookup_uses_id_list_never_search_query() -> None:
    assert lookup_params(ArxivId("2301.12345", "v2")) == {
        "id_list": "2301.12345v2",
        "max_results": "1",
    }


# --- categories ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category",
    ["cs.LG", "q-bio.QM", "stat.ML", "hep-th", "math.AP", "physics.optics", "cond-mat.str-el"],
)
def test_valid_categories(category: str) -> None:
    assert validate_category(f" {category} ") == category


@pytest.mark.parametrize(
    "category",
    ["cs LG", "cs.LG OR all:x", "cat:cs.LG", "cs.LG)", "CS.LG", "cs.*", "cs.", "machine learning"],
)
def test_invalid_categories(category: str) -> None:
    with pytest.raises(InvalidQueryError, match="Invalid category"):
        validate_category(category)


def test_category_clause() -> None:
    assert category_clause("cs.LG") == "cat:cs.LG"
    # Live probe: cat:cs matches nothing, cat:cs.* matches the subject classes.
    assert category_clause("hep-th") == "(cat:hep-th OR cat:hep-th.*)"


# --- keywords -> query ---------------------------------------------------------------------


def test_keywords_are_anded_over_all_fields() -> None:
    query = SearchQuery.build("graph neural network", None, 5, "relevance")
    assert query.search_query() == "all:graph AND all:neural AND all:network"


def test_hyphenated_word_and_quoted_phrase_become_phrases() -> None:
    query = SearchQuery.build('protein-protein "graph attention network"', None, 5, "relevance")
    assert query.terms == ("protein protein", "graph attention network")
    assert query.search_query() == 'all:"protein protein" AND all:"graph attention network"'


def test_stopwords_and_request_words_are_dropped_and_reported() -> None:
    terms, ignored = split_keywords("Recent papers on graph neural networks for the PPI")
    assert terms == ("graph", "neural", "networks", "ppi")
    assert ignored == ("recent", "papers", "on", "for", "the")


@pytest.mark.parametrize(
    ("keywords", "expected"),
    [
        # Operators: lowercased words that are stopwords, so they vanish.
        ("graph OR electron", "all:graph AND all:electron"),
        ("graph ANDNOT electron", "all:graph AND all:electron"),
        # Field prefixes, quotes, parentheses, wildcards: punctuation is a separator.
        ("ti:graph", 'all:"ti graph"'),
        ('graph" OR all:"x', 'all:graph AND all:"all x"'),
        ("(graph) AND cat:cs.*", 'all:graph AND all:"cat cs"'),
        ("graph%20OR%20x", 'all:"graph 20or 20x"'),
        ("C++ GNN's", 'all:c AND all:"gnn s"'),
    ],
)
def test_query_syntax_in_keywords_cannot_survive(keywords: str, expected: str) -> None:
    query = SearchQuery.build(keywords, None, 5, "relevance")
    assert query.search_query() == expected
    for term in query.terms:
        assert all(ch.isalnum() or ch == " " for ch in term)


def test_unicode_letters_are_kept() -> None:
    assert split_keywords("Schrödinger équation")[0] == ("schrödinger", "équation")


def test_duplicate_terms_are_dropped() -> None:
    assert split_keywords("graph Graph GRAPH network")[0] == ("graph", "network")


def test_category_and_sort_params() -> None:
    query = SearchQuery.build("graph", "cs.LG", 3, "date")
    assert query.params() == {
        "search_query": "all:graph AND cat:cs.LG",
        "start": "0",
        "max_results": "3",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    assert SearchQuery.build("graph", None, 5, "relevance").params()["sortBy"] == "relevance"


def test_describe_is_readable() -> None:
    query = SearchQuery.build("protein-protein interaction", "q-bio.QM", 5, "relevance")
    assert query.describe() == '"protein protein" AND interaction in q-bio.QM'


@pytest.mark.parametrize(
    ("keywords", "max_results", "message"),
    [
        ("   ", 5, "empty"),
        ("x" * 201, 5, "too long"),
        ("the of and", 5, "no usable keywords"),
        ('"" ()', 5, "no usable keywords"),
        ("a1 a2 a3 a4 a5 a6 a7 a8 a9", 5, "Too many keywords"),
        ("graph", 0, "between 1 and 10"),
        ("graph", 11, "between 1 and 10"),
    ],
)
def test_invalid_searches(keywords: str, max_results: int, message: str) -> None:
    with pytest.raises(InvalidQueryError, match=message):
        SearchQuery.build(keywords, None, max_results, "relevance")


def test_invalid_category_in_search() -> None:
    with pytest.raises(InvalidQueryError, match="Invalid category"):
        SearchQuery.build("graph", "cs.LG OR all:x", 5, "relevance")
