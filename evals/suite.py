"""Case schema and suite loader. A suite is a JSON file: `{"cases": [Case, ...]}`.

    {
      "id": "calc_basic",                       # unique, [a-z0-9_]+
      "category": "single_tool",
      "question": "What is 12% of 2450?",
      "arxiv": {                                # optional: what the fake arXiv serves
        "search": "search_x.xml",               #   any search -> this feed, cut to max_results
        "papers": {"1706.02216": "paper.xml"},  #   lookups; hits of `search` are served too;
                                                #   any other id -> arXiv's "not found"
        "failure_mode": "http_503",             #   or "timeout": every request fails
        "default_search": "empty"               #   no `search` fixture: answer any search with
                                                #   0 results instead of a fixture miss
      },
      "frozen_now": "2026-11-09T10:18:18",      # what current_date returns (this default)
      "checks": [{"name": "tool_called", "params": {"tool": "calculator"}},
                 {"name": "answer_signals_failure", "params": {"regex": "..."}}],
      "expected_stop_reason": "final_answer",   # this default; becomes a stop_reason_is check
      "synthetic": false                        # true: serves a hand-written fixture
    }

Fixture file names are relative to `evals/fixtures/arxiv/` (hand-written ones live in
`synthetic/`, e.g. "synthetic/paper_2601.00001.xml"). `load_suite` raises `SuiteError`
with the file, the case and the field at fault; nothing runs until the whole suite is valid.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from local_research_agent.agent import StopReason

from .checks import CheckSpecError, validate_check

EVALS_DIR = Path(__file__).parent
CASES_DIR = EVALS_DIR / "cases"
FIXTURES_DIR = EVALS_DIR / "fixtures" / "arxiv"

NOT_FOUND_FILE = "not_found.xml"  # arXiv's real answer for an unknown id, used by every case
MANIFEST_FILE = "MANIFEST.json"  # file -> request it recorded; part of the suite hash

DEFAULT_FROZEN_NOW = datetime.datetime(2026, 11, 9, 10, 18, 18)

FailureMode = Literal["http_503", "timeout"]


class SuiteError(ValueError):
    """The suite file is missing, is not valid JSON, or a case is invalid."""


class CheckSpec(BaseModel):
    """One declared check: a name from `checks.CHECKS` and its keyword params."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    heuristic: bool | None = None  # None = the check's own flag

    @model_validator(mode="after")
    def _valid_check(self) -> CheckSpec:
        try:
            validate_check(self.name, self.params)
        except CheckSpecError as exc:
            raise ValueError(str(exc)) from None
        return self


class ArxivSpec(BaseModel):
    """What the case's fake arXiv serves (see `arxiv_fake.CaseArxivTransport`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    search: str | None = None
    papers: dict[str, str] = Field(default_factory=dict)
    failure_mode: FailureMode | None = None
    # What an uncovered search gets when the case has no `search` fixture. None: it is a
    # fixture miss (the run is inconclusive). "empty": a zero-result Atom feed, a normal answer
    # (for cases designed around a lookup tool, where a fallback search is plausible).
    default_search: Literal["empty"] | None = None

    @model_validator(mode="after")
    def _one_search_source(self) -> ArxivSpec:
        if self.search is not None and self.default_search is not None:
            raise ValueError("give either `search` (a fixture) or `default_search`, not both")
        return self


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    arxiv: ArxivSpec = Field(default_factory=ArxivSpec)
    frozen_now: datetime.datetime = DEFAULT_FROZEN_NOW
    checks: tuple[CheckSpec, ...] = Field(min_length=1)
    expected_stop_reason: str = StopReason.FINAL_ANSWER.value
    # True when the case serves a HAND-WRITTEN fixture (not recorded from arXiv), e.g. an
    # abstract carrying a prompt injection. Tagged "synthetic" in the report.
    synthetic: bool = False

    @field_validator("expected_stop_reason")
    @classmethod
    def _known_stop_reason(cls, value: str) -> str:
        known = [r.value for r in StopReason]
        if value not in known:
            raise ValueError(f"unknown stop reason {value!r}; known: {known}")
        return value

    def all_checks(self) -> tuple[CheckSpec, ...]:
        """The declared checks plus the implicit `stop_reason_is(expected_stop_reason)`."""
        implicit = CheckSpec(name="stop_reason_is", params={"value": self.expected_stop_reason})
        return (*self.checks, implicit)


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = ""
    cases: tuple[Case, ...] = Field(min_length=1)


def resolve_suite_path(suite: str) -> Path:
    """`minimal` -> evals/cases/minimal.json; anything with a suffix is used as a path."""
    path = Path(suite)
    return path if path.suffix else CASES_DIR / f"{suite}.json"


def load_suite(path: str | Path, *, fixtures_dir: Path = FIXTURES_DIR) -> Suite:
    """Parse and validate a suite file. Raises `SuiteError` with a precise location."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SuiteError(f"{path}: suite file not found") from None
    except json.JSONDecodeError as exc:
        raise SuiteError(f"{path}: invalid JSON at line {exc.lineno}: {exc.msg}") from None
    try:
        suite = Suite.model_validate(raw)
    except ValidationError as exc:
        raise SuiteError(f"{path}: {_format_errors(exc, raw)}") from None

    problems = _duplicate_ids(suite) + _missing_fixtures(suite, fixtures_dir)
    if problems:
        raise SuiteError(f"{path}: " + "; ".join(problems))
    return suite


def suite_hashes(path: str | Path, *, fixtures_dir: Path = FIXTURES_DIR) -> dict[str, str]:
    """What the batch config stores to tell whether two batches ran the same suite.

    `suite_sha256` = sha256 over the two component hashes (cases file, then the fixtures'
    MANIFEST.json, raw bytes). A missing manifest hashes as the literal "missing", so its
    absence is visible too. The fixture XML files themselves are NOT hashed: a re-recorded
    fixture is detected only through the manifest (re-capture rewrites `captured_at`).
    """
    cases = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest_path = fixtures_dir / MANIFEST_FILE
    manifest = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path.is_file()
        else "missing"
    )
    combined = hashlib.sha256(f"cases:{cases}\nmanifest:{manifest}\n".encode()).hexdigest()
    return {
        "suite_sha256": combined,
        "cases_sha256": cases,
        "fixtures_manifest_sha256": manifest,
    }


def select_cases(suite: Suite, ids: list[str] | None) -> tuple[Case, ...]:
    """The cases named in `ids` (suite order), or all. Raises `SuiteError` for unknown ids."""
    if not ids:
        return suite.cases
    known = {case.id for case in suite.cases}
    unknown = sorted(set(ids) - known)
    if unknown:
        raise SuiteError(f"unknown case id(s) {unknown}; known: {sorted(known)}")
    return tuple(case for case in suite.cases if case.id in ids)


def _format_errors(exc: ValidationError, raw: Any) -> str:
    """'case calc_basic (cases.0) checks.1: <message>' for every error."""
    parts = []
    for err in exc.errors(include_url=False):
        loc = list(err["loc"])
        where = ".".join(str(p) for p in loc) or "<root>"
        if len(loc) >= 2 and loc[0] == "cases" and isinstance(loc[1], int):
            case_id = _case_id(raw, loc[1])
            rest = ".".join(str(p) for p in loc[2:]) or "<case>"
            where = f"case {case_id} (cases.{loc[1]}) {rest}"
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


def _case_id(raw: Any, index: int) -> str:
    try:
        return repr(raw["cases"][index]["id"])
    except (KeyError, IndexError, TypeError):
        return "<no id>"


def _duplicate_ids(suite: Suite) -> list[str]:
    seen: set[str] = set()
    duplicates = []
    for case in suite.cases:
        if case.id in seen:
            duplicates.append(f"duplicate case id {case.id!r}")
        seen.add(case.id)
    return duplicates


def _missing_fixtures(suite: Suite, fixtures_dir: Path) -> list[str]:
    problems = []
    if not (fixtures_dir / NOT_FOUND_FILE).is_file():
        problems.append(f"fixture {NOT_FOUND_FILE!r} (served for unknown ids) not found")
    for case in suite.cases:
        files = [case.arxiv.search, *case.arxiv.papers.values()]
        for name in filter(None, files):
            if not (fixtures_dir / name).is_file():
                problems.append(f"case {case.id!r}: fixture {name!r} not found in {fixtures_dir}")
    return problems
