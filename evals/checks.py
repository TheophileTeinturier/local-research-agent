"""Checks: PURE functions `(trace, **params) -> (passed, detail)` over a `RunTrace`.

No check reads a clock, a file or the network, or mutates its input, so each one is unit-tested
with hand-built traces. A case declares checks by name + params (see `evals/cases/*.json`);
`validate_check` rejects unknown names, missing/unknown params, wrong types and bad regexes
when the suite is loaded, before any model runs.

Deterministic checks look at the tool trace, the stop reason and counters, plus two regex
checks on the answer whose pattern is unambiguous (e.g. the number 294).
Heuristic checks (`heuristic=True`) are regexes over free-form wording ("the paper was not
found"): useful signals, reported separately, never part of `passed`.

Conventions:
- "called" counts EXECUTED calls: repetition warnings (`repeated=True`) are not calls.
- A missing final answer is "" for the regex checks: `answer_matches` fails,
  `answer_not_matches` passes. The `stop_reason_is` check catches the missing answer.
- arXiv ids are compared without version ("1706.02216v2" -> "1706.02216").
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .records import CheckResult, RunTrace, ToolCallTrace

Verdict = tuple[bool, str]

# New style 1706.02216(v2), old style hep-th/9901001 or math.GT/0309136. Not inside a longer
# number ("12.1706.02216", "1706.022161") thanks to the lookarounds.
_NEW_ID = r"(?<![\d.])(?P<new>\d{4}\.\d{4,5})(?:v\d+)?(?!\d)"
_OLD_ID = r"\b(?P<archive>[a-z]+(?:-[a-z]+)?)(?:\.[A-Z]{2})?/(?P<number>\d{7})(?:v\d+)?(?!\d)"
ARXIV_ID_RE = re.compile(f"{_NEW_ID}|{_OLD_ID}")


def extract_arxiv_ids(text: str | None) -> set[str]:
    """Every arXiv id in `text`, normalized (no version, no old-style subject class)."""
    ids: set[str] = set()
    for match in ARXIV_ID_RE.finditer(text or ""):
        ids.add(match["new"] or f"{match['archive']}/{match['number']}")
    return ids


def _executed(trace: RunTrace, tool: str) -> list[ToolCallTrace]:
    return [c for c in trace.tool_calls if c.name == tool and not c.repeated]


def _answer(trace: RunTrace) -> str:
    return trace.final_answer or ""


def _excerpt(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --- deterministic checks -------------------------------------------------------------------


def tool_called(trace: RunTrace, *, tool: str, min: int = 1) -> Verdict:
    """`tool` was executed at least `min` times."""
    count = len(_executed(trace, tool))
    return count >= min, f"{tool} executed {count} time(s), need >= {min}"


def tool_not_called(trace: RunTrace, *, tool: str) -> Verdict:
    """`tool` was never requested (a repetition warning counts: the model asked for it)."""
    count = sum(c.name == tool for c in trace.tool_calls)
    return count == 0, f"{tool} requested {count} time(s)"


def no_tool_called(trace: RunTrace) -> Verdict:
    """The model requested no tool at all."""
    names = [c.name for c in trace.tool_calls]
    return not names, f"tools requested: {names}" if names else "no tool requested"


def tool_arg_min(trace: RunTrace, *, tool: str, arg: str, value: float) -> Verdict:
    """At least one executed `tool` call had numeric `arg` >= `value`.

    An absent argument does not count (the tool's default is not assumed here).
    """
    seen = [c.args().get(arg) for c in _executed(trace, tool)]
    numbers = [_as_number(v) for v in seen]
    passed = any(n is not None and n >= value for n in numbers)
    return passed, f"{tool}.{arg} values: {seen}, need one >= {value:g}"


def tool_arg_equals(trace: RunTrace, *, tool: str, arg: str, value: Any) -> Verdict:
    """At least one executed `tool` call had `arg` exactly equal to `value`."""
    seen = [c.args().get(arg) for c in _executed(trace, tool)]
    return value in seen, f"{tool}.{arg} values: {seen}, need {value!r}"


def tool_arg_covers(trace: RunTrace, *, tool: str, arg: str, values: list) -> Verdict:
    """Every one of `values` was the `arg` of at least one executed `tool` call, ok or failed.

    Strings are compared after stripping whitespace and a trailing arXiv version
    ("1706.02216v2" == "1706.02216"); nothing else is normalized ("arXiv:1706.02216" is a
    different value, as in `tool_arg_equals`). Repetition warnings are not calls.
    """
    seen = [c.args().get(arg) for c in _executed(trace, tool)]
    normalized = {_normalize_arg(v) for v in seen}
    missing = [v for v in values if _normalize_arg(v) not in normalized]
    if missing:
        return False, f"{tool}.{arg} values: {seen}, missing {missing}"
    return True, f"{tool}.{arg} values: {seen}, all of {values} covered"


def answer_matches(trace: RunTrace, *, regex: str) -> Verdict:
    """The final answer matches `regex` (re.search)."""
    match = re.search(regex, _answer(trace))
    if match:
        return True, f"matched {match.group(0)!r}"
    return False, f"no match in answer {_excerpt(_answer(trace))!r}"


def answer_not_matches(trace: RunTrace, *, regex: str) -> Verdict:
    """The final answer does not match `regex`."""
    match = re.search(regex, _answer(trace))
    if match:
        return False, f"unexpected match {match.group(0)!r}"
    return True, "no match"


def cited_ids_subset_of_tool_ids(trace: RunTrace, *, include_question: bool = True) -> Verdict:
    """Every arXiv id in the answer appears in a SUCCESSFUL tool result (or the question).

    Error results do not count: "No arXiv paper has the id X" is no evidence that X exists.
    The question counts by default, so "paper X could not be retrieved" is not an invention.
    """
    cited = extract_arxiv_ids(trace.final_answer)
    known: set[str] = set()
    for call in trace.tool_calls:
        if call.ok and not call.repeated:
            known |= extract_arxiv_ids(call.content)
    if include_question:
        known |= extract_arxiv_ids(trace.question)
    unknown = sorted(cited - known)
    if unknown:
        return False, f"cited ids not returned by any tool: {unknown}"
    return True, f"{len(cited)} cited id(s), all known"


DETAILS_TOOL = "get_paper_details"


def cited_ids_subset_of_opened_ids(trace: RunTrace) -> Verdict:
    """Every arXiv id in the answer was OPENED: the `arxiv_id` of a successful details call.

    Stricter than `cited_ids_subset_of_tool_ids`: an id seen only in search results (a title
    and a 200-char snippet) does not count, nor does the question. A failed or repeated call
    opens nothing (a repeat of a successful call is covered by that earlier call). The
    argument is normalized like the answer ("arXiv:1706.02216v2" -> "1706.02216").
    """
    cited = extract_arxiv_ids(trace.final_answer)
    opened: set[str] = set()
    for call in _executed(trace, DETAILS_TOOL):
        arg = call.args().get("arxiv_id")
        if call.ok and isinstance(arg, str):
            opened |= extract_arxiv_ids(arg)
    unopened = sorted(cited - opened)
    if unopened:
        return False, f"cited ids never opened with {DETAILS_TOOL}: {unopened}"
    return True, f"{len(cited)} cited id(s), all opened"


def cited_ids_count(
    trace: RunTrace,
    *,
    exactly: int | None = None,
    min: int | None = None,
    max: int | None = None,
) -> Verdict:
    """The answer cites `exactly` distinct arXiv ids, or between `min` and `max` (inclusive).

    Give `exactly`, or `min` and/or `max` (checked at load time by `_validate_count`).
    """
    cited = sorted(extract_arxiv_ids(trace.final_answer))
    n = len(cited)
    if exactly is not None:
        return n == exactly, f"{n} distinct id(s) cited {cited}, need {exactly}"
    passed = (min is None or n >= min) and (max is None or n <= max)
    bounds = " and ".join(
        part
        for part in (f">= {min}" if min is not None else "", f"<= {max}" if max is not None else "")
        if part
    )
    return passed, f"{n} distinct id(s) cited {cited}, need {bounds}"


def _validate_count(params: Mapping[str, Any]) -> None:
    given = {k for k in ("exactly", "min", "max") if params.get(k) is not None}
    if not given:
        raise CheckSpecError("cited_ids_count: give `exactly`, or `min` and/or `max`")
    if "exactly" in given and given != {"exactly"}:
        raise CheckSpecError("cited_ids_count: `exactly` cannot be combined with `min`/`max`")
    if given == {"min", "max"} and params["min"] > params["max"]:
        raise CheckSpecError(f"cited_ids_count: min {params['min']} > max {params['max']}")


def max_iterations(trace: RunTrace, *, n: int) -> Verdict:
    """At most `n` LLM calls."""
    return trace.iterations <= n, f"{trace.iterations} iteration(s), max {n}"


def max_tool_errors(trace: RunTrace, *, n: int) -> Verdict:
    """At most `n` failed tool executions."""
    return trace.tool_errors <= n, f"{trace.tool_errors} tool error(s), max {n}"


def stop_reason_is(trace: RunTrace, *, value: str) -> Verdict:
    """The run ended with stop reason `value`. Added to every case by the runner."""
    return trace.stop_reason == value, f"stop_reason={trace.stop_reason}, expected {value}"


# --- heuristic checks -----------------------------------------------------------------------


def heuristic_answer_matches(trace: RunTrace, *, regex: str) -> Verdict:
    """HEURISTIC: the answer's free wording matches `regex` (asks back, names the topic...).

    Same mechanics as `answer_matches`; separate so it is always reported as heuristic and
    never part of `passed`.
    """
    return answer_matches(trace, regex=regex)


def answer_signals_failure(trace: RunTrace, *, regex: str) -> Verdict:
    """HEURISTIC: the answer's wording says something went wrong (not found, unavailable...).

    `heuristic_answer_matches` under its original name, kept so the existing cases and the
    check labels of past batches stay unchanged.
    """
    return heuristic_answer_matches(trace, regex=regex)


# --- registry and validation ----------------------------------------------------------------


@dataclass(frozen=True)
class CheckDef:
    fn: Callable[..., Verdict]
    heuristic: bool = False
    # Cross-parameter rules the type hints cannot express; raises CheckSpecError.
    validate: Callable[[Mapping[str, Any]], None] | None = None


CHECKS: dict[str, CheckDef] = {
    "tool_called": CheckDef(tool_called),
    "tool_not_called": CheckDef(tool_not_called),
    "no_tool_called": CheckDef(no_tool_called),
    "tool_arg_min": CheckDef(tool_arg_min),
    "tool_arg_equals": CheckDef(tool_arg_equals),
    "tool_arg_covers": CheckDef(tool_arg_covers),
    "answer_matches": CheckDef(answer_matches),
    "answer_not_matches": CheckDef(answer_not_matches),
    "cited_ids_subset_of_tool_ids": CheckDef(cited_ids_subset_of_tool_ids),
    "cited_ids_subset_of_opened_ids": CheckDef(cited_ids_subset_of_opened_ids),
    "cited_ids_count": CheckDef(cited_ids_count, validate=_validate_count),
    "max_iterations": CheckDef(max_iterations),
    "max_tool_errors": CheckDef(max_tool_errors),
    "stop_reason_is": CheckDef(stop_reason_is),
    "heuristic_answer_matches": CheckDef(heuristic_answer_matches, heuristic=True),
    "answer_signals_failure": CheckDef(answer_signals_failure, heuristic=True),
}


class CheckSpecError(ValueError):
    """A check declaration is invalid (unknown name, bad params)."""


def validate_check(name: str, params: Mapping[str, Any]) -> None:
    """Raise `CheckSpecError` unless `name(**params)` is a well-formed check."""
    definition = CHECKS.get(name)
    if definition is None:
        raise CheckSpecError(f"unknown check {name!r}; known: {', '.join(sorted(CHECKS))}")
    hints = typing.get_type_hints(definition.fn)
    accepted = {p for p in hints if p not in ("trace", "return")}
    required = _required_params(definition.fn)
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise CheckSpecError(f"{name}: unknown param(s) {unknown}; accepted: {sorted(accepted)}")
    missing = sorted(required - set(params))
    if missing:
        raise CheckSpecError(f"{name}: missing param(s) {missing}")
    for key, value in params.items():
        if not _type_ok(value, hints[key]):
            raise CheckSpecError(f"{name}: param {key!r}={value!r} is not {hints[key]}")
        if key == "regex":
            try:
                re.compile(value)
            except re.error as exc:
                raise CheckSpecError(f"{name}: invalid regex {value!r}: {exc}") from None
    if definition.validate is not None:
        definition.validate(params)


def run_check(
    name: str, params: Mapping[str, Any], trace: RunTrace, *, heuristic: bool | None = None
) -> CheckResult:
    """Evaluate one declared check. `heuristic` overrides the check's default flag."""
    definition = CHECKS[name]
    passed, detail = definition.fn(trace, **params)
    return CheckResult(
        name=check_label(name, params),
        passed=bool(passed),
        heuristic=definition.heuristic if heuristic is None else heuristic,
        detail=detail,
    )


def check_label(name: str, params: Mapping[str, Any]) -> str:
    """Stable readable name, e.g. "tool_arg_min(tool=search_arxiv, arg=max_results, value=6)"."""
    inner = ", ".join(f"{key}={value}" for key, value in params.items())
    return f"{name}({inner})"


def _required_params(fn: Callable[..., Verdict]) -> set[str]:
    return {
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    }


def _type_ok(value: Any, hint: Any) -> bool:
    if hint is Any:
        return True
    if isinstance(hint, types.UnionType):  # e.g. `int | None`
        return any(_type_ok(value, member) for member in typing.get_args(hint))
    if hint is type(None):
        return value is None
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:  # an int is fine where a float is expected
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, hint)


def _normalize_arg(value: Any) -> Any:
    """Strings: stripped, trailing arXiv version removed. Other values: hashable as is."""
    if isinstance(value, str):
        return re.sub(r"v\d+$", "", value.strip())
    try:
        hash(value)
    except TypeError:
        return repr(value)  # a dict/list argument: compared by its text
    return value


def _as_number(value: Any) -> float | None:
    """A number, or a numeric string as small models sometimes send ("10"); else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
