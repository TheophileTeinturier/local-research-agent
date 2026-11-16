"""Run cases against an `LLMClient` and turn each run into exactly one `RunRecord`.

`run_case` builds the case's world (fake arXiv transport, frozen `current_date`), calls the
UNCHANGED `run_agent`, then evaluates the checks. It never raises (except Ctrl-C/SystemExit):
any exception, in the agent or in the harness, becomes a record with `harness_error`.

`run_batch` runs every case `repetitions` times, repetition-major (all cases once, then all
again), so a slow drift of the machine (thermal throttling, other load) is spread across cases.
`warm_up` makes one untimed call first so the model load is not charged to the first case.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from local_research_agent.agent import (
    DEFAULT_PROMPT_VARIANT,
    AgentConfig,
    AgentLimits,
    AgentResult,
    get_system_prompt,
    run_agent,
)
from local_research_agent.llm import ChatOptions, LLMClient, LLMError, user_message
from local_research_agent.llm.ollama import DEFAULT_MODEL
from local_research_agent.tools import build_registry

from .arxiv_fake import CaseArxivTransport
from .checks import run_check
from .records import RunRecord, RunTrace, ToolCallTrace
from .suite import FIXTURES_DIR, Case

_DEFAULT_OPTIONS = ChatOptions()

WARMUP_PROMPT = "Reply with the single word OK."
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


@dataclass(frozen=True)
class EvalConfig:
    """Everything that defines a batch. Stored with the batch, so batches are comparable."""

    model: str = DEFAULT_MODEL
    think: bool = False
    prompt_variant: str = DEFAULT_PROMPT_VARIANT
    num_ctx: int = _DEFAULT_OPTIONS.num_ctx
    temperature: float = _DEFAULT_OPTIONS.temperature
    seed: int | None = None  # None: each run uses its repetition index as seed
    repetitions: int = 1
    label: str = "unlabeled"
    limits: AgentLimits = field(default_factory=AgentLimits)

    def __post_init__(self) -> None:
        get_system_prompt(self.prompt_variant)  # ValueError for an unknown variant
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if not LABEL_RE.fullmatch(self.label):
            # The label becomes a file name (evals/results/<label>.md).
            raise ValueError(f"label {self.label!r} must match {LABEL_RE.pattern}")

    def seed_for(self, repetition: int) -> int:
        return self.seed if self.seed is not None else repetition

    def chat_options(self, repetition: int) -> ChatOptions:
        return ChatOptions(
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            think=self.think,
            seed=self.seed_for(repetition),
        )

    def agent_config(self, repetition: int) -> AgentConfig:
        return AgentConfig(
            system_prompt=get_system_prompt(self.prompt_variant),
            limits=self.limits,
            llm_options=self.chat_options(repetition),
        )

    def to_json(self) -> dict[str, Any]:
        """The config plus a hash of the system prompt text (a variant's text may change)."""
        prompt = get_system_prompt(self.prompt_variant)
        return {
            **asdict(self),
            "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }


@dataclass(frozen=True)
class WarmupResult:
    """The untimed model-loading call. Recorded with the batch, excluded from all metrics."""

    ok: bool
    latency_ms: float
    load_ms: float
    error: str | None = None


def warm_up(llm: LLMClient, config: EvalConfig) -> WarmupResult:
    """One tiny chat with the batch's options (a different num_ctx would reload the model)."""
    start = time.perf_counter()
    try:
        response = llm.chat([user_message(WARMUP_PROMPT)], options=config.chat_options(0))
    except LLMError as exc:
        return WarmupResult(False, _ms_since(start), 0.0, str(exc))
    load_ms = response.metrics.load_duration_ms or 0.0
    return WarmupResult(True, _ms_since(start), round(load_ms, 1))


def run_case(
    case: Case,
    config: EvalConfig,
    repetition: int,
    *,
    llm: LLMClient,
    fixtures_dir: Path = FIXTURES_DIR,
) -> RunRecord:
    """Run one case once. ALWAYS returns a record; never raises for an ordinary exception."""
    start = time.perf_counter()
    base: dict[str, Any] = {
        "case_id": case.id,
        "category": case.category,
        "repetition": repetition,
        "model": config.model,
        "think": config.think,
        "prompt_variant": config.prompt_variant,
        "seed": config.seed_for(repetition),
        "expected_stop_reason": case.expected_stop_reason,
    }
    try:
        transport = CaseArxivTransport(case.arxiv, fixtures_dir)
        registry = build_registry(arxiv_transport=transport, clock=lambda: case.frozen_now)
        result = run_agent(llm, registry, config.agent_config(repetition), case.question)
        trace = trace_from_result(case.question, result)
        checks = tuple(
            run_check(spec.name, spec.params, trace, heuristic=spec.heuristic)
            for spec in case.all_checks()
        )
    except Exception as exc:  # deliberately broad: one broken run must not end the batch
        empty = RunTrace(case.question, None, None, 0, (), 0)
        return RunRecord(
            **base,
            trace=empty,
            wall_time_ms=_ms_since(start),
            harness_error=f"{type(exc).__name__}: {exc}",
        )

    metrics = result.metrics
    return RunRecord(
        **base,
        trace=trace,
        llm_calls=len(result.state.llm_calls),
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        llm_latency_ms=metrics.llm_latency_ms,
        llm_load_ms=metrics.llm_load_ms,
        wall_time_ms=metrics.wall_time_ms,
        agent_error=result.state.error,
        fixture_misses=tuple(transport.misses),
        checks=checks,
    )


def run_batch(
    cases: Sequence[Case],
    config: EvalConfig,
    *,
    llm: LLMClient,
    on_record: Callable[[RunRecord], None] = lambda _record: None,
    fixtures_dir: Path = FIXTURES_DIR,
) -> list[RunRecord]:
    """Every case `config.repetitions` times; `on_record` sees each record as it is made."""
    records = []
    for repetition in range(config.repetitions):
        for case in cases:
            record = run_case(case, config, repetition, llm=llm, fixtures_dir=fixtures_dir)
            on_record(record)
            records.append(record)
    return records


def trace_from_result(question: str, result: AgentResult) -> RunTrace:
    """The check-facing view of an `AgentResult`."""
    calls = tuple(
        ToolCallTrace(
            iteration=c.iteration,
            name=c.name,
            arguments=c.arguments,
            ok=c.ok,
            content=c.content,
            repeated=c.repeated,
            error_type=c.error_type,
        )
        for c in result.state.tool_calls
    )
    return RunTrace(
        question=question,
        final_answer=result.final_answer,
        stop_reason=result.stop_reason.value,
        iterations=result.metrics.iterations,
        tool_calls=calls,
        tool_errors=result.metrics.tool_errors,
    )


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
