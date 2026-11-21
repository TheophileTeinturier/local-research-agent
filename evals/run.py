"""Run a suite against Ollama and store the batch.

    uv run python -m evals.run --suite minimal --model qwen3:8b --think off \
        --prompt-variant baseline --repeat 3 --label baseline [--cases id1,id2]

One line per run while it goes, then a summary. Every record is saved as soon as it exists,
so an interrupted batch keeps its finished runs. Exit code: 0 when the batch completed
(whatever the pass rate), 2 for a usage error, an invalid suite, a failed warm-up or (with
`--require-clean`) uncommitted changes in the working tree,
3 (with a warning on stderr) when more than 10% of the runs are inconclusive (fixture miss);
the batch is stored anyway.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from local_research_agent.agent import DEFAULT_PROMPT_VARIANT, PROMPT_VARIANTS
from local_research_agent.llm import ChatOptions, OllamaClient
from local_research_agent.llm.ollama import DEFAULT_MODEL
from local_research_agent.observability import configure_logging

from . import provenance
from .provenance import read_ollama_info, working_tree_changes
from .records import RunRecord
from .report import INCONCLUSIVE_LIMIT, summarize
from .runner import TOOL_BACKENDS, EvalConfig, run_batch, warm_up
from .store import DEFAULT_DB_PATH, ResultStore
from .suite import (
    EVALS_DIR,
    SuiteError,
    load_suite,
    resolve_suite_path,
    select_cases,
    suite_hashes,
)

REPO_ROOT = EVALS_DIR.parent

EXIT_TOO_MANY_INCONCLUSIVE = 3  # 2 is taken by usage errors and a failed warm-up


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments of `evals.run`."""
    options = ChatOptions()
    parser = argparse.ArgumentParser(prog="evals.run", description="Run an evaluation suite.")
    parser.add_argument("--suite", default="minimal", help="name in evals/cases/ or a path")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--host", help="Ollama URL (default: $OLLAMA_HOST or 127.0.0.1:11434)")
    parser.add_argument("--think", choices=["on", "off"], default="off")
    parser.add_argument(
        "--prompt-variant", choices=list(PROMPT_VARIANTS), default=DEFAULT_PROMPT_VARIANT
    )
    parser.add_argument("--num-ctx", type=int, default=options.num_ctx)
    parser.add_argument(
        "--temperature",
        type=float,
        default=options.temperature,
        help=f"sampling temperature, stored in the batch config (default {options.temperature})",
    )
    parser.add_argument("--seed", type=int, help="fixed seed (default: the repetition index)")
    parser.add_argument("--repeat", type=int, default=1, help="repetitions per case")
    parser.add_argument(
        "--tools",
        choices=list(TOOL_BACKENDS),
        default="local",
        help="tools in-process, or in an MCP server subprocess (one per run)",
    )
    parser.add_argument("--label", required=True, help="batch name, used by evals.report")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--log-level", default="WARNING", type=str.upper, help="agent logs on stderr"
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="refuse to start if the git working tree has uncommitted changes "
        "(files ignored by .gitignore do not count)",
    )
    return parser


def clean_tree_problem(repo: Path | None = None) -> str | None:
    """Why the tree is not clean (or cannot be checked); None when it is clean."""
    changes = working_tree_changes(repo or REPO_ROOT)
    if changes is None:
        return "cannot read the git state (is git installed, is this a repository?)"
    if not changes:
        return None
    shown = "\n  ".join(changes[:10]) + ("\n  ..." if len(changes) > 10 else "")
    return (
        f"the working tree has {len(changes)} uncommitted change(s); commit or stash them "
        f"(or drop --require-clean):\n  {shown}"
    )


def schema_hash_config(hashes: Sequence[str]) -> dict[str, object]:
    """The batch-config entry: one hash, or None plus every hash seen if runs disagreed."""
    if len(hashes) == 1:
        return {"tool_schemas_sha256": hashes[0]}
    return {"tool_schemas_sha256": None, "tool_schemas_sha256_all": list(hashes)}


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, warm up, run the batch, store every record and print a summary."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]

    if args.require_clean:  # before anything else: no Ollama call, no batch row
        problem = clean_tree_problem()
        if problem is not None:
            print(f"error: --require-clean: {problem}", file=sys.stderr)
            return 2

    suite_path = resolve_suite_path(args.suite)
    try:
        suite = load_suite(suite_path)
        case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None
        cases = select_cases(suite, case_ids)
        config = EvalConfig(
            model=args.model,
            think=args.think == "on",
            prompt_variant=args.prompt_variant,
            num_ctx=args.num_ctx,
            temperature=args.temperature,
            seed=args.seed,
            repetitions=args.repeat,
            label=args.label,
            tool_backend=args.tools,
        )
    except (SuiteError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with (
        OllamaClient.from_env(model=args.model, host=args.host) as llm,
        ResultStore(args.db) as store,
    ):
        print(f"warm-up: loading {config.model} (untimed, excluded from metrics) ...", flush=True)
        warmup = warm_up(llm, config)
        if not warmup.ok:
            print(f"error: warm-up failed, nothing measured: {warmup.error}", file=sys.stderr)
            return 2
        print(f"warm-up: {warmup.latency_ms:.0f} ms (model load {warmup.load_ms:.0f} ms)")

        info = read_ollama_info(getattr(llm, "host", None), config.model)
        print(
            f"ollama {info.version or 'unknown'}, model digest "
            f"{(info.model_digest or 'unknown')[:12]}"
        )
        batch_config = {
            **config.to_json(),
            "suite": _display_path(suite_path),
            **suite_hashes(suite_path),
            "cases": [case.id for case in cases],
            "ollama_version": info.version,
            "model_digest": info.model_digest,
            # Known once a run has called the LLM: filled in by on_record below.
            "tool_schemas_sha256": None,
        }
        batch_id = store.create_batch(
            config.label, batch_config, git_commit=git_commit(), warmup=asdict(warmup)
        )
        total = len(cases) * config.repetitions
        done = 0
        schema_hashes: list[str] = []

        def on_record(record: RunRecord) -> None:
            nonlocal done
            store.save_run(batch_id, record)
            digest = record.tool_schemas_sha256
            if digest is not None and digest not in schema_hashes:
                schema_hashes.append(digest)
                store.update_batch_config(batch_id, schema_hash_config(schema_hashes))
            done += 1
            print(format_progress(done, total, record), flush=True)

        records = run_batch(cases, config, llm=llm, on_record=on_record)
        batch = store.latest_batch(config.label)

    assert batch is not None and batch.id == batch_id
    summary = summarize(batch, records)
    print()
    print(
        f"batch {batch_id} '{config.label}': {summary.total.fraction} pass, "
        f"{summary.total.fraction_with_heuristics} with heuristics, "
        f"excluded: {summary.total.excluded}"
    )
    for name, count in summary.categories.items():
        print(
            f"  {name:<12} {count.fraction:>5}  (with heuristics {count.fraction_with_heuristics})"
        )
    print(f"report: uv run python -m evals.report --label {config.label}")
    if summary.too_many_inconclusive:
        total_count = summary.total
        print(
            f"warning: {total_count.fixture_miss}/{total_count.runs} runs are inconclusive "
            f"(fixture_miss), more than {INCONCLUSIVE_LIMIT:.0%}: the batch is stored, but "
            "fix the cases' fixtures (or `default_search`) before trusting it. "
            "Details: the report's 'Inconclusive runs' section.",
            file=sys.stderr,
        )
        return EXIT_TOO_MANY_INCONCLUSIVE
    return 0


def format_progress(done: int, total: int, record: RunRecord) -> str:
    """One compact line per run."""
    width = len(str(total))
    trace = record.trace
    head = f"[{done:>{width}}/{total}] {record.case_id:<24} rep {record.repetition}"
    if record.harness_error is not None:
        return f"{head}  ERROR  {record.harness_error}"
    status = "MISS " if record.fixture_misses else "PASS " if record.passed else "FAIL "
    line = (
        f"{head}  {status} stop={trace.stop_reason} it={trace.iterations} "
        f"tools={len(trace.tool_calls)} err={trace.tool_errors} {record.wall_time_ms / 1000:.1f}s"
    )
    if record.mcp_startup_ms is not None:
        line += f" (mcp startup {record.mcp_startup_ms / 1000:.1f}s)"
    failed = [c.name.split("(")[0] for c in record.checks if not c.passed and not c.heuristic]
    heuristic = [c.name.split("(")[0] for c in record.checks if not c.passed and c.heuristic]
    if failed:
        line += f"  failed: {', '.join(failed)}"
    if heuristic:
        line += f"  heuristic failed: {', '.join(heuristic)}"
    return line


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return str(path)


def git_commit() -> str | None:
    """Short HEAD hash, with "+dirty" when the working tree has changes; None if unknown."""
    return provenance.git_commit(REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
