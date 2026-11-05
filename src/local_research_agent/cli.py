"""Command-line entry point.

    python -m local_research_agent.cli "What is 17 * 23 + 4?"

stdout gets only the answer and the trace; JSON logs and errors go to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from .agent import LLMStep, RunResult, Step, run_once
from .llm import ChatOptions, LLMError, OllamaClient
from .observability import configure_logging
from .tools import build_default_registry

_PREVIEW_CHARS = 120


def build_parser() -> argparse.ArgumentParser:
    defaults = ChatOptions()
    parser = argparse.ArgumentParser(
        prog="local_research_agent",
        description="Ask the local agent a question (one tool-calling round trip).",
    )
    parser.add_argument("question")
    parser.add_argument("--model", help="Ollama model (default: $OLLAMA_MODEL or qwen3:8b)")
    parser.add_argument("--host", help="Ollama URL (default: $OLLAMA_HOST or 127.0.0.1:11434)")
    parser.add_argument("--think", action="store_true", help="enable Qwen3 reasoning (slower)")
    parser.add_argument("--num-ctx", type=int, default=defaults.num_ctx)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    # Windows pipes default to a legacy code page; never crash on printing the model's output.
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]

    options = ChatOptions(temperature=args.temperature, num_ctx=args.num_ctx, think=args.think)
    registry = build_default_registry()
    with OllamaClient.from_env(model=args.model, host=args.host) as llm:
        try:
            result = run_once(llm, registry, args.question, options=options)
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(format_result(result))
    return 0


def format_result(result: RunResult) -> str:
    lines = [result.answer.strip() or "(empty answer)", "", "trace:"]
    lines += [f"  {i}. {_format_step(step)}" for i, step in enumerate(result.trace, start=1)]
    if not result.completed:
        lines.append("  ! the model requested more tool calls; run_once does not execute them")
    return "\n".join(lines)


def _format_step(step: Step) -> str:
    if isinstance(step, LLMStep):
        called = ", ".join(call.name for call in step.tool_calls)
        action = f"tool_calls: {called}" if called else "answer"
        m = step.metrics
        thought = f" | thinking {len(step.thinking)} chars" if step.thinking else ""
        return (
            f"llm  -> {action} | tokens in={m.prompt_eval_count} out={m.eval_count} "
            f"| {step.latency_ms:.0f} ms{thought}"
        )
    args = step.call.arguments
    args_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    status = "ok" if step.result.ok else "ERROR"
    return (
        f"tool {step.call.name} {args_text} -> {status}: {_preview(step.result.content)} "
        f"| {step.result.metadata.get('duration_ms', 0):.2f} ms"
    )


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_CHARS else text[: _PREVIEW_CHARS - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
