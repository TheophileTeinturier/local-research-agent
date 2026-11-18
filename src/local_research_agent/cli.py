"""Command-line entry point.

    python -m local_research_agent.cli "What is 17 * 23 + 4?" [--tools local|mcp]

`--tools mcp` runs the same tools in an MCP server subprocess (stdio) instead of in-process;
the agent loop is the same. stdout gets only the answer and the trace; JSON logs go to stderr.
Exit code: 0 when the model answered, 1 for any other stop reason, 2 when the MCP tool
server could not be started (an infrastructure error: the agent never ran).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from .agent import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
    AgentConfig,
    AgentLimits,
    AgentResult,
    StopReason,
    get_system_prompt,
    run_agent,
)
from .agent.state import LLMCallRecord, ToolCallRecord
from .llm import ChatOptions, OllamaClient
from .observability import configure_logging
from .tools import ArxivClient, ToolRegistry, build_default_registry

_PREVIEW_CHARS = 120

EXIT_MCP_STARTUP_FAILED = 2


@dataclass(frozen=True)
class ToolBackend:
    """Which backend served the tools, for the trace."""

    name: str  # "local" | "mcp"
    mcp_startup_ms: float | None = None

    def describe(self) -> str:
        if self.mcp_startup_ms is None:
            return f"tools {self.name}"
        return f"tools {self.name} (server startup {self.mcp_startup_ms:.0f} ms)"


STOP_EXPLANATIONS: dict[StopReason, str] = {
    StopReason.FINAL_ANSWER: "the model answered",
    StopReason.MAX_ITERATIONS: "the model still wanted tools after the maximum number of steps",
    StopReason.TIMEOUT: "the total time budget was exhausted",
    StopReason.REPETITION: "the model kept repeating the same tool call",
    StopReason.LLM_ERROR: "the LLM backend failed",
    StopReason.TOO_MANY_TOOL_ERRORS: "too many tool calls failed in a row",
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments; defaults come from `ChatOptions` and `AgentLimits`."""
    options, limits = ChatOptions(), AgentLimits()
    parser = argparse.ArgumentParser(
        prog="local_research_agent",
        description="Ask the local agent a question (tool-calling loop with guardrails).",
    )
    parser.add_argument("question")
    parser.add_argument("--model", help="Ollama model (default: $OLLAMA_MODEL or qwen3:8b)")
    parser.add_argument("--host", help="Ollama URL (default: $OLLAMA_HOST or 127.0.0.1:11434)")
    parser.add_argument("--think", action="store_true", help="enable Qwen3 reasoning (slower)")
    parser.add_argument("--num-ctx", type=int, default=options.num_ctx)
    parser.add_argument("--temperature", type=float, default=options.temperature)
    parser.add_argument(
        "--prompt-variant",
        default=DEFAULT_PROMPT_VARIANT,
        choices=list(PROMPT_VARIANTS),
        help="named system prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--tools",
        choices=["local", "mcp"],
        default="local",
        help="run the tools in-process, or in an MCP server subprocess (default: %(default)s)",
    )
    parser.add_argument("--max-iterations", type=int, default=limits.max_iterations)
    parser.add_argument(
        "--timeout", type=float, default=limits.total_timeout_s, help="total budget, seconds"
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the agent once and print the answer (or the stop reason) and a compact trace."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    # Windows pipes default to a legacy code page; never crash on printing the model's output.
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]

    config = AgentConfig(
        system_prompt=get_system_prompt(args.prompt_variant),
        limits=AgentLimits(max_iterations=args.max_iterations, total_timeout_s=args.timeout),
        llm_options=ChatOptions(
            temperature=args.temperature, num_ctx=args.num_ctx, think=args.think
        ),
    )
    # ExitStack: whatever happens (an exception included), the arXiv client is closed and the
    # MCP bridge is closed, which terminates the server subprocess.
    with contextlib.ExitStack() as stack:
        if args.tools == "mcp":
            # Imported only in MCP mode: local runs never load the MCP SDK.
            from .mcp.client import McpStartupError

            try:
                registry, backend = _mcp_tools(stack, config.limits.tool_timeout_s, args.log_level)
            except McpStartupError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_MCP_STARTUP_FAILED
        else:
            arxiv = stack.enter_context(ArxivClient())
            registry, backend = build_default_registry(arxiv_client=arxiv), ToolBackend("local")
        llm = stack.enter_context(OllamaClient.from_env(model=args.model, host=args.host))
        result = run_agent(llm, registry, config, args.question)

    print(format_result(result, backend))
    return 0 if result.stop_reason is StopReason.FINAL_ANSWER else 1


def _mcp_tools(
    stack: contextlib.ExitStack, tool_timeout_s: float, log_level: str
) -> tuple[ToolRegistry, ToolBackend]:
    """Start the MCP server (before the agent loop) and wrap it in a registry.

    Raises `McpStartupError`. The bridge is registered on `stack` BEFORE it starts, so even
    a failed start leaves no process behind.
    """
    from .mcp.client import McpClientBridge, server_parameters
    from .mcp.provider import McpToolProvider, call_timeout_for

    bridge = McpClientBridge(server_parameters(log_level=log_level))
    stack.callback(bridge.close)
    bridge.start()
    provider = McpToolProvider(bridge, call_timeout_s=call_timeout_for(tool_timeout_s))
    return ToolRegistry(providers=[provider]), ToolBackend("mcp", bridge.startup_ms)


def format_result(result: AgentResult, backend: ToolBackend | None = None) -> str:
    """Answer (or explanation), then a one-line summary and one line per LLM/tool call."""
    if result.final_answer is not None:
        head = result.final_answer.strip() or "(empty answer)"
    else:
        head = f"No answer: {STOP_EXPLANATIONS[result.stop_reason]} ({result.stop_reason.value})."
        if result.state.error:
            head += f"\nerror: {result.state.error}"

    m = result.metrics
    lines = [
        head,
        "",
        f"trace: stop_reason={result.stop_reason.value} | iterations={m.iterations} "
        f"| tokens in={m.prompt_tokens} out={m.completion_tokens} "
        f"| llm {m.llm_latency_ms:.0f} ms (model load {m.llm_load_ms:.0f} ms) "
        f"| wall {m.wall_time_ms:.0f} ms" + ("" if backend is None else f" | {backend.describe()}"),
    ]
    for llm_call in result.state.llm_calls:
        lines.append(f"  {llm_call.iteration}. {_format_llm_call(llm_call)}")
        lines += [
            f"       {_format_tool_call(call)}"
            for call in result.state.tool_calls
            if call.iteration == llm_call.iteration
        ]
    return "\n".join(lines)


def _format_llm_call(call: LLMCallRecord) -> str:
    action = f"tool_calls: {', '.join(call.tool_calls)}" if call.tool_calls else "answer"
    return (
        f"llm  -> {action} | tokens in={call.prompt_tokens} out={call.completion_tokens} "
        f"| {call.latency_ms:.0f} ms"
    )


def _format_tool_call(call: ToolCallRecord) -> str:
    status = "REPEATED" if call.repeated else ("ok" if call.ok else "ERROR")
    return (
        f"tool {call.name} {call.arguments} -> {status}: {_preview(call.content)} "
        f"| {call.duration_ms:.2f} ms"
    )


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_CHARS else text[: _PREVIEW_CHARS - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
