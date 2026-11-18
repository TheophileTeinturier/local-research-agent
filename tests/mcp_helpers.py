"""Helpers for the MCP tests: registry factories the SERVER subprocess imports, and utilities.

The factories are loaded by `python -m local_research_agent.mcp.server --registry-factory
mcp_helpers:<name>`, with PYTHONPATH pointing at this directory. They are also called
in-process by the parity tests, so the local and the MCP side serve the SAME registry.
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field

from fakes import ARXIV_FIXTURES
from local_research_agent.mcp.client import McpClientBridge, server_parameters
from local_research_agent.tools import Tool, ToolRegistry, build_default_registry
from local_research_agent.tools.arxiv.testing import FakeArxivTransport, offline_arxiv_client

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FROZEN_NOW = datetime.datetime(2026, 11, 9, 10, 18, 18)
SLOW_TOOL_SECONDS = 3.0


# --- test-only tools ------------------------------------------------------------------------


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(description="Text to repeat.")
    times: int = Field(default=1, ge=1, le=10_000)


class SleepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: float = Field(ge=0, le=60)


def _echo(args: EchoArgs) -> str:
    return args.text * args.times


def _boom(args: EchoArgs) -> str:
    raise RuntimeError("disk on fire")


def _sleep(args: SleepArgs) -> str:
    time.sleep(args.seconds)
    return f"slept {args.seconds:g} s"


TEST_TOOLS = [
    Tool("echo", "Repeat text (for output-size tests).", EchoArgs, _echo),
    Tool("boom", "Always raises.", EchoArgs, _boom),
    Tool("slow", "Sleep, then answer (for timeout tests).", SleepArgs, _sleep),
]


# --- factories (zero-argument, as the server requires) -----------------------------------------


def parity_registry() -> ToolRegistry:
    """The four real tools on recorded arXiv data and a frozen clock, plus the test tools."""
    client = offline_arxiv_client(FakeArxivTransport.from_directory(ARXIV_FIXTURES))
    registry = build_default_registry(arxiv_client=client, clock=lambda: FROZEN_NOW)
    for tool in TEST_TOOLS:
        registry.register(tool)
    return registry


def broken_registry() -> ToolRegistry:
    raise RuntimeError("factory exploded")


# --- utilities --------------------------------------------------------------------------------


def make_bridge(
    registry_factory: str | None = "mcp_helpers:parity_registry",
    *,
    errlog: TextIO | None = None,
    command: str | None = None,
    args: list[str] | None = None,
) -> McpClientBridge:
    """A bridge to a server serving `registry_factory` (None: the production default)."""
    params = server_parameters(
        registry_factory=registry_factory,
        extra_env={"PYTHONPATH": os.pathsep.join([str(TESTS_DIR), str(REPO_ROOT)])},
        cwd=REPO_ROOT,
    )
    if command is not None:
        params = params.model_copy(update={"command": command, "args": args or []})
    return McpClientBridge(params, errlog=errlog)


def pid_alive(pid: int) -> bool:
    """Whether a process with this PID is running (Windows: OpenProcess; POSIX: kill 0)."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_until_dead(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)
