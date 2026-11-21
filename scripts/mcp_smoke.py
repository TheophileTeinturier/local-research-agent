"""Smoke/debug check of the MCP server, through the real bridge. No network, no LLM.

    uv run python scripts/mcp_smoke.py

Starts `local_research_agent.mcp.server` as a subprocess serving `offline_registry()` below
(fake arXiv from the eval fixtures, frozen clock), prints the tool list, calls each tool once
with simple arguments, then closes the bridge and checks the server process is gone.
Exit code 0 when every call succeeded.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_research_agent.tools import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# The calls to make: simple arguments, all covered by the fixtures below.
CALLS: list[tuple[str, dict[str, object]]] = [
    ("calculator", {"expression": "(17 * 23 + 4) / 3"}),
    ("current_date", {}),
    ("search_arxiv", {"query": '"graph neural network" protein', "max_results": 2}),
    ("get_paper_details", {"arxiv_id": "1706.02216"}),
]


def offline_registry() -> ToolRegistry:
    """Registry factory, run in the SERVER process (`--registry-factory
    mcp_smoke:offline_registry`)."""
    sys.path.insert(0, str(REPO_ROOT))  # for `evals`
    from evals.arxiv_fake import CaseArxivTransport
    from evals.suite import ArxivSpec
    from local_research_agent.tools import build_registry

    spec = ArxivSpec(search="search_gnn_ppi_10.xml", papers={"1706.02216": "paper_1706.02216.xml"})
    frozen = datetime.datetime(2026, 11, 9, 10, 18, 18)
    return build_registry(arxiv_transport=CaseArxivTransport(spec), clock=lambda: frozen)


def main() -> int:
    """Start the server, list and call every tool once, then check the process is gone."""
    from local_research_agent.mcp.client import McpClientBridge, server_parameters
    from local_research_agent.mcp.provider import from_call_tool_result

    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    params = server_parameters(
        registry_factory="mcp_smoke:offline_registry",
        extra_env={"PYTHONPATH": os.pathsep.join([str(SCRIPTS_DIR), str(REPO_ROOT)])},
        cwd=REPO_ROOT,
    )
    failures = 0
    with McpClientBridge(params) as bridge:
        print(f"server started: pid {bridge.server_pid}, startup {bridge.startup_ms:.0f} ms")
        print("tools:")
        for tool in bridge.list_tools():
            required = tool.input_schema.get("required", [])
            print(f"  - {tool.name} (required: {', '.join(required) or 'none'})")
        print()
        for name, arguments in CALLS:
            result = from_call_tool_result(bridge.call_tool(name, dict(arguments), timeout_s=10))
            failures += not result.ok
            first_line = result.content.splitlines()[0] if result.content else ""
            status = "ok" if result.ok else f"ERROR ({result.metadata.get('error_type')})"
            print(f"{name} {arguments} -> {status}: {first_line[:100]}")
        pid = bridge.server_pid
    print()
    print(f"bridge closed; server pid {pid} {'still ALIVE' if _alive(pid) else 'is gone'}")
    return 1 if failures or _alive(pid) else 0


def _alive(pid: int | None) -> bool:
    """Whether a process with this PID still runs (Windows: OpenProcess; POSIX: kill 0)."""
    if pid is None:
        return False
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


if __name__ == "__main__":
    raise SystemExit(main())
