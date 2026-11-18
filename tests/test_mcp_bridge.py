"""The real thing: a server SUBPROCESS behind the bridge. No network (fake arXiv), no Ollama.

Each server start costs about a second, so servers are shared where tests do not disturb
them (module-scoped fixtures), and fresh ones are used for lifecycle and failure tests.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

import pytest

from fakes import FakeLLM, text_response, tool_call_response
from local_research_agent.agent import AgentConfig, AgentLimits, StopReason, run_agent
from local_research_agent.llm import ChatOptions, LLMResponse, Message, ToolSchema
from local_research_agent.mcp.client import (
    McpClientBridge,
    McpError,
    McpServerDied,
    McpStartupError,
)
from local_research_agent.mcp.provider import McpToolProvider
from local_research_agent.tools import ToolRegistry, build_default_registry
from mcp_helpers import FROZEN_NOW, make_bridge, parity_registry, pid_alive, wait_until_dead

pytestmark = pytest.mark.mcp


@pytest.fixture(scope="module")
def errlog(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TextIO]:
    """The servers' stderr (their JSON logs) goes to a file, read when a test fails."""
    path = tmp_path_factory.mktemp("mcp") / "server-stderr.log"
    with path.open("w", encoding="utf-8") as handle:
        yield handle


@pytest.fixture(scope="module")
def parity_bridge(errlog: TextIO) -> Iterator[McpClientBridge]:
    bridge = make_bridge(errlog=errlog)
    bridge.start()
    yield bridge
    bridge.close()


def mcp_registry(bridge: McpClientBridge, **kwargs: Any) -> ToolRegistry:
    return ToolRegistry(providers=[McpToolProvider(bridge, **kwargs)])


# --- contract and parity ------------------------------------------------------------------------


def test_contract_listed_schemas_are_the_local_schemas(errlog: TextIO) -> None:
    """The PRODUCTION server (default factory) lists exactly what the LLM sees locally."""
    with make_bridge(None, errlog=errlog) as bridge:
        remote = McpToolProvider(bridge).schemas()
    local = build_default_registry().schemas()
    assert remote == local
    # The same, as the JSON actually sent to the LLM (key order included).
    assert json.dumps(remote) == json.dumps(local)


PARITY_CALLS: list[tuple[str, Any]] = [
    ("calculator", {"expression": "17 * 23 + 4"}),
    ("calculator", '{"expression": "2 ** 10"}'),  # raw JSON text, as some models send it
    ("calculator", {"expression": "1 / 0"}),  # tool error
    ("calculator", {"expression": 12}),  # invalid arguments (wrong type)
    ("calculator", {"expression": "1", "unit": "kg"}),  # invalid arguments (extra field)
    ("calculator", '{"expression": '),  # malformed JSON
    ("calculator", "[1, 2]"),  # not an object
    ("current_date", {}),  # frozen clock
    (
        "search_arxiv",
        {
            "query": "graph neural networks protein-protein interaction prediction",
            "sort": "date",
            "max_results": 5,
        },
    ),
    ("search_arxiv", {"query": "qwxzvkjhq zzqplmnb"}),  # zero hits
    ("get_paper_details", {"arxiv_id": "2606.01781"}),
    ("get_paper_details", {"arxiv_id": "2301.99999"}),  # not found
    ("get_paper_details", {"arxiv_id": "not an id"}),  # invalid id
    ("no_such_tool", {}),  # unknown tool
    ("echo", {"text": "ab", "times": 5000}),  # oversized output (10000 > 4000 chars)
    ("boom", {"text": "x"}),  # the tool raises
]


@pytest.mark.parametrize(("name", "arguments"), PARITY_CALLS)
def test_parity_local_vs_mcp(parity_bridge: McpClientBridge, name: str, arguments: Any) -> None:
    local = parity_registry().execute(name, arguments)
    remote = mcp_registry(parity_bridge).execute(name, arguments)

    assert (remote.ok, remote.content) == (local.ok, local.content)
    for key in ("error_type", "truncated", "original_length"):
        assert remote.metadata.get(key) == local.metadata.get(key), key


def test_frozen_clock_lives_in_the_server(parity_bridge: McpClientBridge) -> None:
    result = mcp_registry(parity_bridge).execute("current_date", {})
    assert result.content == FROZEN_NOW.isoformat(timespec="seconds")


# --- lifecycle --------------------------------------------------------------------------------


def test_start_and_close_are_idempotent_and_leave_no_process(errlog: TextIO) -> None:
    threads_before = set(threading.enumerate())
    bridge = make_bridge(errlog=errlog)
    bridge.start()
    bridge.start()  # no second server
    pid = bridge.server_pid
    assert pid is not None and pid != os.getpid() and pid_alive(pid)
    assert bridge.startup_ms is not None and bridge.startup_ms > 0

    bridge.close()
    bridge.close()
    assert wait_until_dead(pid), f"server {pid} still alive after close()"
    leftover = [
        t for t in set(threading.enumerate()) - threads_before if t.name == "mcp-bridge-loop"
    ]
    assert leftover == [], "the bridge's event-loop thread was not joined"
    with pytest.raises(McpError):
        bridge.call_tool("calculator", {"expression": "1"})
    with pytest.raises(McpStartupError):
        bridge.start()  # a closed bridge is not restarted


def test_startup_failure_bad_command() -> None:
    bridge = make_bridge(command="definitely-not-a-command-4f9c2")
    started = time.monotonic()
    with pytest.raises(McpStartupError):
        bridge.start(startup_timeout_s=5)
    assert time.monotonic() - started < 5


def test_startup_failure_broken_factory(errlog: TextIO) -> None:
    with pytest.raises(McpStartupError, match="MCP server failed to start"):
        make_bridge("mcp_helpers:broken_registry", errlog=errlog).start(startup_timeout_s=20)


def test_startup_timeout_kills_a_silent_server(errlog: TextIO, tmp_path: Path) -> None:
    """A process that never answers the handshake: timeout, then no process left."""
    pid_file = tmp_path / "pid"
    script = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    bridge = make_bridge(command=sys.executable, args=["-c", script], errlog=errlog)
    started = time.monotonic()
    with pytest.raises(McpStartupError, match="TimeoutError"):
        bridge.start(startup_timeout_s=1.5)
    assert time.monotonic() - started < 10  # timeout + bounded SDK shutdown
    assert wait_until_dead(int(pid_file.read_text()))


# --- failures during a run ----------------------------------------------------------------------


def test_call_timeout_is_an_observation_and_the_session_survives(errlog: TextIO) -> None:
    with make_bridge(errlog=errlog) as bridge:
        registry = mcp_registry(bridge, call_timeout_s=0.5)
        started = time.monotonic()
        result = registry.execute("slow", {"seconds": 3})
        assert time.monotonic() - started < 2.5
        assert not result.ok and result.metadata["error_type"] == "timeout"
        assert result.content == "Error: Tool 'slow' timed out after 0.5 s."
        # The server is still there (its tool thread finishes in the background).
        assert registry.execute("calculator", {"expression": "6 * 7"}).content == "6 * 7 = 42"
        assert bridge.broken is None


class KillingLLM:
    """Wraps FakeLLM: on its `kill_on`-th call it schedules a kill of the server, so the
    server dies WHILE the next tool call (a slow one) is running."""

    def __init__(self, fake: FakeLLM, bridge: McpClientBridge, kill_on: int) -> None:
        self.fake, self.bridge, self.kill_on = fake, bridge, kill_on

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        options: ChatOptions | None = None,
    ) -> LLMResponse:
        if len(self.fake.calls) + 1 == self.kill_on:
            pid = self.bridge.server_pid
            assert pid is not None
            threading.Timer(0.5, os.kill, (pid, signal.SIGTERM)).start()
        return self.fake.chat(messages, tools, options)


def test_server_killed_mid_run_the_loop_ends_cleanly(errlog: TextIO) -> None:
    script = [
        tool_call_response(("calculator", {"expression": "1 + 1"})),
        tool_call_response(("slow", {"seconds": 5})),  # the server dies during this call
        tool_call_response(("calculator", {"expression": "2 + 2"})),  # must fail fast
        text_response("Tools became unavailable; 1 + 1 = 2."),
    ]
    with make_bridge(errlog=errlog) as bridge:
        pid = bridge.server_pid
        assert pid is not None
        llm = KillingLLM(FakeLLM(script), bridge, kill_on=2)
        config = AgentConfig(limits=AgentLimits(tool_timeout_s=10))
        result = run_agent(llm, mcp_registry(bridge, call_timeout_s=9), config, "q")

        calls = result.state.tool_calls
        assert [c.ok for c in calls] == [True, False, False]
        assert calls[1].error_type == calls[2].error_type == "mcp_server_died"
        assert calls[1].duration_ms < 5000  # noticed the death, did not wait for the tool
        assert calls[2].duration_ms < 100  # fail fast: no request was even sent
        assert "no tool can be used" in calls[2].content
        assert result.stop_reason is StopReason.FINAL_ANSWER
        with pytest.raises(McpServerDied):
            bridge.call_tool("calculator", {"expression": "3"})
    assert wait_until_dead(pid)


# --- agent level --------------------------------------------------------------------------------


def test_agent_two_step_run_through_mcp(parity_bridge: McpClientBridge) -> None:
    llm = FakeLLM(
        [
            tool_call_response(("current_date", {})),
            tool_call_response(("calculator", {"expression": "25 + 21"})),
            text_response("There are 46 days."),
        ]
    )
    registry = mcp_registry(parity_bridge)
    result = run_agent(llm, registry, AgentConfig(), "Days until 2026-12-25?")

    assert result.stop_reason is StopReason.FINAL_ANSWER
    assert [c.content for c in result.state.tool_calls] == ["2026-11-09T10:18:18", "25 + 21 = 46"]
    # The LLM saw the same tool definitions as with local tools.
    assert llm.calls[0].tools == parity_registry().schemas()
