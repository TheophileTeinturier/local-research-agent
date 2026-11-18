"""`McpClientBridge`: a SYNCHRONOUS facade over the async MCP client SDK (stdio transport).

Why a bridge: `run_agent` is synchronous and must not change, while the SDK is async
(anyio). The bridge owns a background thread running its own asyncio event loop; the agent's
threads submit coroutines to that loop and block on the result.

    agent thread (sync)                      bridge thread (event loop)
    -------------------                      ---------------------------------------------
    bridge.start()  ---- submit ---------->  _session_main():   [ONE long-lived task]
       wait(ready)                             async with stdio_client(...)  -> spawns server
                                               async with ClientSession(...)
                                                 initialize(); list_tools()
       <------------- ready.set() -----------    wait for the shutdown event ...
    bridge.call_tool() -- submit -------->   session.call_tool()  [a SEPARATE short task]
       future.result(timeout) <-- result --
    bridge.close()  ---- shutdown.set() -->      ... exits ClientSession, then stdio_client
       wait(finished), stop loop, join          (closes stdin, waits, kills the process tree)

Why ONE task enters and exits the contexts: `stdio_client` and `ClientSession` open anyio
task groups, i.e. cancel scopes. anyio requires a cancel scope to be exited by the very task
that entered it (scopes form a per-task stack); exiting from another task raises
"Attempted to exit cancel scope in a different task" and can leave the subprocess running.
So `_session_main` enters both contexts, parks on an event, and exits them itself.
Requests (`call_tool`) are separate tasks: that is fine, they only send and await messages;
they never enter or exit a scope owned by the session.

The session is PERSISTENT: one subprocess for the whole agent run, not one per call (a cold
start costs ~1 s of interpreter start-up, imports and the MCP handshake).

Failure policy (every case is a typed exception here; `McpToolProvider` turns those raised
during a call into `ToolResult(ok=False)` observations):
- startup failure (bad command, server crash, no handshake within `startup_timeout_s`):
  `McpStartupError` from `start()`, raised before the agent loop starts;
- call timeout: `McpTimeout`. The SDK sends `notifications/cancelled`; the server stops
  waiting, but the tool thread inside the server cannot be killed and runs to completion;
- server death (pipe closed): `McpServerDied`, and the bridge is marked BROKEN: every later
  call fails fast with the same error. No automatic restart in this version;
- anything else the SDK raises (a JSON-RPC error, a malformed result): `McpProtocolError`.

Subprocess environment: the SDK does NOT pass our environment through. It builds the child's
environment from an allow-list (`mcp.client.stdio.DEFAULT_INHERITED_ENV_VARS`: on Windows
APPDATA, HOMEDRIVE, HOMEPATH, LOCALAPPDATA, PATH, PATHEXT, PROCESSOR_ARCHITECTURE,
SYSTEMDRIVE, SYSTEMROOT, TEMP, USERNAME, USERPROFILE; on POSIX HOME, LOGNAME, PATH, SHELL,
TERM, USER) and MERGES `StdioServerParameters.env` over it. `server_parameters` adds only what
the server needs: `extra_env` (e.g. PYTHONPATH and the eval harness's case file).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import sys
import threading
import time
from collections.abc import Coroutine, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO, TypeVar

import anyio
import mcp.types as types
from mcp import ClientSession, MCPError, StdioServerParameters, stdio_client
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT

from ..observability import log_event
from .server import SERVER_PID_META_KEY

logger = logging.getLogger(__name__)

T = TypeVar("T")

SERVER_MODULE = "local_research_agent.mcp.server"
DEFAULT_STARTUP_TIMEOUT_S = 30.0
DEFAULT_CALL_TIMEOUT_S = 30.0
# Extra wait on the thread side, beyond the SDK's own request timeout: the SDK raises its
# REQUEST_TIMEOUT first; this backstop only fires if the event loop itself is stuck.
_BACKSTOP_S = 2.0
# close(): the SDK's shutdown is bounded (2 s grace after closing stdin, then a hard kill of
# the process tree, 2 s to reap), so this is only reached if something is badly wrong.
_CLOSE_TIMEOUT_S = 10.0
# Raised when writing to / reading from a server whose pipes are closed.
_PIPE_ERRORS = (anyio.BrokenResourceError, anyio.ClosedResourceError, BrokenPipeError)


class McpError(RuntimeError):
    """Base class of the bridge's errors."""


class McpStartupError(McpError):
    """The server could not be started or did not complete the MCP handshake in time.
    An infrastructure error: raised by `start()`, before the agent loop runs."""


class McpTimeout(McpError):
    """A request did not complete in time. The server may still be running the tool."""


class McpServerDied(McpError):
    """The server process is gone (pipe closed). The bridge is broken from then on."""


class McpProtocolError(McpError):
    """The server answered with a JSON-RPC error or something the SDK could not parse."""


def server_parameters(
    *,
    registry_factory: str | None = None,
    log_level: str = "WARNING",
    extra_env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> StdioServerParameters:
    """How to start our server: `[sys.executable, -m, local_research_agent.mcp.server, ...]`.

    `sys.executable`, not `uv run`: one process layer fewer (uv would add its own process,
    and on Windows that extra parent complicates termination). The interpreter is the one
    running this code, so the server runs with the same installed packages.
    """
    args = ["-m", SERVER_MODULE, "--log-level", log_level]
    if registry_factory is not None:
        args += ["--registry-factory", registry_factory]
    return StdioServerParameters(
        command=sys.executable,
        args=args,
        env=dict(extra_env or {}),
        cwd=None if cwd is None else str(cwd),
    )


class McpClientBridge:
    """One MCP stdio session, persistent for the bridge's lifetime, behind a blocking API.

    Thread-safe: any thread may call `call_tool` (the agent loop calls it from its worker
    threads). Use as a context manager, or call `close()` (idempotent) in a `finally`.
    """

    def __init__(self, params: StdioServerParameters, *, errlog: TextIO | None = None) -> None:
        self._params = params
        # Where the server's stderr (its JSON logs) goes. Must be a real file (it is handed
        # to the child process); pytest's capture objects are not, hence the fallback.
        self._errlog = errlog if errlog is not None else _real_stderr()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._shutdown: asyncio.Event | None = None
        self._ready = threading.Event()  # handshake done, or startup failed
        self._finished = threading.Event()  # _session_main has fully exited its contexts
        self._startup_error: BaseException | None = None
        self._tools: list[types.Tool] = []
        self._broken: str | None = None
        self._lock = threading.Lock()
        self._closed = False
        self.startup_ms: float | None = None
        self.server_pid: int | None = None

    # --- lifecycle ----------------------------------------------------------------------------

    def start(self, startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S) -> None:
        """Spawn the server, initialize the session and list the tools. Idempotent.

        Raises `McpStartupError` (after cleaning up: no process, no thread left behind) if
        that does not succeed within `startup_timeout_s`.
        """
        with self._lock:
            if self._closed:
                raise McpStartupError("this bridge was closed; create a new one")
            if self._thread is not None:
                return
            started = time.perf_counter()
            self._loop = asyncio.new_event_loop()
            self._shutdown = asyncio.Event()
            self._thread = threading.Thread(
                target=self._run_loop, name="mcp-bridge-loop", daemon=True
            )
            self._thread.start()
            asyncio.run_coroutine_threadsafe(self._session_main(startup_timeout_s), self._loop)

        # The timeout is enforced INSIDE the loop (anyio.fail_after, which cleans up properly);
        # this wait only needs a margin on top of it.
        if not self._ready.wait(startup_timeout_s + _BACKSTOP_S):
            self._startup_error = TimeoutError(f"no handshake within {startup_timeout_s:g} s")
        if self._startup_error is not None or self._session is None:
            error = self._startup_error
            self.close()
            raise McpStartupError(f"MCP server failed to start: {_describe(error)}") from error

        self.startup_ms = round((time.perf_counter() - started) * 1000, 1)
        log_event(
            logger,
            "mcp_bridge_started",
            startup_ms=self.startup_ms,
            server_pid=self.server_pid,
            tools=[tool.name for tool in self._tools],
        )

    def close(self) -> None:
        """End the session, terminate the server process and join the thread. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread, shutdown = self._loop, self._thread, self._shutdown
        if loop is None or thread is None or shutdown is None:
            return  # never started
        # Wake _session_main, which then exits its contexts in its own task (see module doc).
        loop.call_soon_threadsafe(shutdown.set)
        if not self._finished.wait(_CLOSE_TIMEOUT_S):
            log_event(logger, "mcp_bridge_close_timeout", level=logging.WARNING)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(_CLOSE_TIMEOUT_S)
        self._session = None
        log_event(logger, "mcp_bridge_closed", server_pid=self.server_pid)

    def __enter__(self) -> McpClientBridge:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- requests -----------------------------------------------------------------------------

    @property
    def broken(self) -> str | None:
        """Why the bridge can no longer be used (server died), or None."""
        return self._broken

    def list_tools(self) -> list[types.Tool]:
        """The server's tools, as listed once at startup (the listing does not change)."""
        self._usable_session()
        return list(self._tools)

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> types.CallToolResult:
        """One `tools/call`, blocking for at most about `timeout_s`.

        Raises `McpTimeout`, `McpServerDied` (also, immediately, on every call after a
        death), or `McpProtocolError`.
        """
        session = self._usable_session()
        what = f"tools/call {name!r}"
        result = self._submit(
            session.call_tool(name, arguments, read_timeout_seconds=timeout_s), timeout_s, what=what
        )
        if not isinstance(result, types.CallToolResult):  # e.g. an input-required result
            raise McpProtocolError(f"{what}: unexpected result type {type(result).__name__}")
        return result

    # --- internals ------------------------------------------------------------------------------

    def _usable_session(self) -> ClientSession:
        if self._broken is not None:
            raise McpServerDied(self._broken)
        session = self._session
        if session is None or self._closed:
            raise McpError("the MCP bridge is not running (start() it first, or it was closed)")
        return session

    def _submit(self, coro: Coroutine[Any, Any, T], timeout_s: float, *, what: str) -> T:
        """Run `coro` on the bridge loop and wait for it, mapping every failure to a type."""
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout_s + _BACKSTOP_S)
        except concurrent.futures.TimeoutError:
            future.cancel()  # cancels the request task; the SDK sends notifications/cancelled
            raise McpTimeout(f"{what} timed out after {timeout_s:g} s") from None
        except MCPError as exc:
            if exc.code == REQUEST_TIMEOUT:
                raise McpTimeout(f"{what} timed out after {timeout_s:g} s") from exc
            if exc.code == CONNECTION_CLOSED:
                raise self._mark_broken(f"{what}: connection closed") from exc
            raise McpProtocolError(f"{what}: server error {exc.code}: {exc.message}") from exc
        except Exception as exc:
            # A write to a dead server's stdin, or the session task gone: the pipe is broken.
            if isinstance(exc, _PIPE_ERRORS) or self._finished.is_set():
                raise self._mark_broken(f"{what}: {_describe(exc)}") from exc
            raise McpProtocolError(f"{what}: {_describe(exc)}") from exc

    def _mark_broken(self, reason: str) -> McpServerDied:
        if self._broken is None:
            self._broken = f"the MCP server is gone ({reason}); restart the agent"
            log_event(
                logger,
                "mcp_server_died",
                level=logging.ERROR,
                server_pid=self.server_pid,
                reason=reason,
            )
        return McpServerDied(self._broken)

    def _run_loop(self) -> None:
        """Bridge thread: run the loop until close(), then cancel leftovers and close it."""
        loop = self._loop
        assert loop is not None
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            # Leftover tasks are requests abandoned after a timeout: cancel them cleanly.
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _session_main(self, startup_timeout_s: float) -> None:
        """The ONE task that owns the transport and the session (enter AND exit)."""
        assert self._shutdown is not None
        try:
            async with stdio_client(self._params, errlog=self._errlog) as streams:
                read_stream, write_stream = streams
                async with ClientSession(read_stream, write_stream) as session:
                    # anyio's own timeout: cancellation stays inside anyio's scopes, so the
                    # shielded shutdown of stdio_client still kills the process on failure.
                    with anyio.fail_after(startup_timeout_s):
                        await session.initialize()
                        listing = await session.list_tools()
                    self._tools = list(listing.tools)
                    pid = (listing.meta or {}).get(SERVER_PID_META_KEY)
                    self.server_pid = pid if isinstance(pid, int) else None
                    self._session = session
                    self._ready.set()
                    await self._shutdown.wait()
                    self._session = None
        except BaseException as exc:  # reported to start(), or recorded as a server death
            if not self._ready.is_set():
                self._startup_error = exc
            elif not self._closed:
                self._mark_broken(f"session ended: {_describe(exc)}")
            if not isinstance(exc, Exception):
                raise  # CancelledError / KeyboardInterrupt: let the loop see it
        finally:
            self._session = None
            self._ready.set()  # unblock start() at once on failure
            self._finished.set()


def _real_stderr() -> TextIO:
    """`sys.stderr` if it is backed by a file descriptor, else the original stderr."""
    try:
        sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        return sys.__stderr__ if sys.__stderr__ is not None else sys.stderr
    return sys.stderr


def _describe(exc: BaseException | None) -> str:
    """Readable one-liner, unwrapping anyio's exception groups to their first leaf."""
    if exc is None:
        return "unknown error"
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    text = str(exc) or type(exc).__name__
    return text if text.startswith(type(exc).__name__) else f"{type(exc).__name__}: {text}"
