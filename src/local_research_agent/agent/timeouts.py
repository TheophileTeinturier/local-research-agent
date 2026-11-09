"""Run a blocking call with a timeout, using a worker thread.

LIMITATION: Python cannot kill a thread. When the timeout expires we stop *waiting*, but the
call keeps running in its (daemon) thread until it returns by itself: a tool stuck in an
infinite loop keeps burning a CPU core, and an abandoned HTTP request keeps the server busy
(the Ollama client's own read timeout eventually ends it). The thread is a daemon so it never
prevents the process from exiting. Real isolation (killable work) would need a subprocess.
The worker never touches the agent state: it only computes a return value.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class CallTimedOut(Exception):
    """The call did not finish within the timeout (it may still be running)."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__(f"timed out after {timeout_s:g} s")
        self.timeout_s = timeout_s


def call_with_timeout(fn: Callable[[], T], timeout_s: float, *, name: str = "worker") -> T:
    """Return `fn()`, re-raise its exception, or raise `CallTimedOut` after `timeout_s`."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # handed over to the caller's thread, re-raised below
            outcome["error"] = exc

    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise CallTimedOut(timeout_s)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
