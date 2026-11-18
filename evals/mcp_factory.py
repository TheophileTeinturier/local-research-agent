"""Registry factory for the MCP server in evaluation runs (runs INSIDE the server process).

    python -m local_research_agent.mcp.server --registry-factory evals.mcp_factory:build

In MCP mode the fake arXiv transport and the frozen clock must live where the tools run: in
the server process. The runner (`runner.run_case`) writes a per-run CASE FILE and passes its
path in `CASE_FILE_ENV`; `build()` reads it and builds the case's world exactly as the local
mode does (`CaseArxivTransport` + frozen `current_date`).

Fixture misses happen in the server, but the runner needs them to flag the run inconclusive.
So the transport APPENDS each miss (one query string per line) to the file named by
`MISSES_FILE_ENV`; the runner reads it after the run. `failure_mode` (http_503, timeout)
needs nothing special: the transport raises/returns the failure inside the server, and the
error observation crosses MCP like any other result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from local_research_agent.tools import ToolRegistry, build_registry

from .arxiv_fake import CaseArxivTransport
from .suite import Case

CASE_FILE_ENV = "LRA_EVAL_CASE_FILE"
MISSES_FILE_ENV = "LRA_EVAL_MISSES_FILE"


def write_case_file(path: Path, case: Case, fixtures_dir: Path) -> None:
    """What `build()` needs to rebuild the case's world in another process."""
    payload = {"case": case.model_dump(mode="json"), "fixtures_dir": str(fixtures_dir)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_misses(path: Path) -> tuple[str, ...]:
    """The misses the server recorded (none if it never wrote the file)."""
    if not path.is_file():
        return ()
    return tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line)


class MissRecordingTransport(CaseArxivTransport):
    """`CaseArxivTransport` that also appends every miss to a file (one line per miss)."""

    def __init__(self, case: Case, fixtures_dir: Path, misses_file: Path) -> None:
        super().__init__(case.arxiv, fixtures_dir)
        self._misses_file = misses_file

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        before = len(self.served)
        try:
            return super().handle_request(request)
        finally:  # also when the transport raises (failure_mode "timeout")
            new_misses = [r.key for r in self.served[before:] if r.outcome == "miss"]
            if new_misses:
                with self._misses_file.open("a", encoding="utf-8") as handle:
                    handle.writelines(f"{key}\n" for key in new_misses)


def build() -> ToolRegistry:
    """The registry of the case named by `CASE_FILE_ENV` (fake arXiv, frozen clock)."""
    payload = json.loads(Path(os.environ[CASE_FILE_ENV]).read_text(encoding="utf-8"))
    case = Case.model_validate(payload["case"])
    transport = MissRecordingTransport(
        case, Path(payload["fixtures_dir"]), Path(os.environ[MISSES_FILE_ENV])
    )
    frozen_now = case.frozen_now
    return build_registry(arxiv_transport=transport, clock=lambda: frozen_now)
