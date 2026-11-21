"""What a batch ran ON, beyond its config: tool schemas, Ollama server and model, git state.

- `tool_schemas_sha256`: SHA-256 of the canonical JSON of the ORDERED list of tool schemas
  (keys sorted, compact separators, UTF-8, list order kept). `SchemaRecorder` wraps the
  run's `LLMClient` and records the `tools` argument of each `chat()` call, so the hash is of
  exactly what the agent sent: after provider aggregation, after `subset`, and, with the MCP
  backend, as rebuilt from `list_tools`.
- `read_ollama_info`: the server version (`GET /api/version`) and the model digest (the
  `digest` of the model's entry in `GET /api/tags`). Never raises: anything unreadable is None.
- `working_tree_changes` / `git_commit`: `git status --porcelain`, which lists modified,
  staged and untracked files but not ignored ones (.gitignore, .git/info/exclude).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from local_research_agent.llm import ChatOptions, LLMClient, LLMResponse, Message, ToolSchema

from .suite import EVALS_DIR

REPO_ROOT = EVALS_DIR.parent
OLLAMA_INFO_TIMEOUT_S = 3.0


# --- tool schemas ---------------------------------------------------------------------------


def canonical_schemas_json(schemas: Sequence[ToolSchema]) -> bytes:
    """The bytes that are hashed. `sort_keys` orders the keys INSIDE each schema; the list
    itself keeps its order, because the model sees the tools in that order."""
    return json.dumps(
        list(schemas), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def tool_schemas_sha256(schemas: Sequence[ToolSchema]) -> str:
    """SHA-256 (hex) of `canonical_schemas_json(schemas)`."""
    return hashlib.sha256(canonical_schemas_json(schemas)).hexdigest()


class SchemaRecorder:
    """An `LLMClient` that forwards every call and remembers the tool schemas it was given."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.hashes: list[str] = []  # one per chat() call, in order

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        options: ChatOptions | None = None,
    ) -> LLMResponse:
        """Record the hash of `tools`, then forward the call unchanged."""
        self.hashes.append(tool_schemas_sha256(tools or []))
        return self._inner.chat(messages, tools=tools, options=options)

    @property
    def sha256(self) -> str | None:
        """The run's schema hash; None if the LLM was never called.

        The loop computes the schemas once per run, so every call carries the same list. If
        they ever differed, the value says so instead of hiding it behind the first call's.
        """
        distinct = list(dict.fromkeys(self.hashes))
        if not distinct:
            return None
        return distinct[0] if len(distinct) == 1 else "inconsistent:" + ",".join(distinct)


# --- Ollama ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaInfo:
    """Ollama server version and model digest; None when unknown."""

    version: str | None
    model_digest: str | None


def read_ollama_info(
    host: str | None,
    model: str,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = OLLAMA_INFO_TIMEOUT_S,
) -> OllamaInfo:
    """Server version and model digest; None for whatever cannot be read. Never raises."""
    if not host:
        return OllamaInfo(None, None)
    try:
        with httpx.Client(base_url=host, timeout=timeout_s, transport=transport) as http:
            version = _get_json(http, "/api/version")
            tags = _get_json(http, "/api/tags")
    except Exception:  # an unreachable server, a bad URL...: provenance must not end the run
        return OllamaInfo(None, None)
    raw_version = version.get("version") if isinstance(version, dict) else None
    return OllamaInfo(
        raw_version if isinstance(raw_version, str) and raw_version else None,
        _digest_of(tags, model),
    )


def _get_json(http: httpx.Client, path: str) -> Any:
    try:
        response = http.get(path)
        return response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def _digest_of(tags: Any, model: str) -> str | None:
    """The `digest` of `model` in a /api/tags body. A name without a tag means `:latest`."""
    models = tags.get("models") if isinstance(tags, dict) else None
    if not isinstance(models, list):
        return None
    wanted = model if ":" in model else f"{model}:latest"
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if wanted in (entry.get("name"), entry.get("model")):
            digest = entry.get("digest")
            return digest if isinstance(digest, str) and digest else None
    return None


# --- git ------------------------------------------------------------------------------------


def working_tree_changes(repo: Path | None = None) -> list[str] | None:
    """`git status --porcelain` lines (ignored files excluded); None if git cannot tell."""
    out = _git(repo or REPO_ROOT, "status", "--porcelain")
    return None if out is None else [line for line in out.splitlines() if line.strip()]


def git_commit(repo: Path | None = None) -> str | None:
    """Short HEAD hash, with "+dirty" when the working tree has changes; None if unknown."""
    head = _git(repo or REPO_ROOT, "rev-parse", "--short", "HEAD")
    changes = working_tree_changes(repo)
    if head is None or changes is None:
        return None
    return head.strip() + ("+dirty" if changes else "")


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
