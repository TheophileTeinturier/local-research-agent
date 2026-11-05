"""Tool definition: the single source of truth for a tool.

A tool is defined once (name, description, Pydantic argument model, plain Python function).
Everything else is derived from it: the JSON schema sent to the LLM now, and the MCP tool
listing later. Nothing about a tool is written twice.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

ArgsT = TypeVar("ArgsT", bound=BaseModel)

# The function-name charset accepted by OpenAI-style and MCP tool APIs.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ToolError(Exception):
    """An expected failure the tool reports to the model (bad input, domain error).

    The registry passes its message to the LLM unchanged, so write it as an instruction the
    model can act on ("use ** for powers"), not as a stack trace.
    """


@dataclass(frozen=True)
class ToolResult:
    """The observation sent back to the LLM. Failures are results too (ok=False), not exceptions."""

    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Tool(Generic[ArgsT]):
    name: str
    description: str  # prompt engineering: this is how the model decides when/how to call it
    args_model: type[ArgsT]
    function: Callable[[ArgsT], Any]  # receives the *validated* model instance

    def __post_init__(self) -> None:
        if not _TOOL_NAME_RE.fullmatch(self.name):
            raise ValueError(f"Invalid tool name {self.name!r}: use [A-Za-z0-9_-], max 64 chars")
        if not self.description.strip():
            raise ValueError(f"Tool {self.name!r} needs a description")

    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema of the arguments, generated from the Pydantic model.

        Field types, `required`, `Field(description=...)`, constraints such as max_length, and
        `additionalProperties: false` (for extra="forbid") all come from Pydantic. The
        auto-generated "title" entries are removed: they add prompt tokens and tell the model
        nothing.
        """
        return _strip_titles(self.args_model.model_json_schema())


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_strip_titles(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            # The keys here are field/definition *names*, not schema keywords: a field named
            # "title" must survive. Only clean the sub-schemas.
            cleaned[key] = {name: _strip_titles(sub) for name, sub in value.items()}
        else:
            cleaned[key] = _strip_titles(value)
    return cleaned
