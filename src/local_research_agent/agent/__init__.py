"""Agent core. Depends only on the `LLMClient` protocol and `ToolRegistry`, never on a backend."""

from .guards import AgentLimits
from .loop import AgentConfig, run_agent
from .prompts import DEFAULT_PROMPT_VARIANT, PROMPT_VARIANTS, SYSTEM_PROMPT, get_system_prompt
from .state import (
    AgentMetrics,
    AgentResult,
    AgentState,
    LLMCallRecord,
    StopReason,
    ToolCallRecord,
)

__all__ = [
    "DEFAULT_PROMPT_VARIANT",
    "PROMPT_VARIANTS",
    "SYSTEM_PROMPT",
    "AgentConfig",
    "AgentLimits",
    "AgentMetrics",
    "AgentResult",
    "AgentState",
    "LLMCallRecord",
    "StopReason",
    "ToolCallRecord",
    "get_system_prompt",
    "run_agent",
]
