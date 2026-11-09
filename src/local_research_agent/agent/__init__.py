"""Agent core. Depends only on the `LLMClient` protocol and `ToolRegistry`, never on a backend."""

from .guards import AgentLimits
from .loop import AgentConfig, run_agent
from .prompts import SYSTEM_PROMPT
from .state import (
    AgentMetrics,
    AgentResult,
    AgentState,
    LLMCallRecord,
    StopReason,
    ToolCallRecord,
)

__all__ = [
    "SYSTEM_PROMPT",
    "AgentConfig",
    "AgentLimits",
    "AgentMetrics",
    "AgentResult",
    "AgentState",
    "LLMCallRecord",
    "StopReason",
    "ToolCallRecord",
    "run_agent",
]
