"""`run_agent` driven by FakeLLM: no Ollama, no real sleeping (FakeClock for the deadline)."""

import threading
from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from fakes import FakeClock, FakeLLM, text_response, tool_call_response
from local_research_agent.agent import (
    SYSTEM_PROMPT,
    AgentConfig,
    AgentLimits,
    AgentResult,
    StopReason,
    run_agent,
)
from local_research_agent.llm import ChatOptions, LLMError, LLMResponse
from local_research_agent.tools import Tool, ToolRegistry, calculator_tool, current_date_tool


class NoArgs(BaseModel):
    pass


@pytest.fixture
def release() -> Iterator[threading.Event]:
    """Set at teardown, so threads blocked by a 'slow' tool/LLM always finish."""
    event = threading.Event()
    yield event
    event.set()


def make_registry(*extra: Tool) -> ToolRegistry:  # type: ignore[type-arg]
    return ToolRegistry([calculator_tool, current_date_tool, *extra])


def run(
    llm: FakeLLM,
    registry: ToolRegistry | None = None,
    *,
    clock: FakeClock | None = None,
    **config: object,
) -> AgentResult:
    return run_agent(
        llm,
        registry or make_registry(),
        AgentConfig(**config),  # type: ignore[arg-type]
        "question?",
        clock=clock or FakeClock(),
    )


def tool_messages(messages: list[dict]) -> list[str]:  # type: ignore[type-arg]
    return [m["content"] for m in messages if m["role"] == "tool"]


def calc(expression: str) -> LLMResponse:
    return tool_call_response(("calculator", {"expression": expression}))


# --- happy paths -----------------------------------------------------------------------------


def test_multi_step_dependent_tool_calls_then_final_answer() -> None:
    llm = FakeLLM([calc("17 * 23"), calc("391 + 4"), text_response("395")])
    options = ChatOptions(temperature=0.0, num_ctx=4096)

    result = run(llm, llm_options=options)

    assert result.stop_reason is StopReason.FINAL_ANSWER
    assert result.final_answer == "395"
    assert result.state.stop_reason is StopReason.FINAL_ANSWER
    assert result.metrics.iterations == 3
    assert len(llm.calls) == 3

    first, second, third = llm.calls
    assert first.messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "question?"},
    ]
    assert first.options is options
    assert [s["function"]["name"] for s in first.tools] == ["calculator", "current_date"]
    # The second call sees the first result: that is what makes the steps dependent.
    assert second.messages[-1] == {
        "role": "tool",
        "tool_name": "calculator",
        "content": "17 * 23 = 391",
    }
    assert [m["role"] for m in third.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert [m["role"] for m in result.state.messages][-1] == "assistant"

    # Metrics are summed over the three LLM calls (fakes: 20/8, 20/8, 10/5 tokens).
    assert result.metrics.prompt_tokens == 50
    assert result.metrics.completion_tokens == 21
    assert [c.ok for c in result.state.tool_calls] == [True, True]


def test_several_tool_calls_in_one_assistant_message() -> None:
    llm = FakeLLM(
        [
            tool_call_response(
                ("calculator", {"expression": "2 + 2"}), ("calculator", {"expression": "3 * 3"})
            ),
            text_response("4 and 9"),
        ]
    )

    result = run(llm)

    second = llm.calls[1].messages
    # The assistant message is appended ONCE, followed by one tool message per call, in order.
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "tool"]
    assert len(second[2]["tool_calls"]) == 2
    assert tool_messages(second) == ["2 + 2 = 4", "3 * 3 = 9"]
    assert result.final_answer == "4 and 9"
    assert [c.iteration for c in result.state.tool_calls] == [1, 1]


def test_no_system_message_when_prompt_is_empty() -> None:
    llm = FakeLLM([text_response("Paris.")])
    run(llm, system_prompt="")
    assert llm.calls[0].messages == [{"role": "user", "content": "question?"}]


# --- MAX_ITERATIONS ----------------------------------------------------------------------------


def test_max_iterations_stops_without_extra_llm_call() -> None:
    # The model always asks for a (different) tool call: it never answers.
    llm = FakeLLM([calc(f"{i} + 1") for i in range(3)])

    result = run(llm, limits=AgentLimits(max_iterations=3))

    assert result.stop_reason is StopReason.MAX_ITERATIONS
    assert result.final_answer is None
    assert len(llm.calls) == 3  # no 4th call to "force" an answer
    # Iterations 1 and 2 ran their tool; the call requested at iteration 3 was NOT executed,
    # but it still got a tool message, so the history is well formed.
    assert len(result.state.tool_calls) == 2
    assert tool_messages(result.state.messages)[-1] == (
        "Error: Not executed: the run stopped (max_iterations)."
    )


def test_final_answer_at_the_last_iteration_is_accepted() -> None:
    llm = FakeLLM([calc("1 + 1"), text_response("2")])
    result = run(llm, limits=AgentLimits(max_iterations=2))
    assert result.stop_reason is StopReason.FINAL_ANSWER
    assert result.final_answer == "2"


# --- REPETITION --------------------------------------------------------------------------------


def test_repetition_warns_first_then_stops() -> None:
    # Same call three times; key order and whitespace differ, the normalized args do not.
    llm = FakeLLM(
        [
            calc("1 + 1"),
            tool_call_response(("calculator", '{ "expression" : "1 + 1" }')),
            calc("1 + 1"),
        ]
    )

    result = run(llm)

    assert result.stop_reason is StopReason.REPETITION
    assert result.final_answer is None
    assert len(llm.calls) == 3
    first, warning, stop = tool_messages(result.state.messages)
    assert first == "1 + 1 = 2"
    assert warning.startswith("Error: Repeated call: you already called 'calculator'")
    assert stop.startswith("Error: Not executed: identical call repeated")
    # The warned call was not executed, and a warning is not a tool error.
    assert [c.repeated for c in result.state.tool_calls] == [False, True]
    assert result.state.consecutive_tool_errors == 0


def test_model_can_recover_after_the_repetition_warning() -> None:
    llm = FakeLLM([calc("1 + 1"), calc("1 + 1"), text_response("2")])
    result = run(llm)
    assert result.stop_reason is StopReason.FINAL_ANSWER
    assert result.final_answer == "2"


def test_retrying_a_failed_call_is_not_a_repetition() -> None:
    # Identical failing calls are bounded by the error budget, not the repetition guard.
    llm = FakeLLM([calc("1 / 0"), calc("1 / 0"), text_response("cannot divide by zero")])
    result = run(llm)
    assert not any(c.repeated for c in result.state.tool_calls)
    assert result.stop_reason is StopReason.FINAL_ANSWER


# --- timeouts -------------------------------------------------------------------------------


def test_tool_timeout_is_an_observation_and_the_loop_continues(
    release: threading.Event,
) -> None:
    slow = Tool("slow", "Never returns in time.", NoArgs, lambda args: release.wait(5))
    llm = FakeLLM([tool_call_response(("slow", {})), text_response("the tool was too slow")])

    result = run(llm, make_registry(slow), limits=AgentLimits(tool_timeout_s=0.05))

    assert result.stop_reason is StopReason.FINAL_ANSWER
    [record] = result.state.tool_calls
    assert not record.ok
    assert record.error_type == "timeout"
    assert tool_messages(llm.calls[1].messages) == ["Error: Tool 'slow' timed out after 0.05 s."]
    assert result.state.consecutive_tool_errors == 1


def test_total_timeout() -> None:
    clock = FakeClock()
    # Each call of this tool "takes" 200 s of fake time: no real sleeping.
    tick = Tool("tick", "Advance time.", NoArgs, lambda args: clock.advance(200) or "tick")
    # Different args each time (NoArgs ignores extra fields) so the repetition guard stays out
    # of the way. Only two responses are scripted: a third LLM call would fail the test.
    llm = FakeLLM([tool_call_response(("tick", {})), tool_call_response(("tick", {"n": 1}))])

    result = run(llm, make_registry(tick), clock=clock, limits=AgentLimits(total_timeout_s=300))

    # t=200 after iteration 1 (< 300, continue); t=400 after iteration 2; the deadline check
    # at the top of iteration 3 stops the run before any LLM call.
    assert result.stop_reason is StopReason.TIMEOUT
    assert result.final_answer is None
    assert len(llm.calls) == 2
    assert result.metrics.wall_time_ms == pytest.approx(400_000)


def test_total_timeout_stops_remaining_calls_of_a_batch() -> None:
    clock = FakeClock()
    tick = Tool("tick", "Advance time.", NoArgs, lambda args: clock.advance(400) or "tick")
    llm = FakeLLM([tool_call_response(("tick", {}), ("calculator", {"expression": "1 + 1"}))])

    result = run(llm, make_registry(tick), clock=clock, limits=AgentLimits(total_timeout_s=300))

    assert result.stop_reason is StopReason.TIMEOUT
    assert tool_messages(result.state.messages) == [
        "tick",
        "Error: Not executed: the run stopped (timeout).",
    ]


def test_llm_timeout_becomes_llm_error(release: threading.Event) -> None:
    class SlowLLM:
        def chat(self, messages, tools=None, options=None):  # type: ignore[no-untyped-def]
            release.wait(5)
            raise AssertionError("should have been abandoned")

    result = run_agent(
        SlowLLM(),
        make_registry(),
        AgentConfig(limits=AgentLimits(llm_timeout_s=0.05)),
        "question?",
        clock=FakeClock(),
    )

    assert result.stop_reason is StopReason.LLM_ERROR
    assert result.state.error == "LLM call timed out after 0.05 s"
    assert result.metrics.iterations == 0


# --- errors ----------------------------------------------------------------------------------


def test_llm_error_stops_without_raising() -> None:
    llm = FakeLLM([calc("1 + 1"), LLMError("Cannot reach Ollama at http://x")])

    result = run(llm)

    assert result.stop_reason is StopReason.LLM_ERROR
    assert result.final_answer is None
    assert result.state.error == "Cannot reach Ollama at http://x"
    assert result.metrics.iterations == 1


def test_repeated_invalid_args_stop_with_too_many_tool_errors() -> None:
    # Different wrong argument each time: this is the error budget, not repetition.
    llm = FakeLLM([tool_call_response(("calculator", {"expr": str(i)})) for i in range(3)])

    result = run(llm)

    assert result.stop_reason is StopReason.TOO_MANY_TOOL_ERRORS
    assert len(llm.calls) == 3
    assert all(
        m.startswith("Error: Invalid arguments") for m in tool_messages(llm.calls[2].messages)
    )


def test_valid_call_resets_the_error_counter() -> None:
    llm = FakeLLM(
        [
            tool_call_response(("calculator", {"expr": "1"})),
            tool_call_response(("calculator", {"expr": "2"})),
            calc("1 + 1"),  # success: counter back to 0
            tool_call_response(("calculator", {"expr": "3"})),
            tool_call_response(("calculator", {"expr": "4"})),
            text_response("2"),
        ]
    )

    result = run(llm)

    assert result.stop_reason is StopReason.FINAL_ANSWER
    assert result.state.consecutive_tool_errors == 2
    assert result.metrics.tool_errors == 4


def test_tool_outside_allowed_tools_is_an_unknown_tool() -> None:
    llm = FakeLLM([tool_call_response(("current_date", {})), text_response("no date tool")])

    result = run(llm, allowed_tools=("calculator",))

    # Only the allowed schema is offered...
    assert [s["function"]["name"] for s in llm.calls[0].tools] == ["calculator"]
    # ...and a call outside the subset is an observation, not a crash or an execution.
    [observation] = tool_messages(llm.calls[1].messages)
    assert observation == "Error: Unknown tool 'current_date'. Available tools: calculator."
    assert result.state.tool_calls[0].error_type == "unknown_tool"
    assert result.stop_reason is StopReason.FINAL_ANSWER


def test_unknown_name_in_allowed_tools_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="web_search"):
        run(FakeLLM([]), allowed_tools=("web_search",))


# --- observability -----------------------------------------------------------------------------


def test_iteration_and_finished_events_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM([calc("1 + 1"), text_response("2")])
    with caplog.at_level("INFO", logger="local_research_agent"):
        run(llm)

    events = {}
    for record in caplog.records:
        events.setdefault(record.getMessage(), []).append(record.fields)  # type: ignore[attr-defined]

    first, second = events["agent_iteration"]
    assert first["tool_calls"] == [{"name": "calculator", "ok": True, "repeated": False}]
    assert first["context_growth_tokens"] is None
    assert second["context_growth_tokens"] == 10 - 20
    assert second["stop_reason"] == "final_answer"
    assert "load_duration_ms" in events["llm_call"][0]
    [finished] = events["agent_finished"]
    assert finished["stop_reason"] == "final_answer"
    assert finished["iterations"] == 2
    assert finished["prompt_tokens"] == 30
