from fakes import FakeLLM, text_response, tool_call_response
from local_research_agent.agent import LLMStep, ToolStep, run_once
from local_research_agent.llm import ChatOptions
from local_research_agent.tools import build_default_registry


def test_tool_call_round_trip() -> None:
    llm = FakeLLM(
        [
            tool_call_response(("calculator", {"expression": "17 * 23 + 4"})),
            text_response("17 * 23 + 4 = 395."),
        ]
    )
    registry = build_default_registry()
    options = ChatOptions(think=True)

    result = run_once(llm, registry, "What is 17 * 23 + 4?", options=options)

    assert result.answer == "17 * 23 + 4 = 395."
    assert result.completed

    # First call: just the user message, with the tool schemas and the options.
    first, second = llm.calls
    assert first.messages == [{"role": "user", "content": "What is 17 * 23 + 4?"}]
    assert first.tools == registry.schemas()
    assert first.options is options

    # Second call: user -> assistant(tool_calls) -> tool(result).
    user, assistant, tool = second.messages
    assert user["role"] == "user"
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    assert tool == {"role": "tool", "tool_name": "calculator", "content": "17 * 23 + 4 = 395"}

    # The returned history also holds the final answer.
    assert [m["role"] for m in result.messages] == ["user", "assistant", "tool", "assistant"]

    llm_step, tool_step, final_step = result.trace
    assert isinstance(llm_step, LLMStep) and llm_step.tool_calls[0].name == "calculator"
    assert isinstance(tool_step, ToolStep) and tool_step.result.ok
    assert isinstance(final_step, LLMStep) and final_step.content == "17 * 23 + 4 = 395."
    assert llm_step.metrics.prompt_eval_count == 20


def test_no_tool_path() -> None:
    llm = FakeLLM([text_response("Paris.")])

    result = run_once(llm, build_default_registry(), "What is the capital of France?")

    assert result.answer == "Paris."
    assert result.completed
    assert len(llm.calls) == 1
    [step] = result.trace
    assert isinstance(step, LLMStep) and step.tool_calls == []
    assert [m["role"] for m in result.messages] == ["user", "assistant"]


def test_tool_error_is_sent_back_as_observation() -> None:
    llm = FakeLLM(
        [
            tool_call_response(("calculator", {"expr": "1+1"})),  # wrong argument name
            text_response("Sorry, I could not compute that."),
        ]
    )

    result = run_once(llm, build_default_registry(), "1+1?")

    tool_message = llm.calls[1].messages[2]
    assert tool_message["role"] == "tool"
    assert tool_message["content"].startswith("Error: Invalid arguments for 'calculator'")
    tool_step = result.trace[1]
    assert isinstance(tool_step, ToolStep) and not tool_step.result.ok
    assert result.answer == "Sorry, I could not compute that."


def test_unknown_tool_does_not_crash_the_run() -> None:
    llm = FakeLLM([tool_call_response(("web_search", {"q": "x"})), text_response("ok")])
    result = run_once(llm, build_default_registry(), "search")
    assert llm.calls[1].messages[2]["content"].startswith("Error: Unknown tool 'web_search'")
    assert result.answer == "ok"


def test_parallel_tool_calls_each_get_a_tool_message() -> None:
    llm = FakeLLM(
        [
            tool_call_response(
                ("calculator", {"expression": "2 + 2"}), ("calculator", {"expression": "3 * 3"})
            ),
            text_response("4 and 9"),
        ]
    )
    run_once(llm, build_default_registry(), "2+2 and 3*3?")
    tool_messages = [m for m in llm.calls[1].messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["2 + 2 = 4", "3 * 3 = 9"]


def test_second_tool_request_is_reported_not_executed() -> None:
    llm = FakeLLM(
        [
            tool_call_response(("calculator", {"expression": "1 + 1"})),
            tool_call_response(("calculator", {"expression": "2 + 2"})),
        ]
    )
    result = run_once(llm, build_default_registry(), "?")
    assert not result.completed
    assert sum(isinstance(step, ToolStep) for step in result.trace) == 1
