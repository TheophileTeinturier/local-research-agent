import json
import logging
import re

import pytest
from pydantic import BaseModel, Field

from local_research_agent.tools import Tool, ToolError, ToolRegistry, build_default_registry


class EchoArgs(BaseModel):
    text: str = Field(description="Text to echo back.")
    times: int = Field(default=1, ge=1, le=10)


def _echo(args: EchoArgs) -> str:
    return args.text * args.times


def _boom(args: EchoArgs) -> str:
    raise RuntimeError("disk on fire")


def _domain_error(args: EchoArgs) -> str:
    raise ToolError("text must not be shouted")


def _returns_dict(args: EchoArgs) -> dict[str, object]:
    return {"text": args.text, "n": args.times}


def make_registry(**kwargs: object) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool("echo", "Echo text.", EchoArgs, _echo),
            Tool("boom", "Always crashes.", EchoArgs, _boom),
            Tool("picky", "Raises a domain error.", EchoArgs, _domain_error),
            Tool("as_dict", "Returns a dict.", EchoArgs, _returns_dict),
        ],
        **kwargs,  # type: ignore[arg-type]
    )


# --- schemas ---------------------------------------------------------------------------


def test_default_registry_tools() -> None:
    assert build_default_registry().names == ["calculator", "current_date"]


def test_schemas_use_ollama_function_format() -> None:
    schema = build_default_registry().schemas()[0]
    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "calculator"
    assert "arithmetic" in function["description"]
    params = function["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["expression"]
    assert params["additionalProperties"] is False  # from extra="forbid"
    assert params["properties"]["expression"]["type"] == "string"
    assert "description" in params["properties"]["expression"]
    assert "title" not in json.dumps(params)  # generated titles are stripped


def test_schema_keeps_a_field_named_title() -> None:
    class Args(BaseModel):
        title: str

    params = ToolRegistry([Tool("t", "d", Args, lambda a: a.title)]).schemas()[0]["function"][
        "parameters"
    ]
    assert list(params["properties"]) == ["title"]
    assert "title" not in params["properties"]["title"]


def test_duplicate_registration_rejected() -> None:
    registry = make_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Tool("echo", "again", EchoArgs, _echo))


def test_invalid_tool_name_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid tool name"):
        Tool("has space", "d", EchoArgs, _echo)


# --- subset ----------------------------------------------------------------------------------


def test_subset_keeps_only_named_tools_and_settings() -> None:
    registry = make_registry(max_output_chars=50)
    subset = registry.subset(["picky", "echo"])
    assert subset.names == ["echo", "picky"]  # registry order, not argument order
    assert subset.max_output_chars == 50
    assert subset.get("echo") is registry.get("echo")  # same Tool objects
    assert len(registry) == 4  # the original is untouched


def test_subset_call_outside_is_unknown_tool() -> None:
    result = make_registry().subset(["echo"]).execute("boom", {"text": "x"})
    assert not result.ok
    assert result.metadata["error_type"] == "unknown_tool"
    assert "Available tools: echo." in result.content


def test_subset_with_unregistered_name_raises() -> None:
    with pytest.raises(ValueError, match="nope"):
        make_registry().subset(["echo", "nope"])


# --- execute: success ----------------------------------------------------------------------


def test_execute_success() -> None:
    result = make_registry().execute("echo", {"text": "ab", "times": 2})
    assert result.ok
    assert result.content == "abab"
    assert result.metadata["duration_ms"] >= 0


def test_current_date_tool() -> None:
    result = build_default_registry().execute("current_date", {})
    assert result.ok
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result.content)


def test_execute_accepts_json_string_arguments() -> None:
    result = make_registry().execute("echo", '{"text": "hi"}')
    assert result.ok and result.content == "hi"


def test_non_string_output_is_serialized_as_compact_json() -> None:
    result = make_registry().execute("as_dict", {"text": "é"})
    assert result.content == '{"text":"é","n":1}'


# --- execute: errors become observations ---------------------------------------------------


def test_unknown_tool() -> None:
    result = make_registry().execute("delete_everything", {})
    assert not result.ok
    assert result.metadata["error_type"] == "unknown_tool"
    assert "Available tools: echo, boom, picky, as_dict" in result.content


@pytest.mark.parametrize(
    ("raw_args", "message_part"),
    [
        ({}, "text: Field required"),
        ({"text": "a", "times": 99}, "times: Input should be less than or equal to 10"),
        ({"text": ["not", "a", "string"]}, "text: Input should be a valid string"),
        ('{"text": ', "not valid JSON"),
        ("[1, 2]", "must be a JSON object, got list"),
    ],
)
def test_invalid_arguments(raw_args: object, message_part: str) -> None:
    result = make_registry().execute("echo", raw_args)  # type: ignore[arg-type]
    assert not result.ok
    assert result.metadata["error_type"] == "invalid_arguments"
    assert result.content.startswith("Error: ")
    assert message_part in result.content


def test_invalid_arguments_message_includes_schema() -> None:
    result = make_registry().execute("echo", {})
    assert 'Expected JSON schema: {"properties":{"text"' in result.content


def test_extra_field_rejected_when_model_forbids_it() -> None:
    result = build_default_registry().execute("calculator", {"expression": "1+1", "unit": "kg"})
    assert not result.ok
    assert "unit: Extra inputs are not permitted" in result.content


def test_tool_exception_is_converted(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="local_research_agent"):
        result = make_registry().execute("boom", {"text": "x"})
    assert not result.ok
    assert result.metadata["error_type"] == "exception"
    assert result.content == "Error: Tool 'boom' failed: RuntimeError: disk on fire"
    # The traceback is logged for the developer, not sent to the model.
    assert any(record.exc_info for record in caplog.records)


def test_tool_error_message_is_passed_through() -> None:
    result = make_registry().execute("picky", {"text": "x"})
    assert not result.ok
    assert result.metadata["error_type"] == "tool_error"
    assert result.content == "Error: text must not be shouted"


def test_keyboard_interrupt_is_not_swallowed() -> None:
    def interrupt(args: EchoArgs) -> str:
        raise KeyboardInterrupt

    registry = ToolRegistry([Tool("stop", "d", EchoArgs, interrupt)])
    with pytest.raises(KeyboardInterrupt):
        registry.execute("stop", {"text": "x"})


# --- output cap -------------------------------------------------------------------------------


def test_oversized_output_is_truncated() -> None:
    registry = make_registry(max_output_chars=50)
    result = registry.execute("echo", {"text": "x" * 40, "times": 3})
    assert result.ok
    assert result.content.startswith("x" * 50)
    assert result.content.endswith("[truncated: showing 50 of 120 characters]")
    assert result.metadata["truncated"] is True
    assert result.metadata["original_length"] == 120


def test_output_at_limit_is_untouched() -> None:
    result = make_registry(max_output_chars=4).execute("echo", {"text": "abcd"})
    assert result.content == "abcd"
    assert "truncated" not in result.metadata


def test_execution_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="local_research_agent"):
        make_registry().execute("echo", {"text": "x"})
    [record] = [r for r in caplog.records if r.getMessage() == "tool_execution"]
    assert record.fields["tool"] == "echo"  # type: ignore[attr-defined]
    assert record.fields["ok"] is True  # type: ignore[attr-defined]
    assert "duration_ms" in record.fields  # type: ignore[attr-defined]
