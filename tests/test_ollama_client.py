"""OllamaClient against a mocked HTTP transport, with payloads captured from Ollama 0.40.14."""

import json

import httpx
import pytest

from local_research_agent.llm import ChatOptions, LLMError, OllamaClient
from local_research_agent.llm.ollama import normalize_host, parse_chat_response

# Captured from `POST /api/chat` with qwen3:8b, think=false (2026-11-04).
REAL_TOOL_CALL_RESPONSE = {
    "model": "qwen3:8b",
    "created_at": "2026-11-04T09:28:03.9585078Z",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_3ufh7f2a",
                "function": {
                    "index": 0,
                    "name": "calculator",
                    "arguments": {"expression": "17*23+4"},
                },
            }
        ],
    },
    "done": True,
    "done_reason": "stop",
    "total_duration": 11677237000,
    "load_duration": 7082206500,
    "prompt_eval_count": 146,
    "prompt_eval_cached_count": 0,
    "prompt_eval_duration": 4122179000,
    "eval_count": 25,
    "eval_duration": 453254000,
}


def make_client(handler) -> OllamaClient:  # type: ignore[no-untyped-def]
    return OllamaClient(model="qwen3:8b", transport=httpx.MockTransport(handler))


def test_request_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=REAL_TOOL_CALL_RESPONSE)

    tools = [{"type": "function", "function": {"name": "calculator"}}]
    with make_client(handler) as client:
        client.chat(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            options=ChatOptions(temperature=0.0, num_ctx=4096, think=True),
        )

    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["body"] == {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "think": True,  # top level, not in options
        "options": {"temperature": 0.0, "num_ctx": 4096},
        "tools": tools,
    }


def test_think_none_omits_field_and_no_tools_omits_tools() -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})

    with make_client(handler) as client:
        client.chat([], options=ChatOptions(think=None))
    assert "think" not in bodies[0]
    assert "tools" not in bodies[0]


def test_parse_real_tool_call_response() -> None:
    response = parse_chat_response(REAL_TOOL_CALL_RESPONSE)
    [call] = response.tool_calls
    assert call.name == "calculator"
    assert call.arguments == {"expression": "17*23+4"}
    assert call.id == "call_3ufh7f2a"
    assert response.metrics.prompt_eval_count == 146
    assert response.metrics.eval_count == 25
    assert response.metrics.total_duration_ms == pytest.approx(11677.237)
    assert response.metrics.load_duration_ms == pytest.approx(7082.2065)  # cold start
    assert response.message is REAL_TOOL_CALL_RESPONSE["message"]  # kept verbatim for history


def test_parse_thinking_and_content() -> None:
    response = parse_chat_response(
        {"message": {"role": "assistant", "content": "395", "thinking": "17*23=391..."}}
    )
    assert response.content == "395"
    assert response.thinking == "17*23=391..."
    assert response.tool_calls == []


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"expression": "1"}', '{"expression": "1"}'),  # string kept raw, decoded by registry
        (None, {}),
        ([1, 2], "[1, 2]"),
    ],
)
def test_parse_unusual_arguments(arguments: object, expected: object) -> None:
    raw = {"message": {"tool_calls": [{"function": {"name": "c", "arguments": arguments}}]}}
    assert parse_chat_response(raw).tool_calls[0].arguments == expected


def test_http_error_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'nope' not found"})

    with make_client(handler) as client, pytest.raises(LLMError, match=r"HTTP 404.*not found"):
        client.chat([])


def test_connection_error_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with make_client(handler) as client, pytest.raises(LLMError, match="Cannot reach Ollama"):
        client.chat([])


def test_missing_message_raises_llm_error() -> None:
    with pytest.raises(LLMError, match="no 'message'"):
        parse_chat_response({"error": "boom"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://localhost:11434/", "http://localhost:11434"),
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("localhost", "http://localhost:11434"),
        ("https://ollama.example.com", "https://ollama.example.com"),
    ],
)
def test_normalize_host(raw: str, expected: str) -> None:
    assert normalize_host(raw) == expected


def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "otherhost:1234")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    client = OllamaClient.from_env()
    assert (client.host, client.model) == ("http://otherhost:1234", "llama3.1:8b")
    assert OllamaClient.from_env(model="qwen3:8b").model == "qwen3:8b"  # argument wins
