# Architecture

This document describes how `local-research-agent` is built: what each module is responsible for, how a tool call travels from the model to Python code and back, how the agent loop stays bounded, how the same tools are served over MCP, and how the evaluation harness measures the agent without changing it.

## 1. Overview

```
            question
               │
               ▼
 ┌───────────────────────────┐   chat(messages, tool schemas)   ┌──────────────────┐
 │  agent loop (run_agent)   │ ───────────────────────────────▶ │  LLMClient       │
 │  guards · timeouts · state│ ◀─────────────────────────────── │  (Ollama /api/chat)
 └─────────────┬─────────────┘   assistant message + tool_calls └──────────────────┘
               │ execute(name, raw_args)
               ▼
 ┌───────────────────────────┐
 │  ToolRegistry             │  validation boundary, output cap, timing, logging
 └──────┬─────────────┬──────┘
        │             │
        ▼             ▼
 LocalToolProvider   McpToolProvider ──stdio JSON-RPC──▶ MCP server ──▶ ToolRegistry ──▶ LocalToolProvider
 (in-process)        (subprocess)
        │
        ▼
 calculator · current_date · search_arxiv · get_paper_details ──▶ ArxivClient ──▶ arXiv API
```

The agent is a single, synchronous, hand-written loop. The LLM backend and the tool backend are both behind small interfaces (`LLMClient`, `ToolProvider`), so the loop does not know whether it talks to Ollama or a scripted fake, or whether tools run in-process or in an MCP server.

## 2. Module responsibilities

| Module | Responsibility |
|---|---|
| `local_research_agent/llm/base.py` | Backend-agnostic types: `Message` (Ollama chat format), `ToolCall`, `LLMResponse`, `LLMMetrics`, `ChatOptions`, `LLMError`, the `LLMClient` protocol and the message builders. The agent core imports only from here. |
| `local_research_agent/llm/ollama.py` | `OllamaClient`: one non-streaming `POST /api/chat` per call. Builds the payload (`think` top-level; `temperature`, `num_ctx`, `seed` in `options`), parses tool calls and metrics, and turns every transport or HTTP failure into `LLMError`. |
| `local_research_agent/agent/loop.py` | `run_agent`: the loop, the single exit point (`_finish`), the per-call timeouts and all mutations of the run state. |
| `local_research_agent/agent/guards.py` | `AgentLimits` and pure guard functions (iterations, deadline, error budget, repetition, argument normalization). No clock, no I/O. |
| `local_research_agent/agent/timeouts.py` | `call_with_timeout`: runs a blocking call in a daemon worker thread and stops waiting after a timeout. |
| `local_research_agent/agent/state.py` | `StopReason`, `AgentState` (the only mutable object of a run), per-call records, `AgentMetrics`, `AgentResult`. |
| `local_research_agent/agent/prompts.py` | Named system prompts (`baseline`, `research_workflow`). |
| `local_research_agent/tools/base.py` | `Tool` (name, description, Pydantic argument model, function), `ToolResult`, `ToolError`. A tool is defined once; its JSON schema is derived from the argument model. |
| `local_research_agent/tools/providers.py` | The `ToolProvider` protocol, `LocalToolProvider`, and helpers shared by every provider (`tool_schema`, `decode_arguments`, `error_result`) so a given input produces the same schema and error text everywhere. |
| `local_research_agent/tools/registry.py` | `ToolRegistry`: aggregates providers, routes calls, rejects duplicate names, supports `subset()`, caps output, times and logs every call, and converts any provider exception into an observation. |
| `local_research_agent/tools/calculator.py`, `current_date.py` | Built-in tools. The calculator walks a Python AST with an allowlist of arithmetic nodes (no `eval`) and resource guards. `current_date` reads an injectable clock. |
| `local_research_agent/tools/arxiv/` | `search_arxiv` and `get_paper_details` over `ArxivClient` (rate limiter, single connection, TTL cache, typed errors), query building, safe XML parsing (`defusedxml`), and an offline fake transport. |
| `local_research_agent/mcp/server.py` | MCP server over stdio (low-level SDK API) exposing any `ToolRegistry`. |
| `local_research_agent/mcp/client.py` | `McpClientBridge`: a synchronous facade over the async MCP client, with typed failures. |
| `local_research_agent/mcp/provider.py` | `McpToolProvider`: the server's tools seen as an ordinary `ToolProvider`. |
| `local_research_agent/observability/logger.py` | JSON-lines logging to stderr (`configure_logging`, `log_event`). |
| `local_research_agent/cli.py` | Command-line entry point (`local-research-agent`): builds the tools (local or MCP), runs the agent once and prints the answer plus a compact trace. |
| `evals/` | Evaluation harness (not part of the installed package): suite schema, fake arXiv, runner, checks, SQLite store, reports, provenance. |
| `scripts/` | Fixture capture from the real arXiv API, and an MCP smoke check. |

## 3. The tool-calling round trip

The agent uses Ollama's **native tool calling**: tool definitions go in the request's `tools` field, and the model answers with structured `tool_calls`, not with text to be parsed.

```
{"role": "system", "content": "You are a careful assistant..."}
{"role": "user", "content": "What is 17 * 23 + 4?"}
{"role": "assistant", "content": "", "tool_calls": [
    {"function": {"name": "calculator", "arguments": {"expression": "17 * 23 + 4"}}}]}
{"role": "tool", "tool_name": "calculator", "content": "17 * 23 + 4 = 395"}
{"role": "assistant", "content": "17 * 23 + 4 = 395."}
```

1. **Schemas.** `ToolRegistry.schemas()` returns each tool in the OpenAI function format: `{"type": "function", "function": {name, description, parameters}}`. `parameters` is the Pydantic model's JSON schema with the auto-generated `title` keys removed. The description is the main prompt-engineering surface: it tells the model when and how to call the tool.
2. **Request.** The loop sends the full history plus the schemas. The assistant message is appended to the history **verbatim**, tool calls included, so the model can relate the following tool messages to its own request.
3. **Execution.** For each tool call, in order, the registry is the trust boundary between model output and Python code. Whatever the model sends (unknown tool, arguments that are not JSON, wrong types, extra fields), `execute()` returns a `ToolResult` and never raises. Pydantic validation errors list each faulty field and repeat the expected schema.
4. **Observation.** Exactly one `tool` message per call, in call order, linked by `tool_name` (Ollama's format has no call id). Errors are observations too: they start with `Error:` so the model cannot mistake them for data, and carry an `error_type` in metadata for logs and evaluation.
5. **Output cap.** Every result, error messages included, is capped at 4000 characters with a visible `[truncated: showing N of M characters]` marker. The cap is idempotent, so a result already capped by an MCP server is not cut again by the client.

## 4. The agent loop and its guardrails

```
deadline left?  --no-->  TIMEOUT
call the LLM with the tool schemas   --LLMError / LLM timeout-->  LLM_ERROR (or TIMEOUT)
append the assistant message (verbatim, with its tool_calls)
no tool call?   --> FINAL_ANSWER
last iteration? --> do not execute the tools, MAX_ITERATIONS
for EACH tool call, in order: exactly one `tool` message
    new call              -> execute with a timeout (errors and timeouts are observations)
    repeated once         -> not executed, warning observation
    repeated again        -> REPETITION
too many consecutive tool errors? --> TOO_MANY_TOOL_ERRORS
repeat
```

### Stop reasons

Every run ends with exactly one `StopReason`: `final_answer`, `max_iterations`, `timeout`, `repetition`, `llm_error` or `too_many_tool_errors`. `final_answer` is set only when the model actually answered; no extra LLM call is made to force an answer.

### Limits (`AgentLimits` defaults)

| Limit | Default | Effect |
|---|---|---|
| `max_iterations` | 8 | Maximum LLM calls. At the last iteration, requested tools are answered with "not executed". |
| `llm_timeout_s` | 120 s | Per LLM call; covers a cold model load. Capped by the remaining total budget. |
| `tool_timeout_s` | 10 s | Per tool call, enforced by a worker thread. |
| `total_timeout_s` | 300 s | Wall-clock budget for the whole run. |
| `max_repeated_calls` | 1 | Identical repeats of a successful call answered with a warning before the run stops. |
| `max_consecutive_tool_errors` | 3 | Consecutive failed tool calls before the run stops. |

### Repetition policy

Arguments are normalized (key order and whitespace do not matter; values are not rewritten). Only successful calls and earlier warnings count as repeats: retrying a call that failed is legitimate and is bounded by the error budget instead. With the default, the first identical call runs, the second gets a warning observation pointing to the earlier result, and the third stops the run.

### Invariants

- **It always terminates.** Every iteration either makes an LLM call, which increments the iteration counter (bounded by `max_iterations`), or returns a stop reason. The deadline is checked before each LLM call and before each tool call.
- **It always sets a stop reason**, in one place (`_finish`), the single exit of `run_agent`.
- **It never raises for expected failures.** `LLMError`, timeouts and every tool failure become a stop reason or an observation. Programming errors (a bug in an adapter, an invalid configuration) still raise, on purpose.
- **The history stays well formed.** Every tool call gets exactly one `tool` message, including calls skipped because the run is stopping.
- **One mutable object.** `AgentState` is mutated only by the loop's main thread. Worker threads compute a return value from a snapshot of the messages and never touch the state.
- **Guards are pure functions**, unit-tested with explicit values; tests inject the clock and never sleep.

### Known limitation of timeouts

Python cannot kill a thread. When a timeout expires the loop stops *waiting*, but the call keeps running in its daemon thread until it returns. The arXiv client has its own HTTP timeout (7 s) below the tool timeout, so a slow API normally surfaces as a clean error observation. Killable isolation would require running tools in a subprocess.

## 5. arXiv tools

- **`search_arxiv`** returns compact summaries (id, year, primary category, title, first author, abstract snippet) within a 3500-character budget, followed by a hint to call `get_paper_details`. The query is plain keywords; quoted multi-word concepts become exact phrases; common words are dropped and reported.
- **`get_paper_details`** returns the full record of one paper: authors, dates, categories, PDF URL and full abstract. The id is validated before any request.
- **`ArxivClient`** follows the arXiv API Terms of Use: request starts spaced by at least 3 s, at most one request in flight, one connection, and a one-hour in-memory cache for identical requests. A 429 (or a 5xx with `Retry-After`) blocks further requests for that long. All failures are typed (`ArxivError` subclasses); the tool layer turns each into a message telling the model what to do next (retry once, stop retrying, use a valid id...).

## 6. The MCP bridge

The same registry can be served over the Model Context Protocol (stdio transport), and the agent can consume it as a client with `--tools mcp`. The agent loop is identical in both modes; only where the tools execute changes.

```
agent main thread        agent worker thread          bridge thread (event loop)       server process
run_agent
 _execute_tool ──────▶ registry.execute
                         McpToolProvider.execute
                           bridge.call_tool ───────▶ session.call_tool
                                                       JSON-RPC ──stdin──▶ tools/call
                                                                            registry.execute
                                                                            (validation, tool, cap)
                                                       ◀──stdout── CallToolResult{text, isError, _meta}
                         from_call_tool_result ◀──────
 ◀── ToolResult
```

### Server (`mcp/server.py`)

- Exactly two handlers: `tools/list` and `tools/call`. The listing is built from `registry.schemas()`, so the server advertises exactly what the local registry would send to the LLM.
- The server **validates every call itself** (inside `registry.execute`), because any MCP client may connect. Bad input never raises: unknown tools, invalid arguments and tool exceptions come back as `isError` results.
- Tool execution runs in a worker thread, so the event loop keeps handling pings and cancellations.
- `--registry-factory module:function` selects the registry to serve. The evaluation harness and the tests use it to serve tools on a fake arXiv and a frozen clock without any test code in the server.
- **stdout carries only protocol messages.** Logs and argparse errors go to stderr, and tools never print.

### Client (`mcp/client.py`, `mcp/provider.py`)

- `McpClientBridge` owns a background thread running an asyncio event loop. One long-lived task enters and exits the SDK's `stdio_client` and `ClientSession` contexts (anyio requires a cancel scope to be exited by the task that entered it); each request is a separate short task. The agent's threads submit coroutines and block on the result.
- The session is **persistent**: one server process per agent run, not one per call, so the arXiv client's rate limiter and cache survive across calls.
- `McpToolProvider` rebuilds the schemas from `tools/list` with the same `tool_schema` builder and re-applies Pydantic's canonical key order, so the schema bytes the LLM sees are identical to the local ones (checked by a contract test).
- Failures map to observations: timeout → `timeout`, server death → `mcp_server_died` (the bridge is then marked broken and every later call fails fast), any other MCP error → `mcp_error`. A failure to start raises `McpStartupError` **before** the loop runs (the CLI exits with code 2).
- The MCP call timeout is 0.5 s shorter than the loop's tool timeout, so the bridge times out first and cancels the request cleanly.
- What crosses the protocol: content and ok/`isError` always; `error_type` through the result's `_meta`; truncation info is recomputed from the cap marker. The server's own timing is not transmitted; the client measures the round trip, which is the latency the agent pays.
- The subprocess is always terminated: `close()` is registered before `start()` (an `ExitStack` in the CLI, `finally` in the harness), and the SDK's shutdown closes stdin, waits, then kills the process tree.

## 7. The evaluation harness

`evals/` runs a suite of cases against the **unchanged** `run_agent` and stores every run in SQLite. It needs no network other than the local Ollama server.

| Module | Role |
|---|---|
| `suite.py` | Case schema (Pydantic, `extra="forbid"`, frozen), suite loading with precise error locations, suite hashing. |
| `arxiv_fake.py` | `CaseArxivTransport`: serves recorded fixtures by rule. Any search gets the case's feed cut to `max_results`; a lookup gets the mapped file, else the entry from the search feed, else arXiv's real "not found" body. `failure_mode` (`http_503`, `timeout`) makes every request fail. |
| `runner.py` | `EvalConfig`, `warm_up`, `run_case`, `run_batch`. Builds each case's world (fake arXiv, frozen clock), runs the agent locally or through an MCP server (one server per run), and evaluates the checks. |
| `checks.py` | Pure check functions `(trace, **params) -> (passed, detail)` and their registry. |
| `records.py` | Frozen records shared by the runner, checks, store and report. |
| `store.py` | SQLite schema (`batches`, `runs`, `check_results`), versioned with in-place migration. |
| `report.py` | Summaries, Markdown reports and batch comparisons. |
| `provenance.py` | Tool-schema hash, Ollama version and model digest, git state. |
| `run.py` | The `evals.run` command line. |

### Cases and checks

A case declares a question, the fixtures its fake arXiv serves, an optional frozen date, the checks to run and the expected stop reason. The shipped suite (`evals/cases/minimal.json`) has 18 cases across single-tool, multi-step, tool-choice, error and edge categories, including one case served by a hand-written fixture with an indirect prompt injection in an abstract (tagged `synthetic`).

Checks are either **deterministic** (tool trace, arguments, cited arXiv ids versus ids returned or opened by tools, iteration and error counts, stop reason, unambiguous answer regexes) or **heuristic** (regexes over free wording). Heuristic results are reported separately and never count towards `passed`.

### Invariants

- **One run, one record.** `run_case` catches every ordinary exception, in the agent or in the harness, and returns a record with `harness_error`. Ctrl-C is not caught; finished runs are already stored.
- **Checks are pure**: no clock, no I/O, no mutation. Malformed check declarations are rejected when the suite loads, before any model runs.
- **A fixture miss is never an agent failure, and never silent.** A request no fixture covers is answered with an empty feed and recorded as a miss; the run is excluded from pass counts, listed in an "Inconclusive runs" report section, and above 10 % inconclusive runs `evals.run` exits with code 3. Cases built around a lookup can declare `default_search: "empty"` so a fallback search is judged rather than excluded.
- **The agent is not modified for evaluation.** The harness uses injection points only: an arXiv transport, a frozen clock, `ChatOptions.seed`, and named prompt variants.
- **Batches are comparable.** Each batch stores its full configuration plus identity hashes:
  - `system_prompt_sha256`: hash of the system prompt text;
  - `suite_sha256`: hash over the cases file and the fixtures' `MANIFEST.json`;
  - `tool_schemas_sha256`: hash of the canonical JSON of the ordered tool schemas actually sent to the LLM, captured at the client boundary (`SchemaRecorder`), so it reflects provider aggregation, subsetting and the MCP rebuild;
  - Ollama server version, model digest, and git commit (with `+dirty` for uncommitted changes; `--require-clean` refuses to run on a dirty tree).

  Comparison mode warns about any difference beyond the intentionally varied setting.
- **Measurement hygiene.** One untimed warm-up call (same options as the batch) loads the model; runs are repetition-major, so slow drift is spread across cases; each repetition uses its index as seed unless a fixed seed is given.

## 8. Testing

- Unit tests (`uv run pytest`) need no network and no Ollama: the LLM is a scripted fake, arXiv is served from recorded fixtures, and clocks are injected.
- Tests marked `mcp` start real MCP server subprocesses (local only).
- Tests marked `integration` hit real services (Ollama with the configured model, or the arXiv API) and are opt-in: `uv run pytest -m integration`.
