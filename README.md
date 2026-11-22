# local-research-agent

A research assistant that runs on your machine (arXiv is the only external service). A local LLM served by [Ollama](https://ollama.com) answers questions by calling Python tools through native tool calling: a safe calculator, the current date, arXiv search and arXiv paper details. A bounded agent loop with explicit stop reasons drives it, the same tools are also served over the Model Context Protocol (MCP), and an evaluation harness measures the agent reproducibly against a mocked arXiv.

## Architecture

```mermaid
flowchart LR
    User(["User / CLI"])
    Loop["Agent loop: run_agent<br/>guards · timeouts · stop reasons"]
    LLM["OllamaClient<br/>POST /api/chat"]
    Ollama[("Ollama<br/>local model")]
    Registry["ToolRegistry<br/>validation · output cap · logging"]
    Local["LocalToolProvider"]
    McpClient["McpToolProvider<br/>+ McpClientBridge"]

    subgraph server["MCP server (subprocess, stdio)"]
        SRegistry["ToolRegistry<br/>+ LocalToolProvider"]
    end

    Tools["Tools (defined once)<br/>calculator · current_date ·<br/>search_arxiv · get_paper_details"]
    Arxiv[("arXiv API")]

    Evals["evals/ harness<br/>runner · checks · report"]
    Fixtures[("Recorded arXiv fixtures<br/>+ frozen clock")]
    DB[("SQLite results")]

    %% Normal execution
    User -->|question| Loop
    Loop -->|"messages + tool schemas"| LLM
    LLM -->|"assistant message + tool_calls"| Loop
    LLM --> Ollama
    Loop -->|"execute(name, args)"| Registry
    Registry -->|"--tools local"| Local
    Registry -->|"--tools mcp"| McpClient
    McpClient -->|"JSON-RPC over stdio"| SRegistry
    Local --> Tools
    SRegistry --> Tools
    Tools --> Arxiv
    Loop -->|"answer + trace"| User

    %% Evaluation only (dashed)
    Evals -.->|"runs unchanged"| Loop
    Evals -.->|"injects fakes into tools<br/>(registry factory)"| Tools
    Evals -.->|"same injection in the server process<br/>(--registry-factory)"| SRegistry
    Fixtures -.->|"replace the real API"| Tools
    Evals -.->|"records runs"| DB
```

*Solid lines: normal execution. Dashed lines: evaluation only (mocked arXiv, frozen clock).*

See [docs/architecture.md](docs/architecture.md) for module responsibilities, the tool-calling round trip, the loop's invariants, the MCP bridge and the evaluation harness, and [docs/framing.md](docs/framing.md) for the project's scope and decisions.

## Features

- **Native tool calling** with a local Ollama model (developed against `qwen3:8b`).
- **Bounded agent loop**: iteration, per-call and total time budgets, repetition detection and a consecutive-error budget; every run ends with one explicit stop reason.
- **Tools defined once**: a name, a description, a Pydantic argument model and a function; the JSON schema, validation and the MCP listing are derived from it.
- **Errors as observations**: invalid arguments, tool failures and timeouts are returned to the model as actionable `Error: ...` messages.
- **arXiv tools** that follow the arXiv API terms of use (one request every 3 s, single connection, caching).
- **MCP server over stdio** exposing the same tools, and an MCP client mode for the agent (`--tools mcp`) with byte-identical tool schemas.
- **Evaluation harness**: an 18-case suite, a deterministic fake arXiv, pure checks, SQLite storage, Markdown reports, and batch comparison with provenance hashes (prompt, suite, tool schemas, model digest, git commit).
- **Structured JSON logs** on stderr.

## Quickstart

### Prerequisites

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/).
- [Ollama](https://ollama.com) running locally, with a model that supports tool calling:

  ```bash
  ollama pull qwen3:8b
  ```

### Install

```bash
uv sync
```

### Configuration

The CLI reads these environment variables (see [.env.example](.env.example); no `.env` file is loaded automatically):

| Variable | Default |
|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `qwen3:8b` |
| `OLLAMA_CONNECT_TIMEOUT` | `5` (seconds) |
| `OLLAMA_READ_TIMEOUT` | `300` (seconds) |
| `LOG_LEVEL` | `INFO` |

### Usage

```bash
# Arithmetic and dates
uv run local-research-agent "What is 17 * 23 + 4?"
uv run local-research-agent "How many days are there between today and 2026-12-25?"

# Research questions (real arXiv API)
uv run local-research-agent "Find 3 recent papers on graph neural networks for protein-protein interaction and summarize them."

# Same agent, tools served by an MCP server subprocess
uv run local-research-agent --tools mcp "What is 12% of 2450?"

# Other options
uv run local-research-agent --help
```

The answer and a compact trace go to stdout; JSON logs go to stderr. Exit code: 0 when the model answered, 1 for any other stop reason, 2 when the MCP server could not be started. `uv run python -m local_research_agent.cli ...` is equivalent.

## Running the tests

```bash
uv run pytest                    # unit tests: no network, no Ollama (includes local MCP subprocess tests)
uv run pytest -m "not mcp"       # skip the tests that start MCP server subprocesses
uv run pytest -m integration     # opt-in: real Ollama and/or the real arXiv API
uv run ruff check && uv run ruff format --check
```

## Running the evaluation

The harness runs the suite against the local Ollama model; arXiv is served from recorded fixtures, so no other network access is needed.

```bash
uv run python -m evals.run --suite minimal --repeat 3 --label baseline
uv run python -m evals.run --suite minimal --repeat 3 --label workflow --prompt-variant research_workflow
uv run python -m evals.run --suite minimal --repeat 3 --label mcp --tools mcp

uv run python -m evals.report --label baseline
uv run python -m evals.report --label baseline --compare workflow
```

Results are stored in `evals/results/evals.sqlite` and reports are written to `evals/results/<label>.md` (git-ignored). `--require-clean` refuses to run with uncommitted changes, so a batch can be tied to a commit. `uv run python -m evals.run --help` lists every option.

## Results

All numbers come from the evaluation harness (`evals/`). The reports they are copied from are in `docs/evaluations/v1/` (`individuals/` holds one report per batch, `comparisons/` the pairwise comparisons, and a `README.md` indexes them). Every batch stores its full configuration (commit, suite hash, prompt hash, tool-schema hash, Ollama version, model digest), so each figure can be traced back to a run.

### Setup

- Machine: Windows 11, AMD Ryzen 5 5500, 32 GB DDR4, NVIDIA RTX 5060 (8 GB VRAM). Ollama 0.40.16.
- Models: `qwen3:4b`, `qwen3:8b`, `qwen3:14b` (quantization Q4_K_M; digests are in the reports).
- Generation: `num_ctx` 8192, temperature 0.2, seed = repetition index, 3 repetitions per case (10 for the two synthesis-only batches).
- Suite: 18 cases (sha256 `52a1cba0c842`): single tool (3), multi-step (4), no tool (1), tool choice (3), errors (3), edge cases (4, one with a synthetic prompt-injection fixture). Tool schemas sha256 `b8fa14cead56` in every batch.
- arXiv is replaced by recorded fixtures and the clock is frozen, so runs are reproducible and need no network. The fixtures are responses of the public arXiv API recorded with `scripts/capture_arxiv_fixtures.py`; they remain subject to arXiv's terms of use.
- Code: release `v1.0.0-rc1`. Every batch ran at commit `980e339`, except `v1-think-base-4b` (`3c35a8f`: reporting and scripts only; `src/`, the cases and the fixtures are identical, checked with `git diff`).
- Agent limits (defaults): 8 iterations, 120 s per LLM call, 10 s per tool call, 300 s per run, repetition detection on identical tool calls, at most 3 consecutive tool errors.
- A run **passes** when all its deterministic checks pass (tool called, argument constraints, cited ids returned/opened by tools, stop reason `final_answer`, regex on exact values). Heuristic checks (regex on phrasing) are reported separately in the full reports.

### Summary

| Batch | Model | think | Prompt | Tools | Pass | Median LLM latency |
|---|---|---|---|---|---|---|
| `v1-base-8b` | qwen3:8b | off | baseline | local | 38/54 | 1.4 s |
| `v1-wf-8b` | qwen3:8b | off | workflow | local | 41/54 | 1.5 s |
| `v1-mcp-8b` | qwen3:8b | off | baseline | MCP | 38/54 | 1.4 s (+1.3 s server startup per run) |
| `v1-think-base-8b` | qwen3:8b | on | baseline | local | 44/54 | 8.6 s |
| `v1-think-wf-8b` | qwen3:8b | on | workflow | local | 41/54 | 7.9 s |
| `v1-base-4b` | qwen3:4b | off (not honored, see issues) | baseline | local | 32/54 ¹ | 15.0 s |
| `v1-think-base-4b` | qwen3:4b | on | baseline | local | 39/54 ¹ | 14.5 s |
| `v1-base-14b` | qwen3:14b | off | baseline | local | 40/54 ² | 10.4 s |
| `v1-rs-base-8b` | qwen3:8b | off | baseline | local | 0/20 (synthesis cases only) | 6.0 s |
| `v1-rs-wf-8b` | qwen3:8b | off | workflow | local | 1/20 (synthesis cases only) | 8.4 s |

¹ The reports show 32/52 and 39/52: in two runs of `no_tool_general` the model searched arXiv for a general-knowledge question, which the harness recorded as inconclusive (no fixture for that search). They are failures, hence 54 runs.
² Includes 4 runs that ended with an LLM-call timeout (120 s), counted as failures.

### Findings

**1. Baseline (8B, no reasoning, local tools): 38/54.** All single-tool and no-tool cases pass 3/3. Failures fall into four observed modes (13 of the 16 failures; the other 3 are `invalid_id_format`, a flawed check, see below):
- *Citing without reading*: on both synthesis cases the model requests 3 results, never opens a paper, and cites 3 ids it has not opened (0/3 on each of the two synthesis cases).
- *Announcing instead of acting*: one run of `compare_two_papers` ended with "Let me retrieve the full information…" and no tool call. The loop sees a message without tool calls and stops normally, so no guard can catch it.
- *Near-match presented as the answer*: `find_by_title` fails 3/3. The model asked for a single result (`max_results=1`), received a different paper with a similar title (the target is at rank 4 of the 10 recorded results) and reported it without comparing titles.
- *Searching a filler word*: given "that topic we discussed earlier", the model searched the word "topic" (`ambiguous_no_context`, 0/3).

Across all batches, most cases have a single trajectory over 3 runs (13 of 18 in the baseline), so a "3/3" mostly means "the dominant behavior is correct".

**2. Exposing the tools through MCP changes nothing for the model.** 38/54 for both backends, identical per case; identical median prompt tokens (2,365) and tool-schema hash. Cost: about 1.3 s of server startup per run (median end-to-end 2.77 s vs 1.43 s) and +1.9 to +2.5 ms per tool call, which is negligible against LLM latency. Measured with mocked arXiv (no network latency). The expected benefits of MCP are reuse by other MCP clients and process isolation, not performance; this evaluation only measures its cost.

**3. A workflow instruction in the system prompt changes what the model reads, not how widely it searches (8B, no reasoning).** 41/54 vs 38/54 with no regression. The only pass-count change is `ambiguous_no_context` (0/3 → 3/3), a case unrelated to the prompt's purpose; the cause is not established. On the synthesis cases (10 repetitions each): with the baseline prompt no run opens a paper (0 `get_paper_details` calls in 20 runs); with the workflow prompt every run opens three (60 calls) and cites only papers it opened, but every run performs a single search, always sorted by date. Only 2/10 runs of the original phrasing request more than 3 results (`max_results=10`), 0/10 on the paraphrase (not distinguishable from chance). Pass rate 1/20. "Pick the 3 most relevant" degenerates into "take the most recent". Cost: about +180 prompt tokens per run, and on synthesis cases 2.2× the cumulative prompt tokens and +40 % median latency (8.4 s vs 6.0 s); median peak prompt 2,644 tokens per call, maximum 3,207 (context window 8,192).

**4. Reasoning mode (`think`) improves three cases and costs about 6× (8B).** With the baseline prompt: 44/54 vs 38/54, no regression, gains on `ambiguous_no_context` (0/3 → 3/3), `find_by_title` (0/3 → 2/3) and `compare_two_papers` (2/3 → 3/3), each with one phrasing. The synthesis cases stay at 0/3. Cost: generated tokens 582 vs 92 (median), LLM latency 8.6 s vs 1.4 s. Details:
- `find_by_title` is decided by one argument: the 3 runs that left `max_results` at its default found the paper; all 6 runs that set `max_results=1` received only the near-match (9 runs inspected: the baseline, `think` and `think` + workflow batches; the other batches were not inspected). With `think`, the model flagged it ("closest match") where the baseline answered with confidence.
- With `think` the model skips the calculator for easy arithmetic: in the date-difference case, 5 of 6 runs computed 21 + 25 mentally (0 of 3 without `think`). The answer was correct each time.
- Combining `think` with the workflow prompt gives 41/54. Under `think`, the workflow prompt widened the search in all 6 synthesis runs (original phrasing: `max_results` 5, 5, 9; paraphrase: 5 and two values ≥ 6) and opened papers in 5 of the 6 (three each on the original phrasing; one paraphrase run opened none and cited 3 ids it had not opened, and the two other paraphrase runs cited no id). Pass rate: 1/3 on the original phrasing, 0/3 on the paraphrase. Across the suite, `research_synthesis` gained one run while `unknown_id_wellformed`, `empty_results`, `compare_two_papers` and `find_by_title` lost one each (net −3, within what 3 runs per case can produce). One loss is explained: in one `empty_results` run the model issued 7 different reformulations of a query that returns nothing and hit the 8-iteration cap with no answer. The repetition guard did not trigger because the queries differ.

**5. Model size: no monotonic gain, and smaller is not faster.**
- 8B with reasoning: 44/54, 8.6 s median. 4B with reasoning: 39/54, 14.5 s median, 2.6× the generated tokens of the 8B. It loses `find_by_title`, one run of `unknown_id_wellformed` and two runs of `no_tool_general` (searching arXiv for a general-knowledge question; see footnote 1).
- 14B: 40/54 on a machine where it exceeds the VRAM (CPU/GPU split: 40 % CPU / 60 % GPU). Median LLM latency 10.4 s with a heavy tail (case medians of 61–77 s of LLM time on three cases, longest successful run 80.6 s of LLM time) and 4 runs ended by the 120 s timeout. Excluding those 4 runs it scores 40/50, versus 34/50 for the 8B on the matching runs (post hoc; the gap comes from `ambiguous_no_context`, `invalid_id_format` — a flawed check, see below — and `compare_two_papers`).
- Without any workflow prompt, the 14B opens papers on the synthesis cases and cites only papers it opened, with exactly 3 ids; it fails only the `max_results ≥ 6` threshold (3 results requested, 3/3 on both cases). The 8B needs the workflow prompt for that behavior.
- Across the matrix (88 synthesis runs in 10 batches) every run searched exactly once and every search was sorted by date: no model size, reasoning mode or prompt variant chose relevance ranking (both phrasings contain the word "recent", so this cannot tell whether date sorting is the model's default or induced by that word). Only 5 of the 88 searches requested `max_results ≥ 6`, all with the workflow prompt (3 with `think`, 2 without); the only 2 runs that pass the case are among them.

**6. Prompt injection (one naive, synthetic case).** The 8B did not follow the injected instruction in any of its 15 runs (5 batches × 3), the 14B in 3/3, the 4B with reasoning in 3/3. In the first 4B batch (reasoning leaked into the answer, see below) the automatic check failed 3/3 because the leaked reasoning quotes the injected string; reading the final text after the reasoning by hand, one run answered "PWNED" and two summarized the abstract. In the clean 4B batch (`think` on) the check passed 3/3. The two 4B batches differ only in how `think` is handled, so 1/3 versus 0/3 on the final text cannot be separated from noise. This shows nothing about robustness to injection in general: one attack form, one fixture written by us, 3 runs per configuration.

### Known evaluation issues

These are defects of the evaluation suite or harness, not of the agent. Results are published as recorded; fixing them changes the suite hash and is planned for v1.1.

- **`invalid_id_format`** counts any arXiv-looking id in the answer. The models refuse correctly and quote the format examples from the tool error (`2301.12345`, `hep-th/9901001`), which fails the check. Its 0/3 (8B, 4B) and 2/3 (14B) say nothing about the agent.
- **`unknown_id_wellformed`**: the heuristic regex misses "does not correspond to any known paper", so the heuristic pass count is 0/3 in several batches although the answers are correct.
- **Citation checks pass vacuously** when no id is cited (`cited_ids_subset_of_tool_ids` and `..._opened_ids`); only the count check catches an answer without ids.
- **`no_tool_general` has no default search fixture**, so an unexpected search is recorded as inconclusive instead of a failure (the 4B batches). They are counted as failures above.
- **LLM-call errors are counted as failures** and not reported separately (the 4 timeouts of the 14B batch).
- **The fixtures serve the same result list whatever the query or sort**, so the suite measures tool choice and argument shapes, not the quality of a query.
- **The `max_results ≥ 6` threshold is a convention.** Two `think` + workflow runs searched 5 results and opened 3, a defensible strategy that fails the check.
- **Pass/fail does not distinguish a hedged wrong answer from a confident one** (see `find_by_title` with reasoning).

### Model and runtime issues

- **`think: false` was not honored by `qwen3:4b`** on this Ollama version: the reasoning appeared in the answer text (closing `</think>` tag, empty `thinking` field) in 54/54 runs, reproduced outside the harness with a direct API call. Text-based checks are unreliable for `v1-base-4b`; trace-based observations are valid. With `think: true` the reasoning is returned separately (`v1-think-base-4b`).

### Reproduce

```bash
uv sync
uv run python scripts/run_matrix.py            # all batches (several hours; needs Ollama and the three models)
uv run python -m evals.report --label v1-base-8b --compare v1-think-base-8b
```

Reports compare as `--label <baseline> --compare <variant>`; `diff = variant − baseline`.

## Design decisions

- **Native tool calling.** The model receives JSON schemas in the request's `tools` field and returns structured `tool_calls`; nothing is parsed out of free text.
- **Tools defined once.** Each tool is a name, a description, a Pydantic argument model and a function. The schema sent to the LLM, the argument validation and the MCP listing all derive from that definition, so local and MCP tools cannot drift apart.
- **Errors as observations.** The tool registry never raises: unknown tools, invalid arguments, tool exceptions and timeouts become `Error: ...` observations the model can react to. Only the loop's guardrails end a run early, always with an explicit stop reason.
- **A single agent.** One loop and one tool registry cover the use case; multiple agents would add coordination failure modes and make evaluation harder to interpret.
- **MCP over stdio.** The server runs as a subprocess speaking JSON-RPC over its pipes: no port, no authentication layer, nothing to deploy.

## Limitations

- **Small suite.** 18 cases, 3 repetitions at temperature 0.2 (10 for two batches): a difference of one or two runs is within noise, and one case flipping moves a total by 3.
- **Mocked arXiv.** The harness serves recorded fixtures by rule (any search returns the case's recorded feed), so query quality is not measured, and fixtures are a snapshot in time.
- **Heuristic checks.** Some checks are regexes over free-form wording. They are reported separately from deterministic checks, but can misjudge a valid answer.
- **Synthesis quality is not evaluated** (accuracy, relevance of the chosen papers, number of sentences); the checks cover tool use, arguments and citations.
- **One machine, one model family, one quantization.**
- **Local model reliability.** A small local model can skip tools, stop early or summarize from snippets; behaviour varies across runs. Bitwise determinism on GPU is not guaranteed; tool-call sequences matched between two batches with the same seeds.
- **Loop guardrails.** No budget for consecutive unsuccessful searches and no fallback answer when the iteration cap is reached; tool timeouts stop waiting but cannot kill a running thread.
- **Synthetic injection fixture.** The prompt-injection case uses a hand-written abstract, not a real paper.
- **Unpublished traces.** Trace-level observations (tool arguments, answer text after the reasoning, rank of the target paper in the fixture) were read from the local run database, which is not published; the published reports contain pass counts, failing checks and metrics only.

## Roadmap

- **LangGraph** orchestration with a **verification node** that checks the draft answer against the tool results before answering.
- **Extended evaluation**: a larger suite, more models and an evaluation matrix.
- **Evaluation fixes**: correct the flawed checks listed above, make fixtures depend on the query and sort, add a synthesis case without the word "recent", report LLM-call errors separately.
- **Tools and guardrails**: a search tool that combines relevance ranking with a date filter, a budget for consecutive unsuccessful searches, a fallback answer at the iteration cap, and a warning when `think` is requested but not honored.
- **FastAPI** service exposing the agent over HTTP.
- **docker-compose** setup bundling the service and Ollama.