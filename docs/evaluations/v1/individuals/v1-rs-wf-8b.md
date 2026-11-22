# Eval report: v1-rs-wf-8b

- batch 7, created 2026-11-21T11:28:29+00:00, git 980e339
- Ollama 0.40.16, model digest 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41, tool schemas sha256 b8fa14cead56c00905370a6aaa65e9cc86d2b49d67204d15ed0685a6475c13db
- model `qwen3:8b`, think off, prompt variant `research_workflow` (sha256 ac6259a046bd), num_ctx 8192, temperature 0.2, seed = repetition index, repetitions 10, suite `evals/cases/minimal.json` (sha256 52a1cba0c842), tools backend `local`
- warm-up (excluded from metrics): ok, 151 ms (model load 10 ms)

## Results

Overall: **1/20** pass (deterministic checks), **1/20** with heuristic checks too. Not counted: 0 inconclusive (fixture_miss), 0 harness_error.

### By category

| category | pass | pass + heuristics | excluded |
|---|---|---|---|
| multi_step | 1/20 | 1/20 | - |

### By case

| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out | med. LLM ms | med. peak prompt/call | distinct trajectories |
|---|---|---|---|---|---|---|---|---|---|
| research_synthesis | multi_step | 1/10 | 1/10 | - | 3 | 5228/540 | 8836 | 2639 | 3 of 10 runs |
| research_synthesis_para | multi_step | 0/10 | 0/10 | - | 3 | 5243/530 | 8347 | 2644 | 1 of 10 runs |

Peak prompt/call: the largest `prompt_eval_count` of one LLM call in a run (median over the case's runs). Distinct trajectories: distinct sequences of (tool, arguments) requested over the case's runs (repetition warnings included); 1 = every repetition did the same thing.

### Failing checks

- **research_synthesis** (1/10 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 8/10: search_arxiv.max_results values: [3], need one >= 6
  - `cited_ids_count(exactly=3)` failed 1/10: 0 distinct id(s) cited [], need 3
- **research_synthesis_para** (0/10 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 10/10: search_arxiv.max_results values: [3], need one >= 6

## Metrics (all runs without harness_error, n=20; warm-up excluded)

| metric | median | max |
|---|---|---|
| iterations | 3 | 5 |
| tool calls | 4 | 4 |
| prompt tokens | 5243 | 11493 |
| completion tokens | 532 | 570 |
| peak prompt tokens per call | 2644 | 3207 |
| LLM latency (ms) | 8440 | 9600 |
| wall time (ms) | 8444 | 9604 |
| end-to-end wall time (ms) | 8446 | 9606 |

Wall time = the agent run only. End-to-end = from building the tools to the answer: for the mcp backend it includes the server spawn and handshake, not its shutdown.

## Tool call latency (local backend, ms)

| tool | median | max |
|---|---|---|
| search_arxiv | 1.13 | 1.99 |
| get_paper_details | 0.62 | 0.83 |
