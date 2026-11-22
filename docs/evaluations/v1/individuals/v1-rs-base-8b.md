# Eval report: v1-rs-base-8b

- batch 6, created 2026-11-21T11:26:34+00:00, git 980e339
- Ollama 0.40.16, model digest 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41, tool schemas sha256 b8fa14cead56c00905370a6aaa65e9cc86d2b49d67204d15ed0685a6475c13db
- model `qwen3:8b`, think off, prompt variant `baseline` (sha256 b89d08f35be3), num_ctx 8192, temperature 0.2, seed = repetition index, repetitions 10, suite `evals/cases/minimal.json` (sha256 52a1cba0c842), tools backend `local`
- warm-up (excluded from metrics): ok, 131 ms (model load 9 ms)

## Results

Overall: **0/20** pass (deterministic checks), **0/20** with heuristic checks too. Not counted: 0 inconclusive (fixture_miss), 0 harness_error.

### By category

| category | pass | pass + heuristics | excluded |
|---|---|---|---|
| multi_step | 0/20 | 0/20 | - |

### By case

| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out | med. LLM ms | med. peak prompt/call | distinct trajectories |
|---|---|---|---|---|---|---|---|---|---|
| research_synthesis | multi_step | 0/10 | 0/10 | - | 2 | 2409/318 | 4669 | 1374 | 1 of 10 runs |
| research_synthesis_para | multi_step | 0/10 | 0/10 | - | 2 | 2419/434 | 6232 | 1379 | 1 of 10 runs |

Peak prompt/call: the largest `prompt_eval_count` of one LLM call in a run (median over the case's runs). Distinct trajectories: distinct sequences of (tool, arguments) requested over the case's runs (repetition warnings included); 1 = every repetition did the same thing.

### Failing checks

- **research_synthesis** (0/10 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 10/10: search_arxiv.max_results values: [3], need one >= 6
  - `tool_called(tool=get_paper_details, min=2)` failed 10/10: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 10/10: cited ids never opened with get_paper_details: ['2603.15047', '2605.07838', '2606.01781']
- **research_synthesis_para** (0/10 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 10/10: search_arxiv.max_results values: [3], need one >= 6
  - `tool_called(tool=get_paper_details, min=2)` failed 10/10: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 10/10: cited ids never opened with get_paper_details: ['2603.15047', '2605.07838', '2606.01781']

## Metrics (all runs without harness_error, n=20; warm-up excluded)

| metric | median | max |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2414 | 2419 |
| completion tokens | 378 | 468 |
| peak prompt tokens per call | 1376 | 1379 |
| LLM latency (ms) | 6021 | 8240 |
| wall time (ms) | 6023 | 8242 |
| end-to-end wall time (ms) | 6024 | 8247 |

Wall time = the agent run only. End-to-end = from building the tools to the answer: for the mcp backend it includes the server spawn and handshake, not its shutdown.

## Tool call latency (local backend, ms)

| tool | median | max |
|---|---|---|
| search_arxiv | 1.18 | 1.57 |
