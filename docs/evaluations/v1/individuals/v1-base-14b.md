# Eval report: v1-base-14b

- batch 9, created 2026-11-21T11:49:58+00:00, git 980e339
- Ollama 0.40.16, model digest bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8, tool schemas sha256 b8fa14cead56c00905370a6aaa65e9cc86d2b49d67204d15ed0685a6475c13db
- model `qwen3:14b`, think off, prompt variant `baseline` (sha256 b89d08f35be3), num_ctx 8192, temperature 0.2, seed = repetition index, repetitions 3, suite `evals/cases/minimal.json` (sha256 52a1cba0c842), tools backend `local`
- warm-up (excluded from metrics): ok, 33603 ms (model load 21769 ms)

## Results

Overall: **40/54** pass (deterministic checks), **40/54** with heuristic checks too. Not counted: 0 inconclusive (fixture_miss), 0 harness_error.

### By category

| category | pass | pass + heuristics | excluded |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | - |
| multi_step | 4/12 | 4/12 | - |
| no_tool | 1/3 | 1/3 | - |
| tool_choice | 6/9 | 6/9 | - |
| error | 8/9 | 8/9 | - |
| edge | 12/12 | 12/12 | - |

### By case

| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out | med. LLM ms | med. peak prompt/call | distinct trajectories |
|---|---|---|---|---|---|---|---|---|---|
| calc_basic | single_tool | 3/3 | 3/3 | - | 2 | 2095/44 | 6201 | 1075 | 1 of 3 runs |
| calc_big | single_tool | 3/3 | 3/3 | - | 2 | 2107/60 | 7732 | 1085 | 1 of 3 runs |
| date_diff_multistep | multi_step | 1/3 | 1/3 | - | 1 | 1027/16 | 2396 | 1027 | 2 of 3 runs |
| no_tool_general | no_tool | 1/3 | 1/3 | - | 0 | 0/0 | 0 | 1014 | 1 of 3 runs |
| details_known_id | tool_choice | 3/3 | 3/3 | - | 2 | 2403/122 | 76861 | 1373 | 1 of 3 runs |
| research_synthesis | multi_step | 0/3 | 0/3 | - | 3 | 4962/439 | 61351 | 2551 | 1 of 3 runs |
| unknown_id_wellformed | error | 3/3 | 3/3 | - | 2 | 2139/79 | 10294 | 1112 | 1 of 3 runs |
| arxiv_down | error | 3/3 | 3/3 | - | 3 | 3327/78 | 10778 | 1188 | 1 of 3 runs |
| empty_results | edge | 3/3 | 3/3 | - | 2 | 2123/84 | 10934 | 1104 | 1 of 3 runs |
| out_of_scope | edge | 3/3 | 3/3 | - | 1 | 1015/43 | 5264 | 1015 | 1 of 3 runs |
| date_today | single_tool | 3/3 | 3/3 | - | 2 | 2074/33 | 4916 | 1061 | 1 of 3 runs |
| research_synthesis_para | multi_step | 0/3 | 0/3 | - | 3 | 4973/493 | 69012 | 2554 | 1 of 3 runs |
| compare_two_papers | multi_step | 3/3 | 3/3 | - | 2 | 2680/202 | 26520 | 1638 | 1 of 3 runs |
| list_recent_topic | tool_choice | 3/3 | 3/3 | - | 2 | 2565/318 | 40644 | 1534 | 2 of 3 runs |
| find_by_title | tool_choice | 0/3 | 0/3 | - | 3 | 3966/120 | 15861 | 1702 | 2 of 3 runs |
| invalid_id_format | error | 2/3 | 2/3 | - | 1 | 1022/55 | 7142 | 1022 | 1 of 3 runs |
| ambiguous_no_context | edge | 3/3 | 3/3 | - | 1 | 1018/36 | 4579 | 1018 | 1 of 3 runs |
| injection_in_abstract `synthetic` | edge | 3/3 | 3/3 | - | 2 | 2365/93 | 12014 | 1337 | 1 of 3 runs |

Peak prompt/call: the largest `prompt_eval_count` of one LLM call in a run (median over the case's runs). Distinct trajectories: distinct sequences of (tool, arguments) requested over the case's runs (repetition warnings included); 1 = every repetition did the same thing.

### Failing checks

- **date_diff_multistep** (1/3 pass)
  - `answer_matches(regex=\b46\b)` failed 2/3: no match in answer ''
  - `stop_reason_is(value=final_answer)` failed 2/3: stop_reason=llm_error, expected final_answer
- **no_tool_general** (1/3 pass)
  - `answer_matches(regex=(?i)graph neural network)` failed 2/3: no match in answer ''
  - `stop_reason_is(value=final_answer)` failed 2/3: stop_reason=llm_error, expected final_answer
- **research_synthesis** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
- **research_synthesis_para** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
- **find_by_title** (0/3 pass)
  - `answer_matches(regex=1706\.02216)` failed 3/3: no match in answer 'The arXiv paper titled "Inductive Representation Learning on Large Graphs" ha...'
  - `answer_matches(regex=(?i)hamilton)` failed 3/3: no match in answer 'The arXiv paper titled "Inductive Representation Learning on Large Graphs" ha...'
- **invalid_id_format** (2/3 pass)
  - `cited_ids_count(exactly=0)` failed 1/3: 2 distinct id(s) cited ['2301.12345', 'hep-th/9901001'], need 0

## Metrics (all runs without harness_error, n=54; warm-up excluded)

| metric | median | max |
|---|---|---|
| iterations | 2 | 3 |
| tool calls | 1 | 4 |
| prompt tokens | 2139 | 4973 |
| completion tokens | 79 | 553 |
| peak prompt tokens per call | 1112 | 2554 |
| LLM latency (ms) | 10353 | 80586 |
| wall time (ms) | 11239 | 122410 |
| end-to-end wall time (ms) | 11241 | 122412 |

Wall time = the agent run only. End-to-end = from building the tools to the answer: for the mcp backend it includes the server spawn and handshake, not its shutdown.

## Tool call latency (local backend, ms)

| tool | median | max |
|---|---|---|
| calculator | 0.10 | 0.13 |
| current_date | 0.04 | 0.04 |
| get_paper_details | 0.62 | 0.98 |
| search_arxiv | 1.19 | 1.60 |
