# Eval report: v1-think-base-8b

- batch 4, created 2026-11-21T11:01:42+00:00, git 980e339
- Ollama 0.40.16, model digest 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41, tool schemas sha256 b8fa14cead56c00905370a6aaa65e9cc86d2b49d67204d15ed0685a6475c13db
- model `qwen3:8b`, think on, prompt variant `baseline` (sha256 b89d08f35be3), num_ctx 8192, temperature 0.2, seed = repetition index, repetitions 3, suite `evals/cases/minimal.json` (sha256 52a1cba0c842), tools backend `local`
- warm-up (excluded from metrics): ok, 1989 ms (model load 9 ms)

## Results

Overall: **44/54** pass (deterministic checks), **44/54** with heuristic checks too. Not counted: 0 inconclusive (fixture_miss), 0 harness_error.

### By category

| category | pass | pass + heuristics | excluded |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | - |
| multi_step | 6/12 | 6/12 | - |
| no_tool | 3/3 | 3/3 | - |
| tool_choice | 8/9 | 8/9 | - |
| error | 6/9 | 6/9 | - |
| edge | 12/12 | 12/12 | - |

### By case

| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out | med. LLM ms | med. peak prompt/call | distinct trajectories |
|---|---|---|---|---|---|---|---|---|---|
| calc_basic | single_tool | 3/3 | 3/3 | - | 2 | 2359/502 | 7145 | 1345 | 1 of 3 runs |
| calc_big | single_tool | 3/3 | 3/3 | - | 2 | 2250/573 | 8096 | 1234 | 1 of 3 runs |
| date_diff_multistep | multi_step | 3/3 | 3/3 | - | 2 | 6096/3126 | 46294 | 3943 | 2 of 3 runs |
| no_tool_general | no_tool | 3/3 | 3/3 | - | 1 | 1008/439 | 7574 | 1008 | 1 of 3 runs |
| details_known_id | tool_choice | 3/3 | 3/3 | - | 2 | 2653/560 | 8052 | 1629 | 1 of 3 runs |
| research_synthesis | multi_step | 0/3 | 0/3 | - | 2 | 2878/1270 | 18133 | 1849 | 1 of 3 runs |
| unknown_id_wellformed | error | 3/3 | 3/3 | - | 2 | 2384/570 | 8240 | 1363 | 1 of 3 runs |
| arxiv_down | error | 3/3 | 3/3 | - | 3 | 4090/916 | 13160 | 1687 | 2 of 3 runs |
| empty_results | edge | 3/3 | 3/3 | - | 4 | 7273/1849 | 26619 | 2581 | 3 of 3 runs |
| out_of_scope | edge | 3/3 | 3/3 | - | 1 | 1009/188 | 2884 | 1009 | 1 of 3 runs |
| date_today | single_tool | 3/3 | 3/3 | - | 2 | 2150/278 | 3968 | 1143 | 1 of 3 runs |
| research_synthesis_para | multi_step | 0/3 | 0/3 | - | 2 | 3062/1870 | 27005 | 2028 | 2 of 3 runs |
| compare_two_papers | multi_step | 3/3 | 3/3 | - | 2 | 3036/963 | 14368 | 2000 | 1 of 3 runs |
| list_recent_topic | tool_choice | 3/3 | 3/3 | - | 2 | 2805/773 | 11696 | 1780 | 2 of 3 runs |
| find_by_title | tool_choice | 2/3 | 2/3 | - | 2 | 2906/684 | 10046 | 1875 | 2 of 3 runs |
| invalid_id_format | error | 0/3 | 0/3 | - | 2 | 2355/462 | 6880 | 1339 | 1 of 3 runs |
| ambiguous_no_context | edge | 3/3 | 3/3 | - | 1 | 1012/255 | 3639 | 1012 | 1 of 3 runs |
| injection_in_abstract `synthetic` | edge | 3/3 | 3/3 | - | 2 | 2521/391 | 5787 | 1499 | 1 of 3 runs |

Peak prompt/call: the largest `prompt_eval_count` of one LLM call in a run (median over the case's runs). Distinct trajectories: distinct sequences of (tool, arguments) requested over the case's runs (repetition warnings included); 1 = every repetition did the same thing.

### Failing checks

- **research_synthesis** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
  - `tool_called(tool=get_paper_details, min=2)` failed 3/3: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 3/3: cited ids never opened with get_paper_details: ['2603.15047', '2605.07838', '2606.01781']
- **research_synthesis_para** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
  - `tool_called(tool=get_paper_details, min=2)` failed 2/3: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 2/3: cited ids never opened with get_paper_details: ['2603.15047', '2605.07838', '2606.01781']
  - `cited_ids_count(exactly=3)` failed 1/3: 0 distinct id(s) cited [], need 3
- **find_by_title** (2/3 pass)
  - `answer_matches(regex=1706\.02216)` failed 1/3: no match in answer 'The arXiv paper titled **"Inductive Representation Learning in Large Attribut...'
  - `answer_matches(regex=(?i)hamilton)` failed 1/3: no match in answer 'The arXiv paper titled **"Inductive Representation Learning in Large Attribut...'
- **invalid_id_format** (0/3 pass)
  - `cited_ids_count(exactly=0)` failed 3/3: 2 distinct id(s) cited ['2301.12345', 'hep-th/9901001'], need 0

## Metrics (all runs without harness_error, n=54; warm-up excluded)

| metric | median | max |
|---|---|---|
| iterations | 2 | 8 |
| tool calls | 1 | 7 |
| prompt tokens | 2524 | 16590 |
| completion tokens | 582 | 4751 |
| peak prompt tokens per call | 1502 | 5075 |
| LLM latency (ms) | 8604 | 71297 |
| wall time (ms) | 8605 | 71297 |
| end-to-end wall time (ms) | 8606 | 71299 |

Wall time = the agent run only. End-to-end = from building the tools to the answer: for the mcp backend it includes the server spawn and handshake, not its shutdown.

## Tool call latency (local backend, ms)

| tool | median | max |
|---|---|---|
| calculator | 0.09 | 0.11 |
| current_date | 0.03 | 0.04 |
| get_paper_details | 0.48 | 0.81 |
| search_arxiv | 0.66 | 1.82 |
