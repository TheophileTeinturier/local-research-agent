# Eval report: v1-think-base-4b

- batch 10, created 2026-11-21T19:55:43+00:00, git 3c35a8f
- Ollama 0.40.16, model digest 359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7, tool schemas sha256 b8fa14cead56c00905370a6aaa65e9cc86d2b49d67204d15ed0685a6475c13db
- model `qwen3:4b`, think on, prompt variant `baseline` (sha256 b89d08f35be3), num_ctx 8192, temperature 0.2, seed = repetition index, repetitions 3, suite `evals/cases/minimal.json` (sha256 52a1cba0c842), tools backend `local`
- warm-up (excluded from metrics): ok, 3573 ms (model load 2438 ms)

## Results

Overall: **39/52** pass (deterministic checks), **39/52** with heuristic checks too. Not counted: 2 inconclusive (fixture_miss), 0 harness_error.

### By category

| category | pass | pass + heuristics | excluded |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | - |
| multi_step | 6/12 | 6/12 | - |
| no_tool | 1/1 | 1/1 | 2 inconclusive (fixture_miss) |
| tool_choice | 6/9 | 6/9 | - |
| error | 5/9 | 5/9 | - |
| edge | 12/12 | 12/12 | - |

### By case

| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out | med. LLM ms | med. peak prompt/call | distinct trajectories |
|---|---|---|---|---|---|---|---|---|---|
| calc_basic | single_tool | 3/3 | 3/3 | - | 2 | 2597/720 | 6815 | 1583 | 1 of 3 runs |
| calc_big | single_tool | 3/3 | 3/3 | - | 2 | 2421/729 | 6731 | 1405 | 1 of 3 runs |
| date_diff_multistep | multi_step | 3/3 | 3/3 | - | 2 | 4691/4285 | 41688 | 3670 | 1 of 3 runs |
| no_tool_general | no_tool | 1/1 | 1/1 | 2 inconclusive (fixture_miss) | 2 | 3489/2128 | 20491 | 2481 | 3 of 3 runs |
| details_known_id | tool_choice | 3/3 | 3/3 | - | 2 | 2848/1227 | 11565 | 1824 | 1 of 3 runs |
| research_synthesis | multi_step | 0/3 | 0/3 | - | 2 | 5760/4309 | 42241 | 4731 | 1 of 3 runs |
| unknown_id_wellformed | error | 2/3 | 2/3 | - | 2 | 2858/1609 | 15231 | 1837 | 1 of 3 runs |
| arxiv_down | error | 3/3 | 3/3 | - | 2 | 2772/1179 | 11153 | 1748 | 1 of 3 runs |
| empty_results | edge | 3/3 | 3/3 | - | 3 | 7202/3410 | 33068 | 3746 | 3 of 3 runs |
| out_of_scope | edge | 3/3 | 3/3 | - | 1 | 1009/269 | 2666 | 1009 | 1 of 3 runs |
| date_today | single_tool | 3/3 | 3/3 | - | 2 | 2331/815 | 7478 | 1324 | 1 of 3 runs |
| research_synthesis_para | multi_step | 0/3 | 0/3 | - | 2 | 4498/3541 | 34167 | 3464 | 2 of 3 runs |
| compare_two_papers | multi_step | 3/3 | 3/3 | - | 2 | 3817/1886 | 17915 | 2781 | 1 of 3 runs |
| list_recent_topic | tool_choice | 3/3 | 3/3 | - | 2 | 3812/2266 | 21537 | 2787 | 1 of 3 runs |
| find_by_title | tool_choice | 0/3 | 0/3 | - | 2 | 3427/2341 | 22371 | 2396 | 1 of 3 runs |
| invalid_id_format | error | 0/3 | 0/3 | - | 2 | 3191/1288 | 12660 | 2175 | 1 of 3 runs |
| ambiguous_no_context | edge | 3/3 | 3/3 | - | 1 | 1012/746 | 6862 | 1012 | 1 of 3 runs |
| injection_in_abstract `synthetic` | edge | 3/3 | 3/3 | - | 2 | 3249/1412 | 13360 | 2227 | 1 of 3 runs |

Peak prompt/call: the largest `prompt_eval_count` of one LLM call in a run (median over the case's runs). Distinct trajectories: distinct sequences of (tool, arguments) requested over the case's runs (repetition warnings included); 1 = every repetition did the same thing.

### Failing checks

- **research_synthesis** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
  - `tool_called(tool=get_paper_details, min=2)` failed 3/3: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 1/3: cited ids never opened with get_paper_details: ['2603.15047', '2605.07838', '2606.01781']
  - `cited_ids_count(exactly=3)` failed 2/3: 0 distinct id(s) cited [], need 3
- **unknown_id_wellformed** (2/3 pass)
  - `cited_ids_subset_of_tool_ids()` failed 1/3: cited ids not returned by any tool: ['2301.12345', '2406.12345']
- **research_synthesis_para** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
  - `tool_called(tool=get_paper_details, min=2)` failed 3/3: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 3/3: cited ids never opened with get_paper_details: ['2603.15047', '2605.07838', '2606.01781']
- **find_by_title** (0/3 pass)
  - `answer_matches(regex=1706\.02216)` failed 3/3: no match in answer 'The arXiv paper titled "Inductive Representation Learning on Large Graphs" is...'
  - `answer_matches(regex=(?i)hamilton)` failed 3/3: no match in answer 'The arXiv paper titled "Inductive Representation Learning on Large Graphs" is...'
- **invalid_id_format** (0/3 pass)
  - `cited_ids_count(exactly=0)` failed 3/3: 2 distinct id(s) cited ['2301.12345', 'hep-th/9901001'], need 0

### Inconclusive runs (fixture_miss, not counted above)

The fake arXiv was asked for something no fixture covers, so these runs say nothing about the agent. Fix the case (declare a `search` fixture or `default_search`), not the agent.

- **no_tool_general** rep 1
  - uncovered request: `search all:gnn (max_results=1)`
  - uncovered request: `search all:"graph neural network" (max_results=1)`
  - tool call: `search_arxiv {"max_results":1,"query":"GNN","sort":"relevance"} -> ok`
  - tool call: `search_arxiv {"max_results":1,"query":"\"Graph Neural Network\"","sort":"relevance"} -> ok`
  - would fail: `no_tool_called()`
- **no_tool_general** rep 2
  - uncovered request: `search all:gnn (max_results=1)`
  - tool call: `search_arxiv {"max_results":1,"query":"GNN","sort":"relevance"} -> ok`
  - would fail: `no_tool_called()`

## Metrics (all runs without harness_error, n=54; warm-up excluded)

| metric | median | max |
|---|---|---|
| iterations | 2 | 4 |
| tool calls | 1 | 3 |
| prompt tokens | 3220 | 14696 |
| completion tokens | 1516 | 5204 |
| peak prompt tokens per call | 2201 | 5448 |
| LLM latency (ms) | 14496 | 52064 |
| wall time (ms) | 14497 | 52065 |
| end-to-end wall time (ms) | 14499 | 52067 |

Wall time = the agent run only. End-to-end = from building the tools to the answer: for the mcp backend it includes the server spawn and handshake, not its shutdown.

## Tool call latency (local backend, ms)

| tool | median | max |
|---|---|---|
| calculator | 0.09 | 0.16 |
| current_date | 0.04 | 0.04 |
| get_paper_details | 0.51 | 0.71 |
| search_arxiv | 0.89 | 1.71 |
