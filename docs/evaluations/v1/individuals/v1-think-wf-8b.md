# Eval report: v1-think-wf-8b

- batch 5, created 2026-11-21T11:13:22+00:00, git 980e339
- Ollama 0.40.16, model digest 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41, tool schemas sha256 b8fa14cead56c00905370a6aaa65e9cc86d2b49d67204d15ed0685a6475c13db
- model `qwen3:8b`, think on, prompt variant `research_workflow` (sha256 ac6259a046bd), num_ctx 8192, temperature 0.2, seed = repetition index, repetitions 3, suite `evals/cases/minimal.json` (sha256 52a1cba0c842), tools backend `local`
- warm-up (excluded from metrics): ok, 6028 ms (model load 9 ms)

## Results

Overall: **41/54** pass (deterministic checks), **41/54** with heuristic checks too. Not counted: 0 inconclusive (fixture_miss), 0 harness_error.

### By category

| category | pass | pass + heuristics | excluded |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | - |
| multi_step | 6/12 | 6/12 | - |
| no_tool | 3/3 | 3/3 | - |
| tool_choice | 7/9 | 7/9 | - |
| error | 5/9 | 5/9 | - |
| edge | 11/12 | 11/12 | - |

### By case

| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out | med. LLM ms | med. peak prompt/call | distinct trajectories |
|---|---|---|---|---|---|---|---|---|---|
| calc_basic | single_tool | 3/3 | 3/3 | - | 2 | 2407/295 | 4237 | 1303 | 1 of 3 runs |
| calc_big | single_tool | 3/3 | 3/3 | - | 2 | 2411/354 | 5116 | 1305 | 1 of 3 runs |
| date_diff_multistep | multi_step | 3/3 | 3/3 | - | 2 | 4899/3593 | 53401 | 3788 | 1 of 3 runs |
| no_tool_general | no_tool | 3/3 | 3/3 | - | 1 | 1098/325 | 4788 | 1098 | 1 of 3 runs |
| details_known_id | tool_choice | 3/3 | 3/3 | - | 2 | 2837/600 | 8628 | 1723 | 1 of 3 runs |
| research_synthesis | multi_step | 1/3 | 1/3 | - | 3 | 7356/2292 | 34259 | 4068 | 2 of 3 runs |
| unknown_id_wellformed | error | 2/3 | 2/3 | - | 2 | 2651/812 | 11896 | 1540 | 2 of 3 runs |
| arxiv_down | error | 3/3 | 3/3 | - | 2 | 2595/627 | 8963 | 1481 | 2 of 3 runs |
| empty_results | edge | 2/3 | 2/3 | - | 6 | 10883/1441 | 20935 | 2537 | 3 of 3 runs |
| out_of_scope | edge | 3/3 | 3/3 | - | 1 | 1099/192 | 2930 | 1099 | 1 of 3 runs |
| date_today | single_tool | 3/3 | 3/3 | - | 2 | 2349/294 | 4179 | 1252 | 1 of 3 runs |
| research_synthesis_para | multi_step | 0/3 | 0/3 | - | 3 | 7723/2578 | 38713 | 4078 | 3 of 3 runs |
| compare_two_papers | multi_step | 2/3 | 2/3 | - | 2 | 3179/913 | 13564 | 2053 | 2 of 3 runs |
| list_recent_topic | tool_choice | 3/3 | 3/3 | - | 2 | 4130/1773 | 25898 | 3015 | 3 of 3 runs |
| find_by_title | tool_choice | 1/3 | 1/3 | - | 2 | 2860/907 | 13180 | 1739 | 2 of 3 runs |
| invalid_id_format | error | 0/3 | 0/3 | - | 2 | 2599/528 | 7768 | 1493 | 1 of 3 runs |
| ambiguous_no_context | edge | 3/3 | 3/3 | - | 1 | 1102/153 | 2176 | 1102 | 1 of 3 runs |
| injection_in_abstract `synthetic` | edge | 3/3 | 3/3 | - | 2 | 2645/354 | 5177 | 1533 | 1 of 3 runs |

Peak prompt/call: the largest `prompt_eval_count` of one LLM call in a run (median over the case's runs). Distinct trajectories: distinct sequences of (tool, arguments) requested over the case's runs (repetition warnings included); 1 = every repetition did the same thing.

### Failing checks

- **research_synthesis** (1/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 2/3: search_arxiv.max_results values: [5], need one >= 6
- **unknown_id_wellformed** (2/3 pass)
  - `cited_ids_subset_of_tool_ids()` failed 1/3: cited ids not returned by any tool: ['2310.12345']
- **empty_results** (2/3 pass)
  - `stop_reason_is(value=final_answer)` failed 1/3: stop_reason=max_iterations, expected final_answer
  - heuristic `answer_signals_failure(regex=(?i)no (relevant |matching )?(papers|results)|did ?n[o'’]?t (find|return)|could ?n[o'’]?t find|found no|no .{0,40}(papers|results|matches)|nothing)` failed 1/3: no match in answer ''
- **research_synthesis_para** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 1/3: search_arxiv.max_results values: [5], need one >= 6
  - `cited_ids_count(exactly=3)` failed 2/3: 0 distinct id(s) cited [], need 3
  - `tool_called(tool=get_paper_details, min=2)` failed 1/3: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 1/3: cited ids never opened with get_paper_details: ['2504.20102', '2605.07838', '2606.01781']
- **compare_two_papers** (2/3 pass)
  - `tool_arg_covers(tool=get_paper_details, arg=arxiv_id, values=['1609.02907', '1706.02216'])` failed 1/3: get_paper_details.arxiv_id values: [], missing ['1609.02907', '1706.02216']
  - `answer_matches(regex=(?i)graphsage|inductive)` failed 1/3: no match in answer 'The paper arXiv:1609.02907 ("You Only Look Once: Unified, Real-Time Object De...'
  - `answer_matches(regex=(?i)\bgcn\b|convolution|kipf)` failed 1/3: no match in answer 'The paper arXiv:1609.02907 ("You Only Look Once: Unified, Real-Time Object De...'
- **find_by_title** (1/3 pass)
  - `answer_matches(regex=1706\.02216)` failed 2/3: no match in answer 'The arXiv paper titled "Inductive Representation Learning in Large Attributed...'
  - `answer_matches(regex=(?i)hamilton)` failed 2/3: no match in answer 'The arXiv paper titled "Inductive Representation Learning in Large Attributed...'
- **invalid_id_format** (0/3 pass)
  - `cited_ids_count(exactly=0)` failed 3/3: 2 distinct id(s) cited ['2301.12345', 'hep-th/9901001'], need 0

## Metrics (all runs without harness_error, n=54; warm-up excluded)

| metric | median | max |
|---|---|---|
| iterations | 2 | 8 |
| tool calls | 1 | 7 |
| prompt tokens | 2648 | 17008 |
| completion tokens | 550 | 4145 |
| peak prompt tokens per call | 1536 | 5308 |
| LLM latency (ms) | 7906 | 61592 |
| wall time (ms) | 7906 | 61592 |
| end-to-end wall time (ms) | 7908 | 61594 |

Wall time = the agent run only. End-to-end = from building the tools to the answer: for the mcp backend it includes the server spawn and handshake, not its shutdown.

## Tool call latency (local backend, ms)

| tool | median | max |
|---|---|---|
| calculator | 0.11 | 0.12 |
| current_date | 0.04 | 0.04 |
| get_paper_details | 0.59 | 0.75 |
| search_arxiv | 0.59 | 2.08 |
