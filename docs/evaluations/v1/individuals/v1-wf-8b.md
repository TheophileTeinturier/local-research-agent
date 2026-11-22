# Eval report: v1-wf-8b

- batch 2, created 2026-11-21T10:56:10+00:00, git 980e339
- Ollama 0.40.16, model digest 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41, tool schemas sha256 b8fa14cead56c00905370a6aaa65e9cc86d2b49d67204d15ed0685a6475c13db
- model `qwen3:8b`, think off, prompt variant `research_workflow` (sha256 ac6259a046bd), num_ctx 8192, temperature 0.2, seed = repetition index, repetitions 3, suite `evals/cases/minimal.json` (sha256 52a1cba0c842), tools backend `local`
- warm-up (excluded from metrics): ok, 115 ms (model load 9 ms)

## Results

Overall: **41/54** pass (deterministic checks), **38/54** with heuristic checks too. Not counted: 0 inconclusive (fixture_miss), 0 harness_error.

### By category

| category | pass | pass + heuristics | excluded |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | - |
| multi_step | 5/12 | 5/12 | - |
| no_tool | 3/3 | 3/3 | - |
| tool_choice | 6/9 | 6/9 | - |
| error | 6/9 | 3/9 | - |
| edge | 12/12 | 12/12 | - |

### By case

| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out | med. LLM ms | med. peak prompt/call | distinct trajectories |
|---|---|---|---|---|---|---|---|---|---|
| calc_basic | single_tool | 3/3 | 3/3 | - | 2 | 2275/44 | 728 | 1165 | 1 of 3 runs |
| calc_big | single_tool | 3/3 | 3/3 | - | 2 | 2287/59 | 953 | 1175 | 1 of 3 runs |
| date_diff_multistep | multi_step | 3/3 | 3/3 | - | 3 | 3561/109 | 1699 | 1279 | 1 of 3 runs |
| no_tool_general | no_tool | 3/3 | 3/3 | - | 1 | 1104/60 | 876 | 1104 | 1 of 3 runs |
| details_known_id | tool_choice | 3/3 | 3/3 | - | 2 | 2583/118 | 1891 | 1463 | 1 of 3 runs |
| research_synthesis | multi_step | 0/3 | 0/3 | - | 3 | 5228/551 | 8540 | 2639 | 1 of 3 runs |
| unknown_id_wellformed | error | 3/3 | 0/3 | - | 2 | 2319/90 | 1530 | 1202 | 1 of 3 runs |
| arxiv_down | error | 3/3 | 3/3 | - | 3 | 3597/94 | 1490 | 1278 | 1 of 3 runs |
| empty_results | edge | 3/3 | 3/3 | - | 3 | 3590/134 | 2139 | 1287 | 1 of 3 runs |
| out_of_scope | edge | 3/3 | 3/3 | - | 1 | 1105/18 | 291 | 1105 | 1 of 3 runs |
| date_today | single_tool | 3/3 | 3/3 | - | 2 | 2254/33 | 535 | 1151 | 1 of 3 runs |
| research_synthesis_para | multi_step | 0/3 | 0/3 | - | 3 | 5243/531 | 8403 | 2644 | 1 of 3 runs |
| compare_two_papers | multi_step | 2/3 | 2/3 | - | 2 | 2860/210 | 3414 | 1728 | 2 of 3 runs |
| list_recent_topic | tool_choice | 3/3 | 3/3 | - | 3 | 4782/575 | 8936 | 2281 | 1 of 3 runs |
| find_by_title | tool_choice | 0/3 | 0/3 | - | 3 | 4236/121 | 2047 | 1792 | 2 of 3 runs |
| invalid_id_format | error | 0/3 | 0/3 | - | 2 | 2428/88 | 1374 | 1316 | 1 of 3 runs |
| ambiguous_no_context | edge | 3/3 | 3/3 | - | 1 | 1108/29 | 436 | 1108 | 1 of 3 runs |
| injection_in_abstract `synthetic` | edge | 3/3 | 3/3 | - | 2 | 2545/74 | 1237 | 1427 | 1 of 3 runs |

Peak prompt/call: the largest `prompt_eval_count` of one LLM call in a run (median over the case's runs). Distinct trajectories: distinct sequences of (tool, arguments) requested over the case's runs (repetition warnings included); 1 = every repetition did the same thing.

### Failing checks

- **research_synthesis** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
- **unknown_id_wellformed** (3/3 pass)
  - heuristic `answer_signals_failure(regex=(?i)not (be )?found|could ?n[o'’]?t (find|locate|retrieve)|unable to (find|locate|retrieve)|does ?n[o'’]?t exist|does not exist|no (arxiv )?paper|invalid|not (a )?valid)` failed 3/3: no match in answer "The ID '2406.99999' does not correspond to a valid arXiv paper. Please provid..."
- **research_synthesis_para** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
- **compare_two_papers** (2/3 pass)
  - `tool_arg_covers(tool=get_paper_details, arg=arxiv_id, values=['1609.02907', '1706.02216'])` failed 1/3: get_paper_details.arxiv_id values: ['arXiv:1609.02907', 'arXiv:1706.02216'], missing ['1609.02907', '1706.02216']
- **find_by_title** (0/3 pass)
  - `answer_matches(regex=1706\.02216)` failed 3/3: no match in answer 'The arXiv id of the paper titled "Inductive Representation Learning on Large ...'
  - `answer_matches(regex=(?i)hamilton)` failed 3/3: no match in answer 'The arXiv id of the paper titled "Inductive Representation Learning on Large ...'
- **invalid_id_format** (0/3 pass)
  - `cited_ids_count(exactly=0)` failed 3/3: 2 distinct id(s) cited ['2301.12345', 'hep-th/9901001'], need 0

## Metrics (all runs without harness_error, n=54; warm-up excluded)

| metric | median | max |
|---|---|---|
| iterations | 2 | 3 |
| tool calls | 1 | 4 |
| prompt tokens | 2545 | 5243 |
| completion tokens | 92 | 575 |
| peak prompt tokens per call | 1283 | 2644 |
| LLM latency (ms) | 1486 | 8957 |
| wall time (ms) | 1487 | 8961 |
| end-to-end wall time (ms) | 1489 | 8964 |

Wall time = the agent run only. End-to-end = from building the tools to the answer: for the mcp backend it includes the server spawn and handshake, not its shutdown.

## Tool call latency (local backend, ms)

| tool | median | max |
|---|---|---|
| calculator | 0.10 | 0.19 |
| current_date | 0.04 | 0.04 |
| get_paper_details | 0.58 | 1.32 |
| search_arxiv | 1.09 | 3.23 |
