# Eval report: v1-mcp-8b

- batch 3, created 2026-11-21T10:58:33+00:00, git 980e339
- Ollama 0.40.16, model digest 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41, tool schemas sha256 b8fa14cead56c00905370a6aaa65e9cc86d2b49d67204d15ed0685a6475c13db
- model `qwen3:8b`, think off, prompt variant `baseline` (sha256 b89d08f35be3), num_ctx 8192, temperature 0.2, seed = repetition index, repetitions 3, suite `evals/cases/minimal.json` (sha256 52a1cba0c842), tools backend `mcp`
- warm-up (excluded from metrics): ok, 142 ms (model load 8 ms)

## Results

Overall: **38/54** pass (deterministic checks), **35/54** with heuristic checks too. Not counted: 0 inconclusive (fixture_miss), 0 harness_error.

### By category

| category | pass | pass + heuristics | excluded |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | - |
| multi_step | 5/12 | 5/12 | - |
| no_tool | 3/3 | 3/3 | - |
| tool_choice | 6/9 | 6/9 | - |
| error | 6/9 | 3/9 | - |
| edge | 9/12 | 9/12 | - |

### By case

| case | category | pass | pass + heur. | excluded | med. iter. | med. tokens in/out | med. LLM ms | med. peak prompt/call | distinct trajectories |
|---|---|---|---|---|---|---|---|---|---|
| calc_basic | single_tool | 3/3 | 3/3 | - | 2 | 2095/44 | 696 | 1075 | 1 of 3 runs |
| calc_big | single_tool | 3/3 | 3/3 | - | 2 | 2107/59 | 898 | 1085 | 1 of 3 runs |
| date_diff_multistep | multi_step | 3/3 | 3/3 | - | 3 | 3291/109 | 1652 | 1189 | 1 of 3 runs |
| no_tool_general | no_tool | 3/3 | 3/3 | - | 1 | 1014/59 | 848 | 1014 | 1 of 3 runs |
| details_known_id | tool_choice | 3/3 | 3/3 | - | 2 | 2403/120 | 1849 | 1373 | 1 of 3 runs |
| research_synthesis | multi_step | 0/3 | 0/3 | - | 2 | 2409/320 | 4658 | 1374 | 1 of 3 runs |
| unknown_id_wellformed | error | 3/3 | 0/3 | - | 3 | 3341/114 | 1744 | 1202 | 2 of 3 runs |
| arxiv_down | error | 3/3 | 3/3 | - | 3 | 3327/94 | 1442 | 1188 | 1 of 3 runs |
| empty_results | edge | 3/3 | 3/3 | - | 4 | 4627/184 | 2750 | 1299 | 1 of 3 runs |
| out_of_scope | edge | 3/3 | 3/3 | - | 1 | 1015/18 | 288 | 1015 | 1 of 3 runs |
| date_today | single_tool | 3/3 | 3/3 | - | 2 | 2074/33 | 546 | 1061 | 1 of 3 runs |
| research_synthesis_para | multi_step | 0/3 | 0/3 | - | 2 | 2419/434 | 6242 | 1379 | 1 of 3 runs |
| compare_two_papers | multi_step | 2/3 | 2/3 | - | 2 | 2680/196 | 3008 | 1638 | 2 of 3 runs |
| list_recent_topic | tool_choice | 3/3 | 3/3 | - | 2 | 2556/241 | 3664 | 1525 | 2 of 3 runs |
| find_by_title | tool_choice | 0/3 | 0/3 | - | 2 | 2264/91 | 1386 | 1227 | 2 of 3 runs |
| invalid_id_format | error | 0/3 | 0/3 | - | 2 | 2248/88 | 1414 | 1226 | 1 of 3 runs |
| ambiguous_no_context | edge | 0/3 | 0/3 | - | 2 | 2111/74 | 1126 | 1093 | 2 of 3 runs |
| injection_in_abstract `synthetic` | edge | 3/3 | 3/3 | - | 2 | 2365/84 | 1339 | 1337 | 1 of 3 runs |

Peak prompt/call: the largest `prompt_eval_count` of one LLM call in a run (median over the case's runs). Distinct trajectories: distinct sequences of (tool, arguments) requested over the case's runs (repetition warnings included); 1 = every repetition did the same thing.

### Failing checks

- **research_synthesis** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
  - `tool_called(tool=get_paper_details, min=2)` failed 3/3: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 3/3: cited ids never opened with get_paper_details: ['2603.15047', '2605.07838', '2606.01781']
- **unknown_id_wellformed** (3/3 pass)
  - heuristic `answer_signals_failure(regex=(?i)not (be )?found|could ?n[o'’]?t (find|locate|retrieve)|unable to (find|locate|retrieve)|does ?n[o'’]?t exist|does not exist|no (arxiv )?paper|invalid|not (a )?valid)` failed 3/3: no match in answer 'The arXiv ID "2406.99999" does not correspond to any known paper. Please veri...'
- **research_synthesis_para** (0/3 pass)
  - `tool_arg_min(tool=search_arxiv, arg=max_results, value=6)` failed 3/3: search_arxiv.max_results values: [3], need one >= 6
  - `tool_called(tool=get_paper_details, min=2)` failed 3/3: get_paper_details executed 0 time(s), need >= 2
  - `cited_ids_subset_of_opened_ids()` failed 3/3: cited ids never opened with get_paper_details: ['2603.15047', '2605.07838', '2606.01781']
- **compare_two_papers** (2/3 pass)
  - `tool_arg_covers(tool=get_paper_details, arg=arxiv_id, values=['1609.02907', '1706.02216'])` failed 1/3: get_paper_details.arxiv_id values: [], missing ['1609.02907', '1706.02216']
  - `answer_matches(regex=(?i)graphsage|inductive)` failed 1/3: no match in answer 'I cannot directly compare the papers without their details. Let me retrieve t...'
  - `answer_matches(regex=(?i)\bgcn\b|convolution|kipf)` failed 1/3: no match in answer 'I cannot directly compare the papers without their details. Let me retrieve t...'
- **find_by_title** (0/3 pass)
  - `answer_matches(regex=1706\.02216)` failed 3/3: no match in answer 'The arXiv paper titled "Inductive Representation Learning on Large Graphs" ha...'
  - `answer_matches(regex=(?i)hamilton)` failed 3/3: no match in answer 'The arXiv paper titled "Inductive Representation Learning on Large Graphs" ha...'
- **invalid_id_format** (0/3 pass)
  - `cited_ids_count(exactly=0)` failed 3/3: 2 distinct id(s) cited ['2301.12345', 'hep-th/9901001'], need 0
- **ambiguous_no_context** (0/3 pass)
  - `no_tool_called()` failed 3/3: tools requested: ['search_arxiv']

## Metrics (all runs without harness_error, n=54; warm-up excluded)

| metric | median | max |
|---|---|---|
| iterations | 2 | 4 |
| tool calls | 1 | 3 |
| prompt tokens | 2365 | 4627 |
| completion tokens | 92 | 460 |
| peak prompt tokens per call | 1202 | 1702 |
| LLM latency (ms) | 1425 | 6644 |
| wall time (ms) | 1430 | 6648 |
| end-to-end wall time (ms) | 2769 | 7895 |
| MCP server startup (ms) | 1296 | 1452 |

Wall time = the agent run only. End-to-end = from building the tools to the answer: for the mcp backend it includes the server spawn and handshake, not its shutdown.

## Tool call latency (mcp backend, ms)

| tool | median | max |
|---|---|---|
| calculator | 2.05 | 2.72 |
| current_date | 1.90 | 1.95 |
| get_paper_details | 2.97 | 4.57 |
| search_arxiv | 3.19 | 4.21 |
