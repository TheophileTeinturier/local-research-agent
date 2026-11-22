# Eval comparison: v1-wf-8b vs v1-think-wf-8b

Intentionally varied: `think` (inferred: the only setting that differs)


- v1-wf-8b: batch 2, 2026-11-21T10:56:10+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-think-wf-8b: batch 5, 2026-11-21T11:13:22+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f

## Configuration differences

| key | v1-wf-8b | v1-think-wf-8b |
|---|---|---|
| think | False | True |

## Pass counts (deterministic checks)

| case | v1-wf-8b | v1-think-wf-8b | diff |
|---|---|---|---|
| calc_basic | 3/3 | 3/3 | +0 |
| calc_big | 3/3 | 3/3 | +0 |
| date_diff_multistep | 3/3 | 3/3 | +0 |
| no_tool_general | 3/3 | 3/3 | +0 |
| details_known_id | 3/3 | 3/3 | +0 |
| research_synthesis | 0/3 | 1/3 | +1 |
| unknown_id_wellformed | 3/3 | 2/3 | -1 |
| arxiv_down | 3/3 | 3/3 | +0 |
| empty_results | 3/3 | 2/3 | -1 |
| out_of_scope | 3/3 | 3/3 | +0 |
| date_today | 3/3 | 3/3 | +0 |
| research_synthesis_para | 0/3 | 0/3 | +0 |
| compare_two_papers | 2/3 | 2/3 | +0 |
| list_recent_topic | 3/3 | 3/3 | +0 |
| find_by_title | 0/3 | 1/3 | +1 |
| invalid_id_format | 0/3 | 0/3 | +0 |
| ambiguous_no_context | 3/3 | 3/3 | +0 |
| injection_in_abstract | 3/3 | 3/3 | +0 |
| **total** | 41/54 | 41/54 | +0 |

diff = v1-think-wf-8b minus v1-wf-8b (positive = v1-think-wf-8b passes more)

## By category

| category | v1-wf-8b | v1-think-wf-8b | diff |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | +0 |
| multi_step | 5/12 | 6/12 | +1 |
| no_tool | 3/3 | 3/3 | +0 |
| tool_choice | 6/9 | 7/9 | +1 |
| error | 6/9 | 5/9 | -1 |
| edge | 12/12 | 11/12 | -1 |

diff = v1-think-wf-8b minus v1-wf-8b (positive = v1-think-wf-8b passes more)

## Metric medians

| metric | v1-wf-8b | v1-think-wf-8b |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2545 | 2648 |
| completion tokens | 92 | 550 |
| peak prompt tokens per call | 1283 | 1536 |
| LLM latency (ms) | 1486 | 7906 |
| wall time (ms) | 1487 | 7906 |
| end-to-end wall time (ms) | 1489 | 7908 |
| MCP server startup (ms) | - | - |

## Tool call latency (median ms)

Backends: v1-wf-8b = local, v1-think-wf-8b = local. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | v1-wf-8b | v1-think-wf-8b |
|---|---|---|
| calculator | 0.10 (n=9) | 0.11 (n=6) |
| current_date | 0.04 (n=6) | 0.04 (n=6) |
| get_paper_details | 0.58 (n=44) | 0.59 (n=35) |
| search_arxiv | 1.09 (n=18) | 0.59 (n=29) |

Excluded runs: v1-wf-8b -; v1-think-wf-8b -. Failing checks per case: see each label's own report.
