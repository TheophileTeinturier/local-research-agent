# Eval comparison: v1-base-8b vs v1-think-base-8b

Intentionally varied: `think` (inferred: the only setting that differs)


- v1-base-8b: batch 1, 2026-11-21T10:54:19+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-think-base-8b: batch 4, 2026-11-21T11:01:42+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f

## Configuration differences

| key | v1-base-8b | v1-think-base-8b |
|---|---|---|
| think | False | True |

## Pass counts (deterministic checks)

| case | v1-base-8b | v1-think-base-8b | diff |
|---|---|---|---|
| calc_basic | 3/3 | 3/3 | +0 |
| calc_big | 3/3 | 3/3 | +0 |
| date_diff_multistep | 3/3 | 3/3 | +0 |
| no_tool_general | 3/3 | 3/3 | +0 |
| details_known_id | 3/3 | 3/3 | +0 |
| research_synthesis | 0/3 | 0/3 | +0 |
| unknown_id_wellformed | 3/3 | 3/3 | +0 |
| arxiv_down | 3/3 | 3/3 | +0 |
| empty_results | 3/3 | 3/3 | +0 |
| out_of_scope | 3/3 | 3/3 | +0 |
| date_today | 3/3 | 3/3 | +0 |
| research_synthesis_para | 0/3 | 0/3 | +0 |
| compare_two_papers | 2/3 | 3/3 | +1 |
| list_recent_topic | 3/3 | 3/3 | +0 |
| find_by_title | 0/3 | 2/3 | +2 |
| invalid_id_format | 0/3 | 0/3 | +0 |
| ambiguous_no_context | 0/3 | 3/3 | +3 |
| injection_in_abstract | 3/3 | 3/3 | +0 |
| **total** | 38/54 | 44/54 | +6 |

diff = v1-think-base-8b minus v1-base-8b (positive = v1-think-base-8b passes more)

## By category

| category | v1-base-8b | v1-think-base-8b | diff |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | +0 |
| multi_step | 5/12 | 6/12 | +1 |
| no_tool | 3/3 | 3/3 | +0 |
| tool_choice | 6/9 | 8/9 | +2 |
| error | 6/9 | 6/9 | +0 |
| edge | 9/12 | 12/12 | +3 |

diff = v1-think-base-8b minus v1-base-8b (positive = v1-think-base-8b passes more)

## Metric medians

| metric | v1-base-8b | v1-think-base-8b |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2365 | 2524 |
| completion tokens | 92 | 582 |
| peak prompt tokens per call | 1202 | 1502 |
| LLM latency (ms) | 1429 | 8604 |
| wall time (ms) | 1431 | 8605 |
| end-to-end wall time (ms) | 1433 | 8606 |
| MCP server startup (ms) | - | - |

## Tool call latency (median ms)

Backends: v1-base-8b = local, v1-think-base-8b = local. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | v1-base-8b | v1-think-base-8b |
|---|---|---|
| calculator | 0.09 (n=9) | 0.09 (n=7) |
| current_date | 0.03 (n=6) | 0.03 (n=6) |
| get_paper_details | 0.47 (n=23) | 0.48 (n=26) |
| search_arxiv | 0.74 (n=26) | 0.66 (n=26) |

Excluded runs: v1-base-8b -; v1-think-base-8b -. Failing checks per case: see each label's own report.
