# Eval comparison: v1-base-8b vs v1-mcp-8b

Intentionally varied: `tool_backend` (inferred: the only setting that differs)


- v1-base-8b: batch 1, 2026-11-21T10:54:19+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-mcp-8b: batch 3, 2026-11-21T10:58:33+00:00, git 980e339, temperature 0.2, tools backend `mcp`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f

## Configuration differences

| key | v1-base-8b | v1-mcp-8b |
|---|---|---|
| tool_backend | local | mcp |

## Pass counts (deterministic checks)

| case | v1-base-8b | v1-mcp-8b | diff |
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
| compare_two_papers | 2/3 | 2/3 | +0 |
| list_recent_topic | 3/3 | 3/3 | +0 |
| find_by_title | 0/3 | 0/3 | +0 |
| invalid_id_format | 0/3 | 0/3 | +0 |
| ambiguous_no_context | 0/3 | 0/3 | +0 |
| injection_in_abstract | 3/3 | 3/3 | +0 |
| **total** | 38/54 | 38/54 | +0 |

diff = v1-mcp-8b minus v1-base-8b (positive = v1-mcp-8b passes more)

## By category

| category | v1-base-8b | v1-mcp-8b | diff |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | +0 |
| multi_step | 5/12 | 5/12 | +0 |
| no_tool | 3/3 | 3/3 | +0 |
| tool_choice | 6/9 | 6/9 | +0 |
| error | 6/9 | 6/9 | +0 |
| edge | 9/12 | 9/12 | +0 |

diff = v1-mcp-8b minus v1-base-8b (positive = v1-mcp-8b passes more)

## Metric medians

| metric | v1-base-8b | v1-mcp-8b |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2365 | 2365 |
| completion tokens | 92 | 92 |
| peak prompt tokens per call | 1202 | 1202 |
| LLM latency (ms) | 1429 | 1425 |
| wall time (ms) | 1431 | 1430 |
| end-to-end wall time (ms) | 1433 | 2769 |
| MCP server startup (ms) | - | 1296 |

## Tool call latency (local vs mcp)

Backends: v1-base-8b = local, v1-mcp-8b = mcp. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | local (v1-base-8b) | mcp (v1-mcp-8b) | mcp - local | mcp / local |
|---|---|---|---|---|
| calculator | 0.09 (n=9) | 2.05 (n=9) | +1.96 | x23.3 |
| current_date | 0.03 (n=6) | 1.90 (n=6) | +1.86 | x55.8 |
| get_paper_details | 0.47 (n=23) | 2.97 (n=23) | +2.51 | x6.4 |
| search_arxiv | 0.74 (n=26) | 3.19 (n=26) | +2.45 | x4.3 |

Excluded runs: v1-base-8b -; v1-mcp-8b -. Failing checks per case: see each label's own report.
