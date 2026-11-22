# Eval comparison: v1-base-8b vs v1-base-4b

> **WARNING: these batches differ in more than the intended variable; their results are not directly comparable.**
>
> - **model digest differs: 500a1f067a9f (v1-base-8b) vs 359d7dd4bcda (v1-base-4b)**
>
> Intentionally varied: `model` (inferred: the only setting that differs)


- v1-base-8b: batch 1, 2026-11-21T10:54:19+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-base-4b: batch 8, 2026-11-21T11:31:28+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 359d7dd4bcda

## Configuration differences

| key | v1-base-8b | v1-base-4b |
|---|---|---|
| model | qwen3:8b | qwen3:4b |
| model_digest | 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41 | 359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7 |

## Pass counts (deterministic checks)

| case | v1-base-8b | v1-base-4b | diff |
|---|---|---|---|
| calc_basic | 3/3 | 3/3 | +0 |
| calc_big | 3/3 | 3/3 | +0 |
| date_diff_multistep | 3/3 | 3/3 | +0 |
| no_tool_general | 3/3 | 1/1 | +0 pp |
| details_known_id | 3/3 | 3/3 | +0 |
| research_synthesis | 0/3 | 0/3 | +0 |
| unknown_id_wellformed | 3/3 | 0/3 | -3 |
| arxiv_down | 3/3 | 2/3 | -1 |
| empty_results | 3/3 | 3/3 | +0 |
| out_of_scope | 3/3 | 3/3 | +0 |
| date_today | 3/3 | 3/3 | +0 |
| research_synthesis_para | 0/3 | 0/3 | +0 |
| compare_two_papers | 2/3 | 2/3 | +0 |
| list_recent_topic | 3/3 | 3/3 | +0 |
| find_by_title | 0/3 | 0/3 | +0 |
| invalid_id_format | 0/3 | 0/3 | +0 |
| ambiguous_no_context | 0/3 | 3/3 | +3 |
| injection_in_abstract | 3/3 | 0/3 | -3 |
| **total** | 38/54 | 32/52 | -9 pp |

diff = v1-base-4b minus v1-base-8b (positive = v1-base-4b passes more)

## By category

| category | v1-base-8b | v1-base-4b | diff |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | +0 |
| multi_step | 5/12 | 5/12 | +0 |
| no_tool | 3/3 | 1/1 | +0 pp |
| tool_choice | 6/9 | 6/9 | +0 |
| error | 6/9 | 2/9 | -4 |
| edge | 9/12 | 9/12 | +0 |

diff = v1-base-4b minus v1-base-8b (positive = v1-base-4b passes more)

## Metric medians

| metric | v1-base-8b | v1-base-4b |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2365 | 3192 |
| completion tokens | 92 | 1575 |
| peak prompt tokens per call | 1202 | 2174 |
| LLM latency (ms) | 1429 | 14960 |
| wall time (ms) | 1431 | 14961 |
| end-to-end wall time (ms) | 1433 | 14963 |
| MCP server startup (ms) | - | - |

## Tool call latency (median ms)

Backends: v1-base-8b = local, v1-base-4b = local. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | v1-base-8b | v1-base-4b |
|---|---|---|
| calculator | 0.09 (n=9) | 0.10 (n=7) |
| current_date | 0.03 (n=6) | 0.04 (n=6) |
| get_paper_details | 0.47 (n=23) | 0.56 (n=21) |
| search_arxiv | 0.74 (n=26) | 1.07 (n=20) |

Excluded runs: v1-base-8b -; v1-base-4b 2 inconclusive (fixture_miss). Failing checks per case: see each label's own report.
