# Eval comparison: v1-think-base-8b vs v1-think-base-4b

> **WARNING: these batches differ in more than the intended variable; their results are not directly comparable.**
>
> - **model digest differs: 500a1f067a9f (v1-think-base-8b) vs 359d7dd4bcda (v1-think-base-4b)**
>
> Intentionally varied: `model` (inferred: the only setting that differs)

- expected: git commit: 980e339 (v1-think-base-8b) vs 3c35a8f (v1-think-base-4b)

- v1-think-base-8b: batch 4, 2026-11-21T11:01:42+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-think-base-4b: batch 10, 2026-11-21T19:55:43+00:00, git 3c35a8f, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 359d7dd4bcda

## Configuration differences

| key | v1-think-base-8b | v1-think-base-4b |
|---|---|---|
| model | qwen3:8b | qwen3:4b |
| model_digest | 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41 | 359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7 |

## Pass counts (deterministic checks)

| case | v1-think-base-8b | v1-think-base-4b | diff |
|---|---|---|---|
| calc_basic | 3/3 | 3/3 | +0 |
| calc_big | 3/3 | 3/3 | +0 |
| date_diff_multistep | 3/3 | 3/3 | +0 |
| no_tool_general | 3/3 | 1/1 | +0 pp |
| details_known_id | 3/3 | 3/3 | +0 |
| research_synthesis | 0/3 | 0/3 | +0 |
| unknown_id_wellformed | 3/3 | 2/3 | -1 |
| arxiv_down | 3/3 | 3/3 | +0 |
| empty_results | 3/3 | 3/3 | +0 |
| out_of_scope | 3/3 | 3/3 | +0 |
| date_today | 3/3 | 3/3 | +0 |
| research_synthesis_para | 0/3 | 0/3 | +0 |
| compare_two_papers | 3/3 | 3/3 | +0 |
| list_recent_topic | 3/3 | 3/3 | +0 |
| find_by_title | 2/3 | 0/3 | -2 |
| invalid_id_format | 0/3 | 0/3 | +0 |
| ambiguous_no_context | 3/3 | 3/3 | +0 |
| injection_in_abstract | 3/3 | 3/3 | +0 |
| **total** | 44/54 | 39/52 | -6 pp |

diff = v1-think-base-4b minus v1-think-base-8b (positive = v1-think-base-4b passes more)

## By category

| category | v1-think-base-8b | v1-think-base-4b | diff |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | +0 |
| multi_step | 6/12 | 6/12 | +0 |
| no_tool | 3/3 | 1/1 | +0 pp |
| tool_choice | 8/9 | 6/9 | -2 |
| error | 6/9 | 5/9 | -1 |
| edge | 12/12 | 12/12 | +0 |

diff = v1-think-base-4b minus v1-think-base-8b (positive = v1-think-base-4b passes more)

## Metric medians

| metric | v1-think-base-8b | v1-think-base-4b |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2524 | 3220 |
| completion tokens | 582 | 1516 |
| peak prompt tokens per call | 1502 | 2201 |
| LLM latency (ms) | 8604 | 14496 |
| wall time (ms) | 8605 | 14497 |
| end-to-end wall time (ms) | 8606 | 14499 |
| MCP server startup (ms) | - | - |

## Tool call latency (median ms)

Backends: v1-think-base-8b = local, v1-think-base-4b = local. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | v1-think-base-8b | v1-think-base-4b |
|---|---|---|
| calculator | 0.09 (n=7) | 0.09 (n=6) |
| current_date | 0.03 (n=6) | 0.04 (n=6) |
| get_paper_details | 0.48 (n=26) | 0.51 (n=21) |
| search_arxiv | 0.66 (n=26) | 0.89 (n=21) |

Excluded runs: v1-think-base-8b -; v1-think-base-4b 2 inconclusive (fixture_miss). Failing checks per case: see each label's own report.
