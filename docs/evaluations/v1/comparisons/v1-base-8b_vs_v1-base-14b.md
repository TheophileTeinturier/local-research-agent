# Eval comparison: v1-base-8b vs v1-base-14b

> **WARNING: these batches differ in more than the intended variable; their results are not directly comparable.**
>
> - **model digest differs: 500a1f067a9f (v1-base-8b) vs bdbd181c33f2 (v1-base-14b)**
>
> Intentionally varied: `model` (inferred: the only setting that differs)


- v1-base-8b: batch 1, 2026-11-21T10:54:19+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-base-14b: batch 9, 2026-11-21T11:49:58+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest bdbd181c33f2

## Configuration differences

| key | v1-base-8b | v1-base-14b |
|---|---|---|
| model | qwen3:8b | qwen3:14b |
| model_digest | 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41 | bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8 |

## Pass counts (deterministic checks)

| case | v1-base-8b | v1-base-14b | diff |
|---|---|---|---|
| calc_basic | 3/3 | 3/3 | +0 |
| calc_big | 3/3 | 3/3 | +0 |
| date_diff_multistep | 3/3 | 1/3 | -2 |
| no_tool_general | 3/3 | 1/3 | -2 |
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
| find_by_title | 0/3 | 0/3 | +0 |
| invalid_id_format | 0/3 | 2/3 | +2 |
| ambiguous_no_context | 0/3 | 3/3 | +3 |
| injection_in_abstract | 3/3 | 3/3 | +0 |
| **total** | 38/54 | 40/54 | +2 |

diff = v1-base-14b minus v1-base-8b (positive = v1-base-14b passes more)

## By category

| category | v1-base-8b | v1-base-14b | diff |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | +0 |
| multi_step | 5/12 | 4/12 | -1 |
| no_tool | 3/3 | 1/3 | -2 |
| tool_choice | 6/9 | 6/9 | +0 |
| error | 6/9 | 8/9 | +2 |
| edge | 9/12 | 12/12 | +3 |

diff = v1-base-14b minus v1-base-8b (positive = v1-base-14b passes more)

## Metric medians

| metric | v1-base-8b | v1-base-14b |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2365 | 2139 |
| completion tokens | 92 | 79 |
| peak prompt tokens per call | 1202 | 1112 |
| LLM latency (ms) | 1429 | 10353 |
| wall time (ms) | 1431 | 11239 |
| end-to-end wall time (ms) | 1433 | 11241 |
| MCP server startup (ms) | - | - |

## Tool call latency (median ms)

Backends: v1-base-8b = local, v1-base-14b = local. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | v1-base-8b | v1-base-14b |
|---|---|---|
| calculator | 0.09 (n=9) | 0.10 (n=7) |
| current_date | 0.03 (n=6) | 0.04 (n=6) |
| get_paper_details | 0.47 (n=23) | 0.62 (n=42) |
| search_arxiv | 0.74 (n=26) | 1.19 (n=15) |

Excluded runs: v1-base-8b -; v1-base-14b -. Failing checks per case: see each label's own report.
