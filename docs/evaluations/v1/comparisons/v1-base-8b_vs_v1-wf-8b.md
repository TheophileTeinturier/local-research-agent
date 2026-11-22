# Eval comparison: v1-base-8b vs v1-wf-8b

Intentionally varied: `prompt_variant` (inferred: the only setting that differs)

- expected: system prompt differs: b89d08f35be3 (v1-base-8b) vs ac6259a046bd (v1-wf-8b), as `prompt_variant` is varied

- v1-base-8b: batch 1, 2026-11-21T10:54:19+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-wf-8b: batch 2, 2026-11-21T10:56:10+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f

## Configuration differences

| key | v1-base-8b | v1-wf-8b |
|---|---|---|
| prompt_variant | baseline | research_workflow |
| system_prompt_sha256 | b89d08f35be3fd0255a2c33099e7830a4e101e4cd69486627b69fddfee450129 | ac6259a046bdf864e8f12451e699d9c85a1901c6db4eee35a1e5a8c011682ebb |

## Pass counts (deterministic checks)

| case | v1-base-8b | v1-wf-8b | diff |
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
| ambiguous_no_context | 0/3 | 3/3 | +3 |
| injection_in_abstract | 3/3 | 3/3 | +0 |
| **total** | 38/54 | 41/54 | +3 |

diff = v1-wf-8b minus v1-base-8b (positive = v1-wf-8b passes more)

## By category

| category | v1-base-8b | v1-wf-8b | diff |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | +0 |
| multi_step | 5/12 | 5/12 | +0 |
| no_tool | 3/3 | 3/3 | +0 |
| tool_choice | 6/9 | 6/9 | +0 |
| error | 6/9 | 6/9 | +0 |
| edge | 9/12 | 12/12 | +3 |

diff = v1-wf-8b minus v1-base-8b (positive = v1-wf-8b passes more)

## Metric medians

| metric | v1-base-8b | v1-wf-8b |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2365 | 2545 |
| completion tokens | 92 | 92 |
| peak prompt tokens per call | 1202 | 1283 |
| LLM latency (ms) | 1429 | 1486 |
| wall time (ms) | 1431 | 1487 |
| end-to-end wall time (ms) | 1433 | 1489 |
| MCP server startup (ms) | - | - |

## Tool call latency (median ms)

Backends: v1-base-8b = local, v1-wf-8b = local. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | v1-base-8b | v1-wf-8b |
|---|---|---|
| calculator | 0.09 (n=9) | 0.10 (n=9) |
| current_date | 0.03 (n=6) | 0.04 (n=6) |
| get_paper_details | 0.47 (n=23) | 0.58 (n=44) |
| search_arxiv | 0.74 (n=26) | 1.09 (n=18) |

Excluded runs: v1-base-8b -; v1-wf-8b -. Failing checks per case: see each label's own report.
