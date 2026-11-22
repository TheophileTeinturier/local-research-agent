# Eval comparison: v1-think-base-8b vs v1-think-wf-8b

Intentionally varied: `prompt_variant` (inferred: the only setting that differs)

- expected: system prompt differs: b89d08f35be3 (v1-think-base-8b) vs ac6259a046bd (v1-think-wf-8b), as `prompt_variant` is varied

- v1-think-base-8b: batch 4, 2026-11-21T11:01:42+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-think-wf-8b: batch 5, 2026-11-21T11:13:22+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f

## Configuration differences

| key | v1-think-base-8b | v1-think-wf-8b |
|---|---|---|
| prompt_variant | baseline | research_workflow |
| system_prompt_sha256 | b89d08f35be3fd0255a2c33099e7830a4e101e4cd69486627b69fddfee450129 | ac6259a046bdf864e8f12451e699d9c85a1901c6db4eee35a1e5a8c011682ebb |

## Pass counts (deterministic checks)

| case | v1-think-base-8b | v1-think-wf-8b | diff |
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
| compare_two_papers | 3/3 | 2/3 | -1 |
| list_recent_topic | 3/3 | 3/3 | +0 |
| find_by_title | 2/3 | 1/3 | -1 |
| invalid_id_format | 0/3 | 0/3 | +0 |
| ambiguous_no_context | 3/3 | 3/3 | +0 |
| injection_in_abstract | 3/3 | 3/3 | +0 |
| **total** | 44/54 | 41/54 | -3 |

diff = v1-think-wf-8b minus v1-think-base-8b (positive = v1-think-wf-8b passes more)

## By category

| category | v1-think-base-8b | v1-think-wf-8b | diff |
|---|---|---|---|
| single_tool | 9/9 | 9/9 | +0 |
| multi_step | 6/12 | 6/12 | +0 |
| no_tool | 3/3 | 3/3 | +0 |
| tool_choice | 8/9 | 7/9 | -1 |
| error | 6/9 | 5/9 | -1 |
| edge | 12/12 | 11/12 | -1 |

diff = v1-think-wf-8b minus v1-think-base-8b (positive = v1-think-wf-8b passes more)

## Metric medians

| metric | v1-think-base-8b | v1-think-wf-8b |
|---|---|---|
| iterations | 2 | 2 |
| tool calls | 1 | 1 |
| prompt tokens | 2524 | 2648 |
| completion tokens | 582 | 550 |
| peak prompt tokens per call | 1502 | 1536 |
| LLM latency (ms) | 8604 | 7906 |
| wall time (ms) | 8605 | 7906 |
| end-to-end wall time (ms) | 8606 | 7908 |
| MCP server startup (ms) | - | - |

## Tool call latency (median ms)

Backends: v1-think-base-8b = local, v1-think-wf-8b = local. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | v1-think-base-8b | v1-think-wf-8b |
|---|---|---|
| calculator | 0.09 (n=7) | 0.11 (n=6) |
| current_date | 0.03 (n=6) | 0.04 (n=6) |
| get_paper_details | 0.48 (n=26) | 0.59 (n=35) |
| search_arxiv | 0.66 (n=26) | 0.59 (n=29) |

Excluded runs: v1-think-base-8b -; v1-think-wf-8b -. Failing checks per case: see each label's own report.
