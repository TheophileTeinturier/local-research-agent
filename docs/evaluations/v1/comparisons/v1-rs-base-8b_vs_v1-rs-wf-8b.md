# Eval comparison: v1-rs-base-8b vs v1-rs-wf-8b

Intentionally varied: `prompt_variant` (inferred: the only setting that differs)

- expected: system prompt differs: b89d08f35be3 (v1-rs-base-8b) vs ac6259a046bd (v1-rs-wf-8b), as `prompt_variant` is varied

- v1-rs-base-8b: batch 6, 2026-11-21T11:26:34+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f
- v1-rs-wf-8b: batch 7, 2026-11-21T11:28:29+00:00, git 980e339, temperature 0.2, tools backend `local`, suite sha256 52a1cba0c842, tool schemas sha256 b8fa14cead56, ollama 0.40.16, model digest 500a1f067a9f

## Configuration differences

| key | v1-rs-base-8b | v1-rs-wf-8b |
|---|---|---|
| prompt_variant | baseline | research_workflow |
| system_prompt_sha256 | b89d08f35be3fd0255a2c33099e7830a4e101e4cd69486627b69fddfee450129 | ac6259a046bdf864e8f12451e699d9c85a1901c6db4eee35a1e5a8c011682ebb |

## Pass counts (deterministic checks)

| case | v1-rs-base-8b | v1-rs-wf-8b | diff |
|---|---|---|---|
| research_synthesis | 0/10 | 1/10 | +1 |
| research_synthesis_para | 0/10 | 0/10 | +0 |
| **total** | 0/20 | 1/20 | +1 |

diff = v1-rs-wf-8b minus v1-rs-base-8b (positive = v1-rs-wf-8b passes more)

## By category

| category | v1-rs-base-8b | v1-rs-wf-8b | diff |
|---|---|---|---|
| multi_step | 0/20 | 1/20 | +1 |

diff = v1-rs-wf-8b minus v1-rs-base-8b (positive = v1-rs-wf-8b passes more)

## Metric medians

| metric | v1-rs-base-8b | v1-rs-wf-8b |
|---|---|---|
| iterations | 2 | 3 |
| tool calls | 1 | 4 |
| prompt tokens | 2414 | 5243 |
| completion tokens | 378 | 532 |
| peak prompt tokens per call | 1376 | 2644 |
| LLM latency (ms) | 6021 | 8440 |
| wall time (ms) | 6023 | 8444 |
| end-to-end wall time (ms) | 6024 | 8446 |
| MCP server startup (ms) | - | - |

## Tool call latency (median ms)

Backends: v1-rs-base-8b = local, v1-rs-wf-8b = local. Local = in-process execution; mcp = round trip through the bridge and the server. Medians in ms over executed calls (n = calls).

| tool | v1-rs-base-8b | v1-rs-wf-8b |
|---|---|---|
| search_arxiv | 1.18 (n=20) | 1.13 (n=20) |
| get_paper_details | - | 0.62 (n=60) |

Excluded runs: v1-rs-base-8b -; v1-rs-wf-8b -. Failing checks per case: see each label's own report.
