# Project framing

## Goal

Build a small research agent that runs entirely on local hardware: a local LLM served by Ollama answers questions by calling Python tools (arithmetic, the current date, arXiv search and paper lookup), with explicit guardrails, structured logs, and an evaluation harness that measures its behaviour reproducibly.

The project favours depth over breadth: one agent, a few tools, and enough measurement to tell whether a change helps or hurts.

## Final decisions

1. **A single agent.** One loop with one tool registry covers the use case (search, read, answer). Multiple cooperating agents would add coordination failure modes that a small local model handles poorly, and would make the evaluation harder to interpret. There is no multi-agent orchestration in v1.0.
2. **No agent-to-agent (A2A) protocol.** With a single agent there is no peer to talk to. Interoperability is provided at the tool level instead, through MCP.
3. **MCP over stdio.** The tools are exposed as an MCP server using the stdio transport: the client spawns the server as a subprocess and talks JSON-RPC over its pipes. This needs no network port, no authentication layer and no service to deploy. HTTP transports are out of scope.
4. **Native tool calling, not prompt-parsed actions.** The model receives JSON schemas and returns structured tool calls; nothing is parsed out of free text.
5. **Errors are observations.** Tool failures, invalid arguments and timeouts are returned to the model as `Error: ...` messages it can act on, instead of aborting the run. Only the loop's guardrails end a run early.
6. **A hand-written loop rather than a framework.** The loop is short enough to read in full, its invariants are tested directly, and the evaluation harness measures it as a baseline.

## Moved to the roadmap

- **LangGraph orchestration with a verification node.** Re-expressing the loop as a graph and adding a node that checks the draft answer against the tool results (for example, that every cited arXiv id was actually returned or opened) before answering. The current harness is designed to compare that version against the hand-written baseline.
- **Extended evaluation.** A larger suite, more models and repetitions, an evaluation matrix across models and prompt variants, and analysis beyond pass counts.
- **A FastAPI service** exposing the agent over HTTP.
- **docker-compose** packaging of the service together with Ollama.

## Non-goals

- Cloud LLM backends (the `LLMClient` interface would allow one, but v1.0 targets local models).
- Long-term memory, retrieval over full-text PDFs, or document ingestion.
- Production hardening of the MCP server (authentication, remote transports).
