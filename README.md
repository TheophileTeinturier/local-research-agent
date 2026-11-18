# local-research-agent
Local LLM agent with tool calling, LangGraph orchestration and an MCP server, evaluated across Ollama models

## MCP server

The agent's tools (calculator, current_date, search_arxiv, get_paper_details) are also served over MCP (stdio):

```
uv run python -m local_research_agent.mcp.server            # serve the tools on stdin/stdout
uv run python -m local_research_agent.cli --tools mcp "..." # the agent, using them through MCP
```

To use the server from Claude Desktop, add it to `claude_desktop_config.json`. The file is in `%APPDATA%\Claude\` on Windows and in `~/Library/Application Support/Claude/` on macOS:

```json
{
  "mcpServers": {
    "local-research-agent": {
      "command": "C:\\path\\to\\local-research-agent\\.venv\\Scripts\\python.exe",
      "args": ["-m", "local_research_agent.mcp.server"]
    }
  }
}
```
