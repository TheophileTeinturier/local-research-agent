"""The tools of the registry, served and consumed over MCP (stdio transport only).

Reading order: `server` (a `ToolRegistry` exposed as an MCP server), `client` (the
`McpClientBridge`: a sync facade over the async MCP client SDK), `provider` (`McpToolProvider`:
the bridge seen as a `ToolProvider`, so the agent loop sees no difference).

This package is named like the SDK it wraps (`mcp`). Inside it, `import mcp` is an absolute
import and always means the SDK; our own modules are reached with relative imports.
Nothing is imported here, so `import local_research_agent` never pulls in the SDK.
"""
