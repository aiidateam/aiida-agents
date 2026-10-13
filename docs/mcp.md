# Model Context Protocol (MCP)

`aiida-agents` exposes read-only AiiDA tools through the Model Context Protocol (MCP). MCP clients such as Claude Code can use these tools to inspect processes, query nodes, retrieve outputs, and search AiiDA documentation.

## Prerequisites

- Install `aiida-agents` in a working Python environment.
- Configure AiiDA and create or select a valid AiiDA profile.
- Ensure the selected profile's storage is accessible.
- For documentation search, install the `rag` extra and build the documentation index as described in the RAG documentation.

The MCP server opens the selected profile's storage during startup. It will not start successfully if the AiiDA configuration or storage is unavailable.

## Claude Code: stdio transport

The stdio transport lets Claude Code launch the MCP server as a subprocess. Run the following command from your terminal, replacing the example path with the absolute path to your local repository:

```bash
claude mcp add --transport stdio aiida-agents -- uv --directory /absolute/path/to/aiida-agents run aiida-agents mcp serve --transport stdio
