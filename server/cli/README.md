# CLI Agent Folder

This folder contains MCP (Model Context Protocol) servers and CLI agent infrastructure for the Pokemon Emerald AI agent project.

## Structure

```
cli/
├── README.md                    # This file
├── GEMINI_CLI_TOOLS.md         # Complete reference for gemini-cli built-in tools
├── pokemon_mcp_server.py       # Pokemon-specific MCP tools (game control, pathfinding, knowledge)
└── baseline_mcp_server.py      # Baseline gemini-cli tools (file system, shell, web, memory)
```

## MCP Servers

### Pokemon MCP Server (`pokemon_mcp_server.py`)

Provides Pokemon Emerald-specific tools:

**Game Control:**
- `get_game_state()` - Get current game state (player, party, map, items)
- `press_buttons(buttons, reasoning)` - Press GBA buttons
- `navigate_to(x, y, reason)` - Pathfinding to coordinates with A* algorithm

**Knowledge Management:**
- `add_knowledge(category, title, content, location, coordinates, importance)` - Store discoveries
- `search_knowledge(category, query, location, min_importance)` - Search stored knowledge
- `get_knowledge_summary(min_importance)` - View important discoveries

**Pokemon Wiki Resources:**
- `lookup_pokemon_info(topic, source)` - Fetch info from Pokemon wikis
  - Sources: Bulbapedia, Serebii, PokemonDB, Marriland
  - Get details about Pokemon, moves, locations, items, NPCs, gym leaders
- `list_wiki_sources()` - List available wiki sources
- `get_walkthrough(part)` - Get official Emerald walkthrough (Parts 1-21)
  - Structured walkthrough from Bulbapedia
  - Covers entire game progression from start to Elite Four

**Usage:**
```bash
# Run standalone
uv run python cli/pokemon_mcp_server.py

# Configure in .gemini/settings.json
{
  "mcpServers": {
    "pokemon-emerald": {
      "command": "uv",
      "args": ["run", "python", "cli/pokemon_mcp_server.py"],
      "cwd": "/home/milkkarten/Pton/Research/pokeagent-speedrun",
      "trust": true
    }
  }
}
```

### Baseline MCP Server (`baseline_mcp_server.py`)

✅ **All 11 standard gemini-cli tools fully implemented:**

**File System Tools (7):**
- `list_directory`, `read_file`, `write_file`
- `glob`, `search_file_content`, `replace`
- `read_many_files`

**System Tools (1):**
- `run_shell_command` (with allowlist security)

**Web Tools (2):**
- `web_fetch` (fetch and parse web pages)
- `google_web_search` (DuckDuckGo search, no API key needed)

**Memory Tools (1):**
- `save_memory`

**Security Features:**
- File writes restricted to `.pokeagent_cache/cli/` only
- Shell commands use allowlist (42 safe commands)
- Path validation and command chaining detection

See [`GEMINI_CLI_TOOLS.md`](./GEMINI_CLI_TOOLS.md) for complete tool documentation.

## Configuration

### .gemini/settings.json

Configure both MCP servers:

```json
{
  "mcpServers": {
    "pokemon-emerald": {
      "command": "uv",
      "args": ["run", "python", "cli/pokemon_mcp_server.py"],
      "cwd": "/home/milkkarten/Pton/Research/pokeagent-speedrun",
      "trust": true,
      "description": "Pokemon Emerald game integration"
    },
    "baseline-tools": {
      "command": "uv",
      "args": ["run", "python", "cli/baseline_mcp_server.py"],
      "cwd": "/home/milkkarten/Pton/Research/pokeagent-speedrun",
      "trust": true,
      "description": "Standard gemini-cli tools (file system, shell, web, memory)"
    }
  }
}
```

### Trust Mode

Both servers are configured with `"trust": true` to auto-approve tool executions. This aligns with the `--yolo` mode in gemini-cli for seamless autonomous agent operation.

## Agent Integration

The CLI agent (`agent/cli_agent.py`) uses these MCP servers through gemini-cli:

```bash
# Run with CLI agent
uv run python run.py --scaffold cli --no-ocr

# Or directly
uv run python agent/cli_agent.py --max-steps 10
```

The agent:
1. Uses gemini-cli with `--yolo` mode for auto-approval
2. Maintains conversation history across steps
3. Logs interactions to web interface via LLM logger
4. Automatically discovers and uses all MCP tools

## Development

### Adding New Tools

To add a new tool to the Pokemon MCP server:

```python
@mcp.tool()
def your_tool(param1: str, param2: int) -> dict:
    """Tool description for the LLM."""
    try:
        # Implementation
        return {"success": True, "result": "..."}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Testing MCP Servers

Test standalone:
```bash
# Start server
uv run python cli/pokemon_mcp_server.py

# In another terminal, use gemini-cli
gemini "Use get_game_state to check the current game state"
```

### Debugging

Enable debug logging:
```bash
# Set environment variable
export MCP_DEBUG=1

# Run with verbose output
uv run python cli/pokemon_mcp_server.py --verbose
```

## References

- **MCP Protocol:** [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **FastMCP SDK:** [github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp)
- **Gemini CLI:** [github.com/google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)
- **Tool Reference:** [GEMINI_CLI_TOOLS.md](./GEMINI_CLI_TOOLS.md)
