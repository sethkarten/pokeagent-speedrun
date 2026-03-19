# Server

The game server and MCP proxy run here. The design is headless: the emulator and game loop run in one process; agents and UIs connect as clients.

## Components

- **`app.py`** — FastAPI game server. Runs the mGBA emulator, game loop, and state caching (100 ms for full state, 5 s for map data). Exposes REST endpoints: `/action`, `/state`, `/status`, `/screenshot`, `/save_state`, `/load_state`, `/checkpoint`, etc. WebSocket `/ws/frames` streams frames to the web UI. MCP tool endpoints under `/mcp/*` let external agents (e.g. Claude Code) interact with the game. **Ports**: game=8000, frame=8001, MCP=8002 (relevant for containerized CLI agent execution).
- **`frame_server.py`** — Lightweight frame streaming for the web UI.
- **`stream.html`** — Web UI for real-time gameplay (e.g. `http://localhost:8000/stream`).
- **`cli/pokemon_mcp_server.py`** — MCP proxy. Translates MCP tool calls (stdio) into HTTP requests to the game server. Used by `run_cli.py` for containerized CLI agents.

## Clients

- **`run.py`** — Starts the server as a subprocess, then runs the in-repo agent client. Polls `/state`, runs the selected agent (e.g. PokeAgent), submits actions via `POST /action`.
- **`run_cli.py`** — For external CLI agents. Spawns the game server and the MCP server; the external agent talks to the MCP server over stdio (SSE in containerized mode).

## State and concurrency

- Clients poll `/state` for the latest game state. The server caches full state for 100 ms and map data for 5 s.
- The server uses shared state (`current_obs`, `step_count`) protected by locks; a thread-safe action queue serializes button inputs.
