# Client-Server Architecture

This document details the architectural patterns and communication mechanisms between the game server and client processes in the Pokemon Emerald Speedrun codebase.

## Overview

The system follows a headless server architecture where the game logic and emulator run in a server process (`server/app.py`), while agents and user interfaces run as separate client processes (`run.py`, `run_cli.py`).

## 1. Connection Mechanisms

### Primary Protocol: HTTP REST API
- **Server**: FastAPI (`server/app.py`) listening on port 8000 (configurable via environment variables).
- **Clients**: Connect via standard HTTP requests using the `requests` library.
- **Base URL**: `http://localhost:{port}`.

### Secondary Protocol: WebSocket
- **Endpoint**: `/ws/frames`
- **Purpose**: Real-time frame streaming to the web UI (`stream.html`).
- **Implementation**: `ConnectionManager` class handles WebSocket connections.
- **Update Rate**: ~60 FPS check rate (16ms intervals), though effective transmission rate depends on network and client processing.

### MCP Protocol (Model Context Protocol)
- **Purpose**: Enables standard CLI agents (like Anthropic's Claude Code or other MCP-compliant tools) to interact with the game server.
- **Implementation**:
  - `run_cli.py` acts as the MCP client/host. Backend selection: `--backend {claude,gemini,codex}`.
  - `server/cli/pokemon_mcp_server.py` acts as a thin proxy layer.
  - **Proxy Pattern**: The MCP server does not contain game logic. It translates MCP tool calls into HTTP POST requests to the main game server.
  - **Transport**: Stdio (standard input/output) communication between the agent and the MCP server process.

## 2. Key API Endpoints & Message Types

### Core Game Control
- `POST /action`: Submit button presses. Accepts a list of buttons and optional speed presets (fast/normal/slow).
- `GET /state`: Retrieve comprehensive game state (visual, player coordinates, party info, map data, milestones).
- `GET /status`: Quick server status check.
- `GET /health`: Health check endpoint.
- `GET /screenshot`: Retrieve the current frame as an image.

### Agent thinking (UI)
- Agent thinking for the stream UI is driven by the LLM log (session log file) and optional `POST /thinking`; the server does not rely on a single static file for live thinking.

### MCP Tool Endpoints
The server exposes endpoints under `/mcp/*` that map to agent tools:
- **Game Interaction**: `/mcp/get_game_state`, `/mcp/press_buttons`, `/mcp/navigate_to`.
- **Knowledge & Wiki**: `/mcp/add_knowledge`, `/mcp/search_knowledge`, `/mcp/lookup_pokemon_info`.
- **Objectives**: `/mcp/complete_direct_objective`, `/mcp/create_direct_objectives`.
- **Memory**: `/mcp/save_memory` (writes to run directory AGENT.md).

### State Management
- `POST /save_state`: Save emulator state (Protected by API Key).
- `POST /load_state`: Load emulator state (Protected by API Key).
- `POST /checkpoint`: Create a checkpoint (state + milestones + maps).
- `POST /sync_llm_metrics`: Sync client-side LLM metrics to the server. Used by `run_cli` to push JSONL-derived token/cost metrics (single-writer pattern; server persists to `cumulative_metrics.json`).

## 3. State Synchronization

### Polling-Based Updates
- **Mechanism**: Clients poll the `/state` endpoint to get the latest game state.
- **Caching**: The server implements a 100ms cache for the comprehensive state to reduce overhead from rapid polling.
- **Map Caching**: Map data is cached per location with a 5-second TTL to avoid re-parsing static map structures unnecessarily.

### Threading Model
- **Concurrency**: The server uses multiple threads to handle the game loop, API requests, and WebSocket streaming.
- **Locks**:
  - `obs_lock`: Protects the `current_obs` (observation) shared between the game loop and API endpoints.
  - `step_lock`: Ensures atomic step updates.
  - `memory_lock`: Protects shared memory structures.
- **Action Queue**: A thread-safe queue manages action sequences, preventing conflicts when multiple requests arrive simultaneously.

### Single Writer Principle (Metrics)
- The server is the authoritative source for game state and certain metrics (total actions, milestones).
- Client-side metrics (tokens, cost) are synced to the server, but the server preserves its authoritative fields during the merge.

## 4. Software Engineering Principles Deviation

**God Class / Separation of Concerns (Violated)**
- **Issue**: `server/app.py` is a monolithic file (~4800 lines) handling disparate responsibilities: HTTP endpoints, game loop, state management, MCP tool logic, video recording, and metrics tracking.
- **Principle**: *Separation of Concerns* / *Single Responsibility Principle*.
- **Impact**: High maintainability cost; difficult to test individual components in isolation.

**Polling vs. Event-Driven (Inefficiency)**
- **Issue**: State synchronization relies heavily on clients polling `/state`.
- **Principle**: *Efficiency* / *Don't Repeat Yourself (in execution)*.
- **Impact**: Wasted bandwidth and CPU cycles checking for updates when nothing has changed. A push-based model (WebSockets for state) would be more efficient.

**Global State usage (Tight Coupling)**
- **Issue**: Heavy reliance on global variables (`env`, `current_obs`, `step_count`) within `server/app.py`.
- **Principle**: *Encapsulation*.
- **Impact**: Makes concurrency difficult to manage correctly (requires manual locking everywhere) and makes the system fragile to state changes.

**Inconsistent Error Handling**
- **Issue**: API endpoints mix `JSONResponse` returns with `HTTPException` raises.
- **Principle**: *Consistency* / *Principle of Least Astonishment*.
- **Impact**: Clients must implement complex error handling logic to cover different response formats.

**Security Gaps**
- **Issue**: While `/save_state` is protected by an API key, `/checkpoint` (which writes to disk) is not.
- **Principle**: *Security*.
- **Impact**: Potential for unauthorized clients to overwrite checkpoints or fill disk space.
