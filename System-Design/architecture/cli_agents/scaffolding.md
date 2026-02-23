# CLI Agents Architecture (Scaffolding Agents)

This document describes the architecture of the **CLI Agents**, which are designed to wrap powerful external model scaffolding (like Anthropic's Claude Code or OpenAI's tools) into a Pokemon-playing context. The primary example is `MyCLIAgent` (`agent/my_cli_agent.py`).

## Overview

Unlike the `AutonomousCLIAgent`, which manages its own objectives and loops entirely, CLI Agents are often designed to integrate with or mimic the structure of provider-specific scaffolding. They focus on executing a specific task or maintaining a session state that aligns with how these models expect to interact (e.g., via a REPL or structured conversation).

## 1. Core Architecture

### Composition over Inheritance
Similar to the autonomous agent, `MyCLIAgent` uses composition:
- **Agent Class**: `MyCLIAgent` encapsulates the game-specific logic.
- **VLM Wrapper**: It composes a `VLM` instance (configured for Anthropic or OpenAI) to handle the LLM interaction.
- **Tool Adapter**: It uses `MCPToolAdapter` to interface with the game server.

### Key Components

#### Provider Wrapping (`MyCLIAgent` specific features)
- **Anthropic/OpenAI Backends**: The agent initializes a `VLM` backend that wraps the respective provider's SDK (`anthropic.Anthropic`, `openai.OpenAI`).
- **Tool Conversion**:
  - **Anthropic**: Tools are converted to the `input_schema` format (JSON Schema with `type: "object"`).
  - **OpenAI**: Tools are converted to the `function` declaration format.
- **System Prompt Caching**: For Anthropic, the agent leverages `cache_control` blocks in the system prompt to optimize performance and cost.

#### Frame Buffering
- **Implementation**: `_sample_frames_loop()` runs in a background thread.
- **Purpose**: Continuously captures game frames from the server to maintain a recent history (buffer) without blocking the main agent loop.
- **Usage**: When the agent needs to make a decision, it can access the latest frames from the buffer to understand the current game state, including motion or recent changes.

#### Game-Specific State Tracking
- **Internal State**: The agent maintains its own tracking of game-specific events:
  - `defeated_trainers`: Set of trainer IDs defeated.
  - `blocked_coords`: Coordinates where movement was blocked.
  - `turnstile_states`: Tracking specific interactable objects.
- **Purpose**: To provide context to the LLM that isn't immediately visible in a single frame (e.g., "I already fought this trainer").

## 2. Agent Loop

1. **Initialization**: Connects to the server, initializes the VLM backend, and starts the frame buffering thread.
2. **Observation**: Retrieves the current game state and recent frames from the buffer.
3. **Prompt Construction**: Builds a prompt with the current state, recent actions, and relevant game knowledge.
4. **Interaction**: Sends the prompt to the VLM backend.
5. **Execution**:
   - If the model requests a tool call (e.g., `press_buttons`), the agent executes it via the `MCPToolAdapter`.
   - If the model provides text, it is logged/displayed.
6. **State Update**: Updates internal tracking (e.g., location, defeated trainers) based on the action result.

## 3. Software Engineering Principles Deviation

**Code Duplication (DRY Violation)**
- **Issue**: `MyCLIAgent` shares a significant amount of boilerplate code with `AutonomousCLIAgent` (VLM setup, tool adapter usage, basic loop structure).
- **Principle**: *Don't Repeat Yourself (DRY)*.
- **Impact**: Maintenance overhead; changes to the core agent logic must be applied in two places.

**Mixed Responsibilities (Separation of Concerns)**
- **Issue**: The agent class handles LLM interaction, tool execution, *and* detailed game state tracking (defeated trainers, turnstiles).
- **Principle**: *Single Responsibility Principle*.
- **Impact**: The agent class becomes a "God Object" for the session. Game state tracking logic should likely be extracted into a `GameStateTracker` or similar component.

**Hardcoded State Logic**
- **Issue**: Specific game logic (like turnstile handling or trainer tracking) is hardcoded within the agent.
- **Principle**: *Open/Closed Principle*.
- **Impact**: Adding support for new game mechanics requires modifying the agent class directly.

**Inconsistent Tool Usage**
- **Issue**: `MyCLIAgent` uses a simpler/different set of tools compared to `AutonomousCLIAgent`.
- **Principle**: *Consistency*.
- **Impact**: Agents have different capabilities, making it harder to compare performance or swap them out for the same task.

**Thread Safety Concerns**
- **Issue**: The frame buffering thread shares state (`latest_frames`, `frame_buffer`) with the main loop. While Python's GIL offers some protection, reliance on shared mutable state without explicit locking in the agent can lead to race conditions or inconsistent reads.
- **Principle**: *Concurrency / Safety*.
- **Impact**: Potential for "glitched" observations where the agent sees a torn frame or inconsistent state.
