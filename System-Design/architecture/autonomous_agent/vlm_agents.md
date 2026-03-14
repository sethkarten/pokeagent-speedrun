# Autonomous Agent Architecture (VLM-based)

This document describes the architecture of the custom autonomous agents in the repository, specifically focusing on `PokeAgent` (`agents/custom/PokeAgent.py`) and its integration with Vision-Language Models (VLMs).

## Overview

The `PokeAgent` represents a significant evolution from simple reactive agents. It is designed to operate autonomously by formulating its own objectives, maintaining a persistent knowledge base, and utilizing a wide range of tools to interact with the game environment. **Entry path**: `pokeagent`, `autonomous_cli`, and `vision_only` are selected in `run.py` via `start_custom_agent()` and do not go through the unified `agents.Agent` in `agents/__init__.py`; `react`, `claudeplays`, and `geminiplays` still use `server/client.py` and `agents.Agent`.

## 1. Core Architecture

### Composition over Inheritance
The agent uses a **composition-based architecture** rather than deep inheritance hierarchies.
- **Agent Class**: `PokeAgent` encapsulates the agent's logic.
- **VLM Wrapper**: It composes a `VLM` instance (from `utils/agent_infrastructure/vlm_backends.py`) to handle all LLM interactions.
- **Tool Adapter**: It uses `MCPToolAdapter` to interface with the game server's MCP endpoints.

### Key Components

#### VLM Integration (`utils/agent_infrastructure/vlm_backends.py`)
- **Facade Pattern**: The `VLM` class acts as a unified facade over multiple backend providers (Google Gemini, Anthropic Claude, OpenAI GPT-4, etc.).
- **Backend Abstraction**: All backends inherit from the `VLMBackend` abstract base class, ensuring a consistent interface (`get_query`, `get_text_query`) regardless of the underlying model.
- **Tool Format Conversion**: The VLM layer automatically converts tool definitions into the specific format required by the chosen provider (e.g., Gemini's `FunctionDeclaration` vs. OpenAI's `tools` schema).

#### Tool Management
- **Dynamic Tool Creation**: Tools are defined programmatically within the agent itself and composed with MCP helpers.
- **MCP Mapping**: The agent maps high-level actions (e.g., "battle", "explore") to low-level MCP server calls via the `MCPToolAdapter`.

#### Prompt Engineering & Optimization
- **System Instructions**: The agent constructs complex system prompts that include:
  - Role definition (Pokemon expert).
  - Current game state (context).
  - Objective history.
  - Knowledge base summaries.
- **Prompt Optimization**: Utilizes a `PromptOptimizer` class (in `agents/custom/utils/prompt_optimizer.py`, custom-agent only) to refine prompts based on past performance or specific constraints (though implementation details vary).

## 2. Agent Loop

1. **Observation**: The agent captures the current game state (screenshot + metadata) from the server.
2. **Context Construction**: It builds a prompt including the image, recent history, and current objectives.
3. **Reasoning (LLM)**: The VLM processes the input and generates a response, which may include text (thought process) and/or tool calls.
4. **Execution**:
   - If tools are called, the agent executes them via the `MCPToolAdapter`.
   - Results are fed back into the conversation history.
5. **Memory Management**: The agent maintains a conversation history, compacting it periodically to stay within context window limits.

## 3. Software Engineering Principles Deviation

**Code Duplication (DRY Violation)**
- **Issue**: Significant logic overlap historically existed between `PokeAgent` and the removed `MyCLIAgent` scaffolds (e.g., VLM initialization, tool handling, basic loop structure).
- **Principle**: *Don't Repeat Yourself (DRY)*.
- **Impact**: Bug fixes or improvements in one agent must be manually ported to the other. A shared base class or mixin strategy would reduce this maintenance burden.

**Backend Implementation Leakage (Abstraction Leak)**
- **Issue**: Some agent code (e.g., in `PokeAgent`) directly imports backend-specific types (like `google.generativeai.types`) or makes assumptions about specific model behaviors.
- **Principle**: *Dependency Inversion* / *Abstraction*.
- **Impact**: Switching backends (e.g., Gemini to Claude) might require code changes in the agent itself, defeating the purpose of the generic `VLM` wrapper.

**Inconsistent Tool Formats**
- **Issue**: `PokeAgent` tends to favor Gemini-style tool definitions (`type_`, `ARRAY`, `OBJECT`), while others use JSON Schema.
- **Principle**: *Consistency*.
- **Impact**: Increases cognitive load when working across different agents and makes the tool definitions less portable.

**Complex Error Handling (Complexity)**
- **Issue**: Error handling is decentralized; some is in the agent loop, some in the VLM wrapper, and some in the adapter. Timeout handling is particularly specific in `PokeAgent`.
- **Principle**: *Keep It Simple Stupid (KISS)* / *Error Handling*.
- **Impact**: Debugging failures (e.g., network timeouts vs. model refusals) is difficult due to scattered try/except blocks.
