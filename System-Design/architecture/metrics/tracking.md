# Metric Tracking Architecture

This document describes the metric tracking and logging systems used in the Pokemon Emerald Speedrun codebase, focusing on performance, cost analysis, and milestone progress.

## Overview

The metric tracking system provides comprehensive visibility into the agent's performance, resource usage (tokens/cost), and game progress. It operates at multiple levels of granularity: per-step, per-milestone, and per-objective.

## 1. Core Components

### Cumulative Metrics (`cumulative_metrics.json`)
- **Purpose**: Stores the aggregated metrics for the current run.
- **Location**: `.pokeagent_cache/{run_id}/cumulative_metrics.json`
- **Structure**:
  ```json
  {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cached_tokens": 0,
    "total_cost": 0.0,
    "total_actions": 0,
    "total_llm_calls": 0,
    "total_run_time": 0,
    "steps": [],        // Detailed step-by-step metrics (unbounded; all steps preserved)
    "milestones": [],   // Milestones with cumulative + delta metrics
    "objectives": []    // Objectives with cumulative + delta metrics
  }
  ```
- **Management**: Handled by the `LLMLogger` class in `utils/llm_logger.py`.

### LLMLogger (`utils/llm_logger.py`)
- **Role**: Centralized singleton for tracking LLM interactions and game metrics.
- **Functionality**:
  - **Logging**: Records every LLM interaction (prompt, response, usage) to `llm_logs/llm_log_{session_id}.jsonl`.
  - **Aggregation**: Updates cumulative metrics in memory and persists them to disk after each interaction.
  - **Checkpointing**: Saves metrics to `cumulative_metrics.json` and logs/state to `checkpoint_llm.txt`. Metrics are restored from JSON, not the text checkpoint.
  - **Cost Calculation**: Applies model-specific pricing (e.g., Claude vs. Gemini rates) including cache hit discounts.

### Metric Granularity

#### 1. Step-Level Metrics
- Tracks tokens, cost, time, and tool calls for individual agent steps.
- **Note**: All steps are preserved (no truncation). Long runs may produce large files.

#### 2. Milestone Metrics
- Triggered when a significant game event occurs (e.g., obtaining a badge, entering a new area).
- **Data**: Records cumulative totals *plus* the delta (change) since the last milestone, allowing analysis of "cost per milestone".

#### 3. Objective Metrics
- Triggered when the agent completes a high-level objective (e.g., "Defeat Gym Leader Roxanne").
- **Data**: Similar to milestones, providing "cost per objective" breakdowns.

## 2. CLI Agent Usage Monitoring (External Agents)

External CLI agents (run via `run_cli.py`) run in a separate process or container and do not integrate directly with `LLMLogger`. Usage is derived from backend-specific sources and synced to the server via the abstract `log_cli_interaction()` method.

### Single-Writer Pattern
- **Server** is the only process that writes `cumulative_metrics.json` (via `LLM_METRICS_WRITE_ENABLED=true`).
- **run_cli** sets `LLM_METRICS_WRITE_ENABLED=false` and accumulates steps in memory, then POSTs to the server.

### Backend-Specific Metric Sources

| Backend | Source | Reader Module | Dedup Strategy |
|---------|--------|--------------|----------------|
| Claude Code | JSONL files in `claude_memory/projects/-workspace/` | `utils/metric_tracking/claude_jsonl_reader.py` | Best-entry by message ID (highest total tokens) |
| Gemini CLI | Telemetry outfile (`gemini_memory/telemetry.jsonl`) | `utils/metric_tracking/gemini_telemetry_reader.py` | SHA-256 hash of (prompt_id, timestamp, model, tokens) with byte-offset incremental reads |

### Flow (Common)
1. **Polling**: `run_cli` calls `backend.log_cli_interaction()` every 15 seconds and once after each session exits.
2. **In-Memory Accumulation**: For each new entry, `append_cli_step()` updates `LLMLogger.cumulative_metrics` in memory.
3. **Sync to Server**: `_sync_metrics_to_server()` POSTs the full cumulative metrics to `POST /sync_llm_metrics`.

### Claude-Specific Details
- **Best-Entry Dedup**: Claude Code emits multiple entries per API call (streaming chunks with `output_tokens=0` and a final entry with complete usage). We keep the highest total per hash.
- **JSONL source**: `.pokeagent_cache/{run_id}/claude_memory/projects/-workspace/*.jsonl`

### Gemini-Specific Details
- **Telemetry source**: `gemini_cli.api_response` events in the OTEL outfile, each representing one API request.
- **Token mapping**: `input_token_count` → prompt, `output_token_count + thoughts_token_count + tool_token_count` → completion, `cached_content_token_count` → cached.
- **Efficient polling**: Uses file byte offsets to read only new data appended since last poll.
- **Configuration**: Telemetry is enabled automatically by writing `settings.json` with `telemetry.enabled: true` and `telemetry.outfile`.

### Pricing
- Claude: `claude-sonnet-4-6` / `claude-sonnet-4.6` pricing. Cache writes: $3.75/M, cache hits: $0.30/M.
- Gemini: `gemini-2.5-pro` / `gemini-2.5-flash` pricing already present in `LLMLogger.pricing`.

### Pre-Scaffold Agents (e.g., `MyCLIAgent`)
- **Stream-JSON**: Output raw JSON lines to stdout/stderr; captured into `run_data/{run_id}/agent_logs/session_{NNN}.jsonl`.
- **Integration**: Sync metrics to the server via `/sync_llm_metrics`.

### Autonomous Agents
- **Direct Logging**: `AutonomousCLIAgent` integrates directly with `LLMLogger`, ensuring all internal thought processes and tool calls are structured and stored in the main `llm_log.jsonl`.

## 3. Areas for Improvement

**CLI Agent Metrics Discrepancy**
- The cumulative statistics reported in agent logs (e.g. `[cli:result] cost=$4.52` from the session stream) can differ from values in `cumulative_metrics.json`.
- The session result event has authoritative `total_cost_usd` and `usage` from the API, but we derive metrics from JSONL polling.
- **Possible causes**: Output token undercount from streaming chunks (mitigated by best-entry dedup but not eliminated), missing subagent JSONL, or pricing differences (e.g. 1h vs 5m cache write TTL).
- **Improvement**: Using the result event's totals at session end to correct or replace the JSONL-derived cumulative values would improve accuracy.

**Incremental Polling Token Undercount (Potential Improvement)**
- **Issue**: Best-entry dedup only works within a single poll. Claude Code emits a streaming chunk (low/zero `output_tokens`) first, then a final entry with full usage. If poll 1 sees the streaming chunk, we add it and add its hash to `processed_hashes`. When poll 2 sees the final entry (same hash), we skip it. Result: ~5–15% undercount of completion tokens.
- **Proposed fix**: Support *updates* when a later poll sees a higher-total entry for an already-processed hash. Changes required:
  1. **Reader** (`claude_jsonl_reader.py`): Replace `processed_hashes: set` with `processed_best: dict[hash, total]`; return `(new_entries, update_entries, new_processed_best)`.
  2. **Caller** (`run_cli.py`): Maintain `hash_to_step_info: dict[hash, (step_number, token_usage)]`; for update entries, call `update_cli_step()` to correct the step and cumulative totals.
  3. **LLM logger** (`llm_logger.py`): Add `update_cli_step(step_number, old_usage, new_usage)` that subtracts old values, adds new values, and updates the step in `steps`.

## 4. Software Engineering Principles Deviation

**Manual Write Control (Concurrency Risk)**
- **Issue**: Writing to `cumulative_metrics.json` is controlled manually via the `LLM_METRICS_WRITE_ENABLED` environment variable.
- **Principle**: *Concurrency Safety* / *Automation*.
- **Impact**: Requires careful orchestration to avoid race conditions in multi-process environments. A proper locking mechanism or database (SQLite) would be safer.

**Limited Aggregation (Data Silos)**
- **Issue**: Metrics are siloed per run. There is no built-in utility to aggregate or compare metrics across multiple runs (e.g., "average cost to beat Roxanne over 10 runs").
- **Principle**: *Observability* / *DRITW (Don't Re-Invent The Wheel)*.
- **Impact**: Developers must write custom scripts to perform cross-run analysis.

**Unbounded Log Growth (Scalability)**
- **Issue**: Both `steps` in cumulative_metrics.json and the main `llm_log.jsonl` file grow indefinitely.
- **Principle**: *Scalability*.
- **Impact**: Long runs can produce massive log files (GBs), making them difficult to open or process. Log rotation or splitting should be implemented.

**Hardcoded Pricing (Maintainability)**
- **Issue**: Model pricing is often hardcoded within `LLMLogger` or related utility functions.
- **Principle**: *Single Source of Truth* / *Configuration*.
- **Impact**: Updates to API pricing require code changes. Pricing should ideally be loaded from an external configuration file or fetched dynamically.
