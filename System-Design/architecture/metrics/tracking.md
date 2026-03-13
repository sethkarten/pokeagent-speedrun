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
| Gemini CLI | Session JSON in `gemini_memory/tmp/workspace/chats/session-*.json` | `utils/metric_tracking/gemini_session_reader.py` | Message ID (globally unique) |
| Codex CLI | Rollout JSONL in `codex_memory/sessions/YYYY/MM/DD/rollout-*.jsonl` | `utils/metric_tracking/codex_session_reader.py` | Session file path + cumulative token total (per rollout) |

### Flow (Common)
1. **Polling**: `run_cli` calls `backend.log_cli_interaction()` every 15 seconds and once after each session exits.
2. **In-Memory Accumulation**: For each new entry, `append_cli_step()` updates `LLMLogger.cumulative_metrics` in memory.
3. **Sync to Server**: `_sync_metrics_to_server()` POSTs the full cumulative metrics to `POST /sync_llm_metrics`.

### Claude-Specific Details
- **Best-Entry Dedup**: Claude Code emits multiple entries per API call (streaming chunks with `output_tokens=0` and a final entry with complete usage). We keep the highest total per hash.
- **JSONL source**: `.pokeagent_cache/{run_id}/claude_memory/projects/-workspace/*.jsonl`

### Gemini-Specific Details
- **Session source**: `tmp/workspace/chats/session-*.json` files (analogous to Claude JSONL). Reader: `utils/metric_tracking/gemini_session_reader.py`.
- **Token mapping**: `input` → prompt, `output + thoughts + tool` → completion, `cached` → cached.
- **Implicit caching**: Gemini CLI uses *implicit caching* (automatic, no explicit cache creation API). Cache creation cost is included in the first request's input tokens. The session format does not expose `cache_write`; we store `cache_write_tokens: null` in step entries for Gemini runs. **This is expected**—plotting/analytics should treat null as "not applicable" rather than zero.
- **Cost calculation (subset vs distinct)**: Certain APIS reports prompt and cached as *additive* (distinct). Gemini, in particular, reports prompt = total input with cached as a *subset*. `LLMLogger` detects subset when `cached + cache_write <= prompt` and computes `uncached = prompt - cached - cache_write` for correct billing. Both schemes produce correct costs. cumulatove totals are still consistent for the purposes of graphing!

### Codex-Specific Details
- **Source**: Codex stores one append-only rollout file per session under `~/.codex/sessions/YYYY/MM/DD/rollout-<uuid>.jsonl` (see [Codex CLI features](https://developers.openai.com/codex/cli/features)). Our cache mounts this as `.pokeagent_cache/{run_id}/codex_memory/sessions/`.
- **Resume behavior**: `codex resume` / `codex exec resume --last` continues the *same* session; new events are appended to the same rollout file. No new file is created for a resumed session.
- **Subagents**: In Codex multi-agent workflows, subagent activity is typically embedded in the parent session’s rollout (same JSONL stream). A separate rollout file is only created when a new Codex process/session is started (e.g. a script runs another `codex exec`, or a different workspace starts its own session).
- **Dedup**: We have no per-response message ID in the rollout. Steps are inferred from `event_msg` with `payload.type === "token_count"`. We deduplicate by `(session_file_path, cumulative_total_tokens, compaction_offset)` so that multiple token_count snapshots for the same API call collapse into one step. Tool calls are taken from `response_item` with `payload.type === "function_call"` and attached to the next advancing token snapshot.
- **Compaction robustness**: When `cumulative_total` decreases (e.g. after [context compaction](https://github.com/openai/codex/pull/3446)), we increment `compaction_offset` and include it in the dedup hash. Post-compaction steps thus get fresh hashes and are not mis-deduplicated with pre-compaction steps. If totals remain monotonic, `compaction_offset` stays 0.
- **Token totals**: Codex/OpenRouter reports `cached_input_tokens` as a *subset* of `input_tokens`. We use the API's `total_tokens` when available to avoid double-counting cached tokens in step totals.

### Potential problems for Codex (turn vs message pattern)
Codex rollout logs use a **turn/task** and **event stream** model rather than discrete **messages** with stable IDs. That leads to the following caveats:

1. **No per-response message ID**: Claude and Gemini expose a unique message (or request) ID per API response, so dedup is globally stable. Codex rollout events do not expose an equivalent; we rely on `(session_key, cumulative_total, compaction_offset)` as a synthetic step key. Compaction is handled by incrementing `compaction_offset` in case cumulative total token usage decreaseses in codex session log with compaction.
2. **Cumulative-total monotonicity**: If cumulative token usage decreases (e.g. after compaction), [context compaction](https://github.com/openai/codex/pull/3446), we increment `compaction_offset` so post-compaction steps get distinct hashes and are not mis-deduplicated. Note [context compaction](https://github.com/openai/codex/pull/3446) suggests that `compaction_offset` may not be needed, though we include it for robustness if this is not the case. 
3. **Same-file ordering**: Subagent and parent events are interleaved in one file. We process the file sequentially and attach buffered tool calls to the next advancing token snapshot. If event order or token_count placement changes in future CLI versions, step boundaries could shift.
4. **Restart of run_cli**: `processed_hashes` is in-memory only. If `run_cli` is restarted against the same run cache, we re-read all rollout files and would re-append steps unless we persist processed hashes (or derive “already applied” from existing `cumulative_metrics.json` steps).

These are documented so that future changes to Codex CLI (compaction, multi-agent logging, or new event shapes) can be checked against this contract.

### Pricing
- Claude: `claude-sonnet-4-6` / `claude-sonnet-4.6` pricing. Cache writes: $3.75/M, cache hits: $0.30/M.
- Gemini: `gemini-2.5-pro` / `gemini-2.5-flash` pricing already present in `LLMLogger.pricing`.

### Legacy Custom Agents (historical)
- **Stream-JSON**: Output raw JSON lines to stdout/stderr; captured into `run_data/{run_id}/agent_logs/session_{NNN}.jsonl`.
- **Integration**: Sync metrics to the server via `/sync_llm_metrics`.

### Autonomous Agents
- **Direct Logging**: `PokeAgent` integrates directly with `LLMLogger`, ensuring all internal thought processes and tool calls are structured and stored in the main `llm_log.jsonl`.

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
