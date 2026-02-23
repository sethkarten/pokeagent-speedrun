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
    "steps": [],        // Detailed step-by-step metrics (last 1000)
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
  - **Checkpointing**: Saves/loads metrics alongside emulator state.
  - **Cost Calculation**: Applies model-specific pricing (e.g., Claude vs. Gemini rates) including cache hit discounts.

### Metric Granularity

#### 1. Step-Level Metrics
- Tracks tokens, cost, time, and tool calls for individual agent steps.
- **Limitation**: Only the last 1000 steps are kept in the JSON to prevent unlimited file growth.

#### 2. Milestone Metrics
- Triggered when a significant game event occurs (e.g., obtaining a badge, entering a new area).
- **Data**: Records cumulative totals *plus* the delta (change) since the last milestone, allowing analysis of "cost per milestone".

#### 3. Objective Metrics
- Triggered when the agent completes a high-level objective (e.g., "Defeat Gym Leader Roxanne").
- **Data**: Similar to milestones, providing "cost per objective" breakdowns.

## 2. CLI Agent Logging

### Pre-Scaffold Agents (e.g., `MyCLIAgent`)
- **Stream-JSON**: These agents typically output raw JSON lines to stdout/stderr.
- **Session Logs**: The `RunDataManager` captures these streams into `run_data/{run_id}/agent_logs/session_{NNN}.jsonl`.
- **Integration**: While they log raw data, they also sync key metrics (like total actions) to the server via `/sync_llm_metrics`.

### Autonomous Agents
- **Direct Logging**: `AutonomousCLIAgent` integrates directly with `LLMLogger`, ensuring all internal thought processes and tool calls are structured and stored in the main `llm_log.jsonl`.

## 3. Software Engineering Principles Deviation

**Manual Write Control (Concurrency Risk)**
- **Issue**: Writing to `cumulative_metrics.json` is controlled manually via the `LLM_METRICS_WRITE_ENABLED` environment variable.
- **Principle**: *Concurrency Safety* / *Automation*.
- **Impact**: Requires careful orchestration to avoid race conditions in multi-process environments. A proper locking mechanism or database (SQLite) would be safer.

**Limited Aggregation (Data Silos)**
- **Issue**: Metrics are siloed per run. There is no built-in utility to aggregate or compare metrics across multiple runs (e.g., "average cost to beat Roxanne over 10 runs").
- **Principle**: *Observability* / *DRITW (Don't Re-Invent The Wheel)*.
- **Impact**: Developers must write custom scripts to perform cross-run analysis.

**Unbounded Log Growth (Scalability)**
- **Issue**: While `steps` are capped at 1000, the main `llm_log.jsonl` file grows indefinitely.
- **Principle**: *Scalability*.
- **Impact**: Long runs can produce massive log files (GBs), making them difficult to open or process. Log rotation or splitting should be implemented.

**Hardcoded Pricing (Maintainability)**
- **Issue**: Model pricing is often hardcoded within `LLMLogger` or related utility functions.
- **Principle**: *Single Source of Truth* / *Configuration*.
- **Impact**: Updates to API pricing require code changes. Pricing should ideally be loaded from an external configuration file or fetched dynamically.
