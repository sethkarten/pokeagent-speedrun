#!/usr/bin/env python3
"""
LLM Logger utility for logging all VLM interactions

This module provides a centralized logging system for all LLM interactions,
including input prompts, responses, and metadata. Logs are saved to dated
files in the llm_logs directory.
"""

from operator import ge
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LLMLogger:
    """Logger for all LLM interactions"""
    
    def __init__(self, log_dir: str = "llm_logs", session_id: Optional[str] = None):
        """Initialize the LLM logger
        
        Args:
            log_dir: Directory to store log files
            session_id: Optional session ID. If None, checks environment variable LLM_SESSION_ID.
                        If still None, generates one from current time.
                        This ensures consistent logging across processes.
        """
        self.log_dir = log_dir
        
        # Check for session_id from environment variable (for multiprocess consistency)
        if session_id is None:
            session_id = os.environ.get("LLM_SESSION_ID")
        
        # Generate new session_id if still None
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_id = session_id
        self.log_file = os.path.join(log_dir, f"llm_log_{self.session_id}.jsonl")
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize cumulative metrics with enhanced structure
        self.cumulative_metrics = {
            # Top-level aggregate metrics (backward compatible)
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,  # NEW: Track cached tokens separately
            "cache_write_tokens": 0,  # NEW: Track cache-write tokens separately (when provider exposes it)
            "total_cost": 0.0,
            "total_actions": 0,
            "start_time": time.time(),
            "total_llm_calls": 0,
            "total_run_time": 0,  # Actual gameplay time in seconds
            "last_update_time": time.time(),  # Track when we last updated
            "metadata": {},
            
            # NEW: Per-step granular tracking
            "steps": [],  # List of {step, prompt_tokens, completion_tokens, cached_tokens, cache_write_tokens, time_taken, timestamp}
            
            # NEW: Per-milestone tracking
            "milestones": [],  # List of milestone completion data with cumulative and split metrics
            
            # Per-objective tracking (direct objective completions)
            "objectives": [],  # List of {objective_id, category, objective_index, + same metrics as milestones}
            
            # Internal tracking for milestone deltas
            "_last_milestone_step": 0,
            "_last_milestone_tokens": {"prompt": 0, "completion": 0, "total": 0, "cached": 0},
            "_last_milestone_time": None,
            # Internal tracking for objective deltas (split = since last objective)
            "_last_objective_step": 0,
            "_last_objective_tokens": {"prompt": 0, "completion": 0, "total": 0, "cached": 0},
            "_last_objective_time": None,
        }
        
        # Model pricing (per 1K tokens)
        self.pricing = {
            # OpenAI GPT-5
            "gpt-5": {"prompt": 0.00125, "completion": 0.01, "cached_prompt": 0.000125},       # $1.25/$10 per 1M, cached input $0.125 per 1M
            "gpt-5-mini": {"prompt": 0.00025, "completion": 0.002, "cached_prompt": 0.000025},  # $0.25/$2 per 1M, cached input $0.025 per 1M
            "gpt-5-nano": {"prompt": 0.00005, "completion": 0.0004, "cached_prompt": 0.000005},  # $0.05/$0.40 per 1M, cached input $0.005 per 1M
            "gpt-5.1": {"prompt": 0.00125, "completion": 0.01, "cached_prompt": 0.000125},       # $1.25/$10 per 1M, cached input $0.125 per 1M
            "gpt-5.2": {"prompt": 0.00175, "completion": 0.014, "cached_prompt": 0.000175},       # $1.75/$14 per 1M, cached input $0.175 per 1M
            "gpt-5.2-pro": {"prompt": 0.021, "completion": 0.168},     # $21/$168 per 1M (no cached)
            "gpt-5-pro": {"prompt": 0.015, "completion": 0.12},        # $15/$120 per 1M (no cached)
            # GPT-5 chat-latest variants
            "gpt-5.2-chat-latest": {"prompt": 0.00175, "completion": 0.014, "cached_prompt": 0.000175},
            "gpt-5.1-chat-latest": {"prompt": 0.00125, "completion": 0.01, "cached_prompt": 0.000125},
            "gpt-5-chat-latest": {"prompt": 0.00125, "completion": 0.01, "cached_prompt": 0.000125},
            # GPT-5 codex variants
            "gpt-5.2-codex": {"prompt": 0.00175, "completion": 0.014, "cached_prompt": 0.000175},
            "gpt-5.1-codex-max": {"prompt": 0.00125, "completion": 0.01, "cached_prompt": 0.000125},
            "gpt-5.1-codex": {"prompt": 0.00125, "completion": 0.01, "cached_prompt": 0.000125},
            "gpt-5-codex": {"prompt": 0.00125, "completion": 0.01, "cached_prompt": 0.000125},

            # OpenAI GPT-4 (Input / Cached input / Output per 1M from official pricing)
            "gpt-4o": {"prompt": 0.0025, "completion": 0.01, "cached_prompt": 0.00125},       # $2.50/$1.25/$10 per 1M
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006, "cached_prompt": 0.000075},  # $0.15/$0.075/$0.60 per 1M
            "gpt-4.1": {"prompt": 0.002, "completion": 0.008, "cached_prompt": 0.0005},      # $2/$0.50/$8 per 1M
            "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016, "cached_prompt": 0.0001},  # $0.40/$0.10/$1.60 per 1M
            "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004, "cached_prompt": 0.000025},  # $0.10/$0.025/$0.40 per 1M

            # OpenAI o-series (reasoning)
            "o4-mini": {"prompt": 0.0011, "completion": 0.0044, "cached_prompt": 0.000275},    # $1.10/$0.275/$4.40 per 1M
            "o3-mini": {"prompt": 0.0011, "completion": 0.0044, "cached_prompt": 0.00055},   # $1.10/$0.55/$4.40 per 1M
            "o3": {"prompt": 0.002, "completion": 0.008, "cached_prompt": 0.0005},           # $2/$0.50/$8 per 1M
            "o3-pro": {"prompt": 0.02, "completion": 0.08},         # $20/$80 per 1M (no cached)
            "o1": {"prompt": 0.015, "completion": 0.06, "cached_prompt": 0.0075},            # $15/$7.50/$60 per 1M
            "o1-pro": {"prompt": 0.15, "completion": 0.60},         # $150/$600 per 1M (no cached)

            # Anthropic Claude (Base input / 5m cache write / Cache hits per MTok from official pricing)
            "claude-sonnet-4-6": {"prompt": 0.003, "completion": 0.015, "cached_prompt": 0.0003, "cache_write_prompt": 0.00375},   # Claude Code CLI model string uses dashes
            "claude-sonnet-4.6": {"prompt": 0.003, "completion": 0.015, "cached_prompt": 0.0003, "cache_write_prompt": 0.00375},
            "claude-sonnet-4.5": {"prompt": 0.003, "completion": 0.015, "cached_prompt": 0.0003, "cache_write_prompt": 0.00375},   # $3/$3.75/$0.30/$15 per 1M
            "claude-sonnet-4": {"prompt": 0.003, "completion": 0.015, "cached_prompt": 0.0003, "cache_write_prompt": 0.00375},
            "claude-sonnet-3.7": {"prompt": 0.003, "completion": 0.015, "cached_prompt": 0.0003, "cache_write_prompt": 0.00375},
            "claude-sonnet-3.5": {"prompt": 0.003, "completion": 0.015, "cached_prompt": 0.0003, "cache_write_prompt": 0.00375},
            "claude-opus-4.1": {"prompt": 0.015, "completion": 0.075, "cached_prompt": 0.0015, "cache_write_prompt": 0.01875},   # $15/$18.75/$1.50/$75 per 1M
            "claude-opus-4": {"prompt": 0.015, "completion": 0.075, "cached_prompt": 0.0015, "cache_write_prompt": 0.01875},
            "claude-opus-3": {"prompt": 0.015, "completion": 0.075, "cached_prompt": 0.0015, "cache_write_prompt": 0.01875},
            "claude-haiku-4.5": {"prompt": 0.001, "completion": 0.005, "cached_prompt": 0.0001, "cache_write_prompt": 0.00125},   # $1/$1.25/$0.10/$5 per 1M
            "claude-haiku-3.5": {"prompt": 0.0008, "completion": 0.004, "cached_prompt": 0.00008, "cache_write_prompt": 0.001},  # $0.80/$1/$0.08/$4 per 1M
            "claude-haiku-3": {"prompt": 0.00025, "completion": 0.00125, "cached_prompt": 0.00003, "cache_write_prompt": 0.0003},  # $0.25/$0.30/$0.03/$1.25 per 1M

            # Gemini 2.x (Input / Output / Context caching per 1M from official pricing)
            "gemini-2.5-flash": {"prompt": 0.0003, "completion": 0.0025, "cached_prompt": 0.00003},  # $0.30/$2.50/$0.03 per 1M
            "gemini-2.5-flash-preview-09-2025": {"prompt": 0.0003, "completion": 0.0025, "cached_prompt": 0.00003},
            "gemini-2.5-pro": {"prompt": 0.00125, "completion": 0.01, "cached_prompt": 0.000125},     # $1.25/$10/$0.125 per 1M (<=200k)
            "gemini-2.5-flash-lite": {"prompt": 0.0001, "completion": 0.0004, "cached_prompt": 0.00001},  # $0.10/$0.40/$0.01 per 1M
            "gemini-2.5-flash-lite-preview-09-2025": {"prompt": 0.0001, "completion": 0.0004, "cached_prompt": 0.00001},
            "gemini-2.0-flash": {"prompt": 0.00015, "completion": 0.0006},  # $0.15/$0.60 per 1M (no cached in snippet)

            # Gemini 3.x
            "gemini-3-pro-preview": {"prompt": 0.002, "completion": 0.012, "cached_prompt": 0.0002},  # $2/$12/$0.20 per 1M (<=200k)
            "gemini-3-pro": {"prompt": 0.002, "completion": 0.012, "cached_prompt": 0.0002},
            "gemini-3-flash": {"prompt": 0.0005, "completion": 0.003, "cached_prompt": 0.00005},     # $0.50/$3/$0.05 per 1M (gemini-3-flash-preview)
            "gemini-3-flash-preview": {"prompt": 0.0005, "completion": 0.003, "cached_prompt": 0.00005},

            "default": {"prompt": 0.001, "completion": 0.002}  # Default pricing
        }
        
        # Initialize log file with session info
        self._log_session_start()

        # ALWAYS try to load cumulative metrics immediately on initialization
        # This prevents overwriting with zeros before metrics are loaded
        metrics_loaded = self.load_cumulative_metrics()
        if metrics_loaded:
            logger.info(f"✅ LLM Logger initialized with loaded cumulative metrics: tokens={self.cumulative_metrics.get('total_tokens', 0):,}, cost=${self.cumulative_metrics.get('total_cost', 0):.2f}")
        else:
            logger.warning(f"⚠️  LLM Logger initialized WITHOUT cumulative metrics - starting fresh")

        logger.info(f"LLM Logger initialized: {self.log_file}")

    def _metrics_write_enabled(self) -> bool:
        """Return True if this process should write cumulative metrics.

        Controlled by LLM_METRICS_WRITE_ENABLED env var.
        - unset: default to True (backward compatible)
        - "false"/"0"/"no": disable writes in this process
        """
        flag = os.environ.get("LLM_METRICS_WRITE_ENABLED")
        if flag is None:
            return True
        return flag.strip().lower() not in ("0", "false", "no")

    def _ensure_metrics_structure(self) -> None:
        """Ensure cumulative_metrics has all required fields (for loading old format)."""
        if "steps" not in self.cumulative_metrics:
            self.cumulative_metrics["steps"] = []
        if "milestones" not in self.cumulative_metrics:
            self.cumulative_metrics["milestones"] = []
        if "metadata" not in self.cumulative_metrics:
            self.cumulative_metrics["metadata"] = {}
        if "objectives" not in self.cumulative_metrics:
            self.cumulative_metrics["objectives"] = []
        if "cache_write_tokens" not in self.cumulative_metrics:
            self.cumulative_metrics["cache_write_tokens"] = 0

    def _log_session_start(self):
        """Log session start information"""
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "type": "session_start",
            "session_id": self.session_id,
            "log_file": self.log_file
        }
        self._write_log_entry(session_info)
    
    def log_interaction(self, 
                       interaction_type: str,
                       prompt: str,
                       response: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       duration: Optional[float] = None,
                       model_info: Optional[Dict[str, Any]] = None,
                       step_number: Optional[int] = None):
        """Log a complete LLM interaction
        
        Args:
            interaction_type: Type of interaction (e.g., "perception", "planning", "action")
            prompt: The input prompt sent to the LLM
            response: The response received from the LLM
            metadata: Additional metadata about the interaction
            duration: Time taken for the interaction in seconds
            model_info: Information about the model used
            step_number: Optional step number for per-step tracking
        """
        effective_step = step_number
        if effective_step is None:
            env_step = os.environ.get("LLM_STEP_NUMBER")
            if env_step and str(env_step).isdigit():
                effective_step = int(env_step)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "interaction",
            "interaction_type": interaction_type,
            "prompt": prompt,
            "response": response,
            "duration": duration,
            "metadata": metadata or {},
            "model_info": model_info or {},
        }
        if effective_step is not None:
            log_entry["agent_step"] = effective_step

        self._write_log_entry(log_entry)

        # Update cumulative metrics
        self.cumulative_metrics["total_llm_calls"] += 1

        # Update gameplay time (only count time when actually interacting)
        current_time = time.time()
        time_since_last_update = current_time - self.cumulative_metrics["last_update_time"]
        # Only add time if it's reasonable (less than 5 minutes since last interaction)
        # This prevents counting idle/stopped time
        if time_since_last_update < 300:  # 5 minutes
            self.cumulative_metrics["total_run_time"] += time_since_last_update
        self.cumulative_metrics["last_update_time"] = current_time
        
        # Track token usage if available
        step_tokens = {"prompt": 0, "completion": 0, "total": 0, "cached": 0, "cache_write": 0}
        if metadata and "token_usage" in metadata:
            token_usage = metadata["token_usage"]
            if token_usage:
                cw = token_usage.get("cache_write_tokens")
                step_tokens = {
                    "prompt": token_usage.get("prompt_tokens", 0),
                    "completion": token_usage.get("completion_tokens", 0),
                    "total": token_usage.get("total_tokens", 0),
                    "cached": token_usage.get("cached_tokens", 0),
                    "cache_write": cw,  # None for Gemini implicit caching; store as-is for step_entry
                }
                cw_for_math = int(cw or 0)

                self.cumulative_metrics["total_tokens"] += step_tokens["total"]
                self.cumulative_metrics["prompt_tokens"] += step_tokens["prompt"]
                self.cumulative_metrics["completion_tokens"] += step_tokens["completion"]
                self.cumulative_metrics["cached_tokens"] += step_tokens["cached"]
                self.cumulative_metrics["cache_write_tokens"] += cw_for_math
                
                # Calculate cost based on model (exact match first, then longest-key match)
                model_name = (model_info.get("model", "") or "").lower()
                pricing = self.pricing.get("default")
                if model_name:
                    # Prefer exact match
                    if model_name in self.pricing:
                        pricing = self.pricing[model_name]
                    else:
                        # Fall back to longest matching key (gpt-5-nano before gpt-5)
                        candidates = [
                            (k, self.pricing[k])
                            for k in self.pricing
                            if k != "default" and k in model_name
                        ]
                        if candidates:
                            # Sort by key length descending; use most specific match
                            candidates.sort(key=lambda x: len(x[0]), reverse=True)
                            pricing = candidates[0][1]
                
                prompt_tokens = max(0, step_tokens["prompt"])
                cached_tokens = max(0, step_tokens["cached"])
                cache_write_tokens = max(0, cw_for_math)

                # Providers report cache buckets in two shapes:
                # 1) Subset (Gemini): prompt_tokens = total input; cached is a SUBSET. uncached = prompt - cached.
                # 2) Distinct (Claude & others): prompt, cached, cache_write are ADDITIVE. uncached = prompt.
                if (cached_tokens + cache_write_tokens) <= prompt_tokens:
                    # Subset style (OpenAI/OpenRouter/Gemini style in most responses)
                    uncached_prompt_tokens = prompt_tokens - cached_tokens - cache_write_tokens
                    cached_prompt_tokens_billable = cached_tokens
                    cache_write_tokens_billable = cache_write_tokens
                else:
                    # Distinct style (Anthropic can report large cache_write/cache_read outside input_tokens)
                    uncached_prompt_tokens = prompt_tokens
                    cached_prompt_tokens_billable = cached_tokens
                    cache_write_tokens_billable = cache_write_tokens

                cached_prompt_rate = pricing.get("cached_prompt", pricing["prompt"])
                cache_write_prompt_rate = pricing.get("cache_write_prompt", pricing["prompt"])
                prompt_cost = (
                    (uncached_prompt_tokens / 1000) * pricing["prompt"]
                    + (cached_prompt_tokens_billable / 1000) * cached_prompt_rate
                    + (cache_write_tokens_billable / 1000) * cache_write_prompt_rate
                )
                completion_cost = (step_tokens["completion"] / 1000) * pricing["completion"]
                self.cumulative_metrics["total_cost"] += prompt_cost + completion_cost
        
        # NEW: Track per-step metrics (effective_step from arg or LLM_STEP_NUMBER)
        if effective_step is not None and duration is not None:
            step_entry = {
                "step": effective_step,
                "prompt_tokens": step_tokens["prompt"],
                "completion_tokens": step_tokens["completion"],
                "cached_tokens": step_tokens["cached"],
                "cache_write_tokens": step_tokens["cache_write"],  # None for Gemini (implicit caching)
                "total_tokens": step_tokens["total"],
                "time_taken": round(duration, 3),
                "timestamp": time.time()
            }
            self.cumulative_metrics["steps"].append(step_entry)

        # Save metrics to cache file after every interaction
        self.save_cumulative_metrics()
        
        # Also log to console for debugging
        if duration is not None:
            logger.info(f"LLM {interaction_type.upper()}: {duration:.2f}s")
            logger.debug(f"Prompt length: {len(prompt)} chars, Response length: {len(response)} chars")
        else:
            logger.info(f"LLM {interaction_type.upper()}")

    def append_cli_step(
        self,
        step_number: int,
        token_usage: Dict[str, Any],
        duration: float,
        timestamp: float,
        model_info: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[list] = None,
    ) -> None:
        """Append one step derived from a Claude Code JSONL entry into cumulative_metrics.

        Unlike log_interaction this method does NOT write to llm_log.jsonl – the
        JSONL files produced by Claude Code already serve as the raw interaction
        log.  Only cumulative_metrics.json is updated.

        In-memory metrics are ALWAYS updated regardless of LLM_METRICS_WRITE_ENABLED so
        that run_cli can accumulate steps in memory and forward them to the server via
        /sync_llm_metrics even when disk writes are disabled.

        Args:
            step_number:  Monotonically increasing step index for this run.
            token_usage:  Dict with keys prompt, completion, cached, cache_write, total, cost.
            duration:     Seconds elapsed since the previous JSONL entry (best estimate).
            timestamp:    UNIX timestamp of the JSONL entry.
            model_info:   Optional dict with at least a "model" key for pricing lookup.
            tool_calls:   Optional list of {name, args} dicts from the assistant message.
        """
        # NOTE: intentionally not gated by _metrics_write_enabled() here – in-memory
        # update must happen even when disk writes are off so the sync path works.

        step_tokens = {
            "prompt": int(token_usage.get("prompt", 0) or 0),
            "completion": int(token_usage.get("completion", 0) or 0),
            "cached": int(token_usage.get("cached", 0) or 0),
            # cache_write may be None (e.g. Gemini implicit caching); use 0 for cumulative math
            "cache_write": token_usage.get("cache_write")
            if token_usage.get("cache_write") is not None
            else None,
            "total": int(token_usage.get("total", 0) or 0),
        }
        cache_write_for_math = int(step_tokens["cache_write"] or 0)

        # Update cumulative token counters
        self.cumulative_metrics["total_tokens"] += step_tokens["total"]
        self.cumulative_metrics["prompt_tokens"] += step_tokens["prompt"]
        self.cumulative_metrics["completion_tokens"] += step_tokens["completion"]
        self.cumulative_metrics["cached_tokens"] += step_tokens["cached"]
        self.cumulative_metrics["cache_write_tokens"] += cache_write_for_math
        self.cumulative_metrics["total_llm_calls"] += 1

        # Cost calculation
        # If token_usage provides explicit cost (e.g. from OpenRouter), use that.
        explicit_cost = float(token_usage.get("cost") or 0.0)
        
        if explicit_cost > 0:
            step_cost = explicit_cost
        else:
            # Fallback to local pricing logic
            model_name = ((model_info or {}).get("model", "") or "").lower()
            pricing = self.pricing.get("default")
            if model_name:
                if model_name in self.pricing:
                    pricing = self.pricing[model_name]
                else:
                    candidates = [
                        (k, self.pricing[k])
                        for k in self.pricing
                        if k != "default" and k in model_name
                    ]
                    if candidates:
                        candidates.sort(key=lambda x: len(x[0]), reverse=True)
                        pricing = candidates[0][1]

            prompt_t = max(0, step_tokens["prompt"])
            cached_t = max(0, step_tokens["cached"])
            cache_write_t = max(0, cache_write_for_math)

            if (cached_t + cache_write_t) <= prompt_t:
                uncached_prompt = prompt_t - cached_t - cache_write_t
            else:
                uncached_prompt = prompt_t

            step_cost = (
                (uncached_prompt / 1000) * pricing["prompt"]
                + (cached_t / 1000) * pricing.get("cached_prompt", pricing["prompt"])
                + (cache_write_t / 1000) * pricing.get("cache_write_prompt", pricing["prompt"])
                + (step_tokens["completion"] / 1000) * pricing["completion"]
            )
        
        self.cumulative_metrics["total_cost"] += step_cost

        step_entry: Dict[str, Any] = {
            "step": step_number,
            "prompt_tokens": step_tokens["prompt"],
            "completion_tokens": step_tokens["completion"],
            "cached_tokens": step_tokens["cached"],
            "cache_write_tokens": step_tokens["cache_write"],  # None for Gemini (implicit caching)
            "total_tokens": step_tokens["total"],
            "time_taken": round(duration, 3),
            "timestamp": timestamp,
        }
        if tool_calls:
            cleaned = [
                {"name": c["name"], "args": c.get("args", {})}
                for c in tool_calls
                if c.get("name")
            ]
            if cleaned:
                step_entry["tool_calls"] = cleaned

        self.cumulative_metrics["steps"].append(step_entry)
        # Only write to disk when this process owns the file (server in single-writer mode).
        # run_cli keeps steps in memory and syncs to server via /sync_llm_metrics.
        if self._metrics_write_enabled():
            self.save_cumulative_metrics()
        logger.debug(
            "CLI step %d appended: tokens=%d cost=$%.5f tools=%d",
            step_number,
            step_tokens["total"],
            step_cost,
            len(step_entry.get("tool_calls", [])),
        )

    def log_error(self, 
                  interaction_type: str,
                  prompt: str,
                  error: str,
                  metadata: Optional[Dict[str, Any]] = None):
        """Log an LLM interaction error
        
        Args:
            interaction_type: Type of interaction that failed
            prompt: The input prompt that was sent
            error: The error message
            metadata: Additional metadata about the error
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "interaction_type": interaction_type,
            "prompt": prompt,
            "error": error,
            "metadata": metadata or {}
        }
        
        self._write_log_entry(log_entry)
        logger.error(f"LLM {interaction_type.upper()} ERROR: {error}")
    
    def log_step_start(self, step: int, step_type: str = "agent_step"):
        """Log the start of an agent step
        
        Args:
            step: Step number
            step_type: Type of step (e.g., "agent_step", "perception", "planning")
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "step_start",
            "step": step,
            "step_type": step_type
        }
        
        self._write_log_entry(log_entry)
        logger.info(f"Starting {step_type} {step}")
    
    def log_step_end(self, step: int, step_type: str = "agent_step", 
                    duration: Optional[float] = None,
                    summary: Optional[str] = None):
        """Log the end of an agent step
        
        Args:
            step: Step number
            step_type: Type of step
            duration: Time taken for the step
            summary: Summary of what happened in the step
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "step_end",
            "step": step,
            "step_type": step_type,
            "duration": duration,
            "summary": summary
        }
        
        self._write_log_entry(log_entry)
        if duration:
            logger.info(f"Completed {step_type} {step} in {duration:.2f}s")
        else:
            logger.info(f"Completed {step_type} {step}")
    
    def log_state_snapshot(self, state_data: Dict[str, Any], step: int):
        """Log a snapshot of the game state
        
        Args:
            state_data: The game state data
            step: Current step number
        """
        # Extract key information to avoid logging too much data
        state_summary = {
            "step": step,
            "player_location": state_data.get("player", {}).get("location"),
            "player_position": state_data.get("player", {}).get("position"),
            "game_state": state_data.get("game", {}).get("game_state"),
            "is_in_battle": state_data.get("game", {}).get("is_in_battle"),
            "party_size": len(state_data.get("player", {}).get("party", [])),
            "money": state_data.get("game", {}).get("money"),
            "dialog_text": state_data.get("game", {}).get("dialog_text", "")[:100] + "..." if state_data.get("game", {}).get("dialog_text") else None
        }
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "state_snapshot",
            "step": step,
            "state_summary": state_summary
        }
        
        self._write_log_entry(log_entry)
    
    def log_action(self, action: str, step: int, reasoning: Optional[str] = None):
        """Log an action taken by the agent
        
        Args:
            action: The action taken
            step: Current step number
            reasoning: Reasoning behind the action
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "action",
            "step": step,
            "action": action,
            "reasoning": reasoning
        }
        
        self._write_log_entry(log_entry)
        logger.info(f"Action {step}: {action}")
        if reasoning:
            logger.debug(f"Reasoning: {reasoning}")
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write a log entry to the log file
        
        Args:
            log_entry: The log entry to write
        """
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write log entry: {e}")

    def log_thinking(
        self,
        text: str,
        interaction_type: str = "thinking",
        duration: float = 0,
        agent_step: Optional[int] = None,
    ) -> None:
        """Log agent thinking for UI streaming (same format as log_interaction, no metrics update).
        Used by CLI agent and any path that POSTs thinking to /agent_step so the SSE has one source.
        """
        step_val = agent_step
        if step_val is None:
            env_step = os.environ.get("LLM_STEP_NUMBER")
            if env_step and str(env_step).isdigit():
                step_val = int(env_step)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "interaction",
            "interaction_type": interaction_type,
            "prompt": "",
            "response": text,
            "duration": duration,
            "metadata": {},
            "model_info": {},
        }
        if step_val is not None:
            log_entry["agent_step"] = step_val
        self._write_log_entry(log_entry)
    
    def increment_action_count(self, count: int = 1):
        """Increment the action counter (called when buttons are actually queued)
        
        Args:
            count: Number of actions to add (default 1)
        """
        self.cumulative_metrics["total_actions"] += count
        # Note: We don't save immediately here to avoid excessive I/O
        # Metrics will be saved on next log_interaction or explicit save
    
    def log_milestone_completion(self, milestone_id: str, step_number: int, timestamp: float = None):
        """Log milestone completion with cumulative and split metrics
        
        Args:
            milestone_id: ID of the completed milestone
            step_number: Current agent step number
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate cumulative metrics (running totals)
        cumulative_step = step_number
        cumulative_prompt_tokens = self.cumulative_metrics.get("prompt_tokens", 0)
        cumulative_completion_tokens = self.cumulative_metrics.get("completion_tokens", 0)
        cumulative_cached_tokens = self.cumulative_metrics.get("cached_tokens", 0)
        cumulative_total_tokens = self.cumulative_metrics.get("total_tokens", 0)
        
        # Calculate split metrics (delta since last milestone)
        last_step = self.cumulative_metrics.get("_last_milestone_step", 0)
        last_tokens = self.cumulative_metrics.get("_last_milestone_tokens", {"prompt": 0, "completion": 0, "total": 0, "cached": 0})
        last_time = self.cumulative_metrics.get("_last_milestone_time")
        
        split_steps = cumulative_step - last_step
        split_prompt_tokens = cumulative_prompt_tokens - last_tokens.get("prompt", 0)
        split_completion_tokens = cumulative_completion_tokens - last_tokens.get("completion", 0)
        split_cached_tokens = cumulative_cached_tokens - last_tokens.get("cached", 0)
        split_total_tokens = cumulative_total_tokens - last_tokens.get("total", 0)
        
        # Calculate time elapsed for this milestone
        if last_time is not None:
            time_elapsed = timestamp - last_time
        else:
            time_elapsed = timestamp - self.cumulative_metrics.get("start_time", timestamp)
        
        # Create milestone entry
        milestone_entry = {
            "milestone_id": milestone_id,
            "timestamp": timestamp,
            
            # Cumulative metrics (running total up to this milestone)
            "cumulative_steps": cumulative_step,
            "cumulative_prompt_tokens": cumulative_prompt_tokens,
            "cumulative_completion_tokens": cumulative_completion_tokens,
            "cumulative_cached_tokens": cumulative_cached_tokens,
            "cumulative_total_tokens": cumulative_total_tokens,
            
            # Split metrics (delta for just this milestone)
            "split_steps": split_steps,
            "split_prompt_tokens": split_prompt_tokens,
            "split_completion_tokens": split_completion_tokens,
            "split_cached_tokens": split_cached_tokens,
            "split_total_tokens": split_total_tokens,
            "split_time_seconds": round(time_elapsed, 2)
        }
        
        self.cumulative_metrics["milestones"].append(milestone_entry)
        
        # Update tracking for next milestone
        self.cumulative_metrics["_last_milestone_step"] = cumulative_step
        self.cumulative_metrics["_last_milestone_tokens"] = {
            "prompt": cumulative_prompt_tokens,
            "completion": cumulative_completion_tokens,
            "cached": cumulative_cached_tokens,
            "total": cumulative_total_tokens
        }
        self.cumulative_metrics["_last_milestone_time"] = timestamp
        
        logger.info(f"📊 Milestone '{milestone_id}': {split_steps} steps, {split_total_tokens} tokens, {time_elapsed:.1f}s")
        
        # Save updated metrics
        self.save_cumulative_metrics()

    def log_objective_completion(
        self,
        objective_id: str,
        category: str,
        objective_index: int,
        step_number: int,
        timestamp: Optional[float] = None,
    ):
        """Log direct objective completion with cumulative and split metrics.

        Appends to cumulative_metrics["objectives"] with the same metric shape as
        milestones plus objective_id, category, and objective_index. Split metrics
        are deltas since the last objective completion.

        Args:
            objective_id: ID of the completed objective.
            category: Category (e.g. story, battling, dynamics, legacy).
            objective_index: Index of the objective within its sequence.
            step_number: Current agent step number.
            timestamp: Optional timestamp (defaults to current time).
        """
        if timestamp is None:
            timestamp = time.time()

        cumulative_step = step_number
        cumulative_prompt_tokens = self.cumulative_metrics.get("prompt_tokens", 0)
        cumulative_completion_tokens = self.cumulative_metrics.get("completion_tokens", 0)
        cumulative_cached_tokens = self.cumulative_metrics.get("cached_tokens", 0)
        cumulative_total_tokens = self.cumulative_metrics.get("total_tokens", 0)

        last_step = self.cumulative_metrics.get("_last_objective_step", 0)
        last_tokens = self.cumulative_metrics.get(
            "_last_objective_tokens", {"prompt": 0, "completion": 0, "total": 0, "cached": 0}
        )
        last_time = self.cumulative_metrics.get("_last_objective_time")

        split_steps = cumulative_step - last_step
        split_prompt_tokens = cumulative_prompt_tokens - last_tokens.get("prompt", 0)
        split_completion_tokens = cumulative_completion_tokens - last_tokens.get("completion", 0)
        split_cached_tokens = cumulative_cached_tokens - last_tokens.get("cached", 0)
        split_total_tokens = cumulative_total_tokens - last_tokens.get("total", 0)

        if last_time is not None:
            time_elapsed = timestamp - last_time
        else:
            time_elapsed = timestamp - self.cumulative_metrics.get("start_time", timestamp)

        objective_entry = {
            "objective_id": objective_id,
            "category": category,
            "objective_index": objective_index,
            "timestamp": timestamp,
            "cumulative_steps": cumulative_step,
            "cumulative_prompt_tokens": cumulative_prompt_tokens,
            "cumulative_completion_tokens": cumulative_completion_tokens,
            "cumulative_cached_tokens": cumulative_cached_tokens,
            "cumulative_total_tokens": cumulative_total_tokens,
            "split_steps": split_steps,
            "split_prompt_tokens": split_prompt_tokens,
            "split_completion_tokens": split_completion_tokens,
            "split_cached_tokens": split_cached_tokens,
            "split_total_tokens": split_total_tokens,
            "split_time_seconds": round(time_elapsed, 2),
        }

        if "objectives" not in self.cumulative_metrics:
            self.cumulative_metrics["objectives"] = []
        self.cumulative_metrics["objectives"].append(objective_entry)

        self.cumulative_metrics["_last_objective_step"] = cumulative_step
        self.cumulative_metrics["_last_objective_tokens"] = {
            "prompt": cumulative_prompt_tokens,
            "completion": cumulative_completion_tokens,
            "cached": cumulative_cached_tokens,
            "total": cumulative_total_tokens,
        }
        self.cumulative_metrics["_last_objective_time"] = timestamp

        logger.info(
            f"📊 Objective '{objective_id}' (category={category}, index={objective_index}): "
            f"{split_steps} steps, {split_total_tokens} tokens, {time_elapsed:.1f}s"
        )
        self.save_cumulative_metrics()

    def get_cumulative_metrics(self) -> Dict[str, Any]:
        """Get cumulative metrics for the session

        Returns:
            Dictionary with cumulative metrics
        """
        # Don't recalculate runtime - it's tracked per-interaction now
        return self.cumulative_metrics.copy()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session
        
        Returns:
            Dictionary with session summary information
        """
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            interactions = 0
            errors = 0
            total_duration = 0.0
            
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "interaction":
                        interactions += 1
                        if entry.get("duration"):
                            total_duration += entry["duration"]
                    elif entry.get("type") == "error":
                        errors += 1
                except json.JSONDecodeError:
                    continue
            
            return {
                "session_id": self.session_id,
                "log_file": self.log_file,
                "total_interactions": interactions,
                "total_errors": errors,
                "total_duration": total_duration,
                "average_duration": total_duration / interactions if interactions > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {"error": str(e)}
    
    def save_cumulative_metrics(self, metrics_file: str = None):
        """Save just the cumulative metrics to a lightweight cache file.

        Single-writer pattern: only the server writes (LLM_METRICS_WRITE_ENABLED=true).
        run_cli accumulates steps in-memory and syncs to the server via /sync_llm_metrics.
        """
        try:
            if not self._metrics_write_enabled():
                logger.debug("Metrics write disabled; skipping save_cumulative_metrics()")
                return

            if metrics_file is None:
                from utils.data_persistence.run_data_manager import get_cache_path
                metrics_file = str(get_cache_path("cumulative_metrics.json"))

            metrics_to_save = self.cumulative_metrics.copy()
            for key in list(metrics_to_save.keys()):
                if key.startswith("_"):
                    del metrics_to_save[key]

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_to_save, f, indent=2)

            logger.debug(f"Saved cumulative metrics to {metrics_file}")

        except Exception as e:
            logger.error(f"Failed to save cumulative metrics: {e}")

    def load_cumulative_metrics(self, metrics_file: str = None) -> bool:
        """Load cumulative metrics from cache file

        Args:
            metrics_file: Path to load metrics from (defaults to cache folder)

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Determine metrics file path
            if metrics_file is None:
                from utils.data_persistence.run_data_manager import get_cache_path
                metrics_file = str(get_cache_path("cumulative_metrics.json"))

            if not os.path.exists(metrics_file):
                logger.debug(f"No cumulative metrics file found at {metrics_file}")
                return False

            # Load metrics
            with open(metrics_file, 'r', encoding='utf-8') as f:
                saved_metrics = json.load(f)

            # Update current metrics (including new arrays if present)
            self.cumulative_metrics.update(saved_metrics)
            self._ensure_metrics_structure()

            # Restore internal tracking from last milestone if available
            if self.cumulative_metrics["milestones"]:
                last_milestone = self.cumulative_metrics["milestones"][-1]
                self.cumulative_metrics["_last_milestone_step"] = last_milestone.get("cumulative_steps", 0)
                self.cumulative_metrics["_last_milestone_tokens"] = {
                    "prompt": last_milestone.get("cumulative_prompt_tokens", 0),
                    "completion": last_milestone.get("cumulative_completion_tokens", 0),
                    "cached": last_milestone.get("cumulative_cached_tokens", 0),
                    "total": last_milestone.get("cumulative_total_tokens", 0)
                }
                self.cumulative_metrics["_last_milestone_time"] = last_milestone.get("timestamp")

            # Restore internal tracking from last objective if available
            if self.cumulative_metrics.get("objectives"):
                last_obj = self.cumulative_metrics["objectives"][-1]
                self.cumulative_metrics["_last_objective_step"] = last_obj.get("cumulative_steps", 0)
                self.cumulative_metrics["_last_objective_tokens"] = {
                    "prompt": last_obj.get("cumulative_prompt_tokens", 0),
                    "completion": last_obj.get("cumulative_completion_tokens", 0),
                    "cached": last_obj.get("cumulative_cached_tokens", 0),
                    "total": last_obj.get("cumulative_total_tokens", 0),
                }
                self.cumulative_metrics["_last_objective_time"] = last_obj.get("timestamp")

            logger.info(f"✅ Loaded cumulative metrics: {saved_metrics.get('total_llm_calls', 0)} calls, {saved_metrics.get('total_actions', 0)} actions, ${saved_metrics.get('total_cost', 0):.4f}, {saved_metrics.get('total_run_time', 0):.0f}s runtime")
            logger.info(f"   - Total tokens: {saved_metrics.get('total_tokens', 0):,}")
            logger.info(f"   - Total cost: ${saved_metrics.get('total_cost', 0):.4f}")
            logger.info(f"   - Steps tracked: {len(self.cumulative_metrics['steps'])}, Milestones tracked: {len(self.cumulative_metrics['milestones'])}")
            return True

        except Exception as e:
            logger.error(f"Failed to load cumulative metrics: {e}")
            return False

    def set_run_metadata(self, metadata: Dict[str, Any], overwrite: bool = False):
        """Set run metadata in cumulative metrics.

        Args:
            metadata: Run metadata to store under cumulative_metrics["metadata"]
            overwrite: If True, replace existing metadata. Otherwise only fill missing keys.
        """
        if not metadata:
            return
        current = self.cumulative_metrics.get("metadata", {})
        if overwrite or not current:
            self.cumulative_metrics["metadata"] = metadata
        else:
            for key, value in metadata.items():
                if key not in current:
                    current[key] = value
            self.cumulative_metrics["metadata"] = current
        self.save_cumulative_metrics()

    def add_step_tool_calls(self, step_number: int, tool_calls: list[Dict[str, Any]]):
        """Attach tool call parameters to a step entry in cumulative metrics."""
        if step_number is None:
            return
        if "steps" not in self.cumulative_metrics:
            self.cumulative_metrics["steps"] = []

        # Keep only name + args to avoid oversized entries
        cleaned_calls = []
        for call in tool_calls or []:
            name = call.get("name")
            args = call.get("args", {})
            if name:
                cleaned_calls.append({"name": name, "args": args})

        # Update existing step entry if present
        for entry in reversed(self.cumulative_metrics["steps"]):
            if entry.get("step") == step_number:
                entry["tool_calls"] = cleaned_calls
                self.save_cumulative_metrics()
                return

        # If no step entry exists, append a minimal one
        self.cumulative_metrics["steps"].append(
            {"step": step_number, "tool_calls": cleaned_calls, "timestamp": time.time()}
        )
        self.save_cumulative_metrics()

    def save_checkpoint(self, checkpoint_file: str = None, agent_step_count: int = None):
        """Save current LLM interaction history to checkpoint file
        
        Args:
            checkpoint_file: Path to save the checkpoint (defaults to cache folder)
            agent_step_count: Current agent step count for persistence
        """
        try:
            # Use cache folder by default
            if checkpoint_file is None or checkpoint_file == "checkpoint_llm.txt":
                from utils.data_persistence.run_data_manager import get_cache_path
                checkpoint_file = str(get_cache_path("checkpoint_llm.txt"))
            # Read all current log entries
            log_entries = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            
            # Don't recalculate run time - it's already tracked accurately per-interaction

            # Add checkpoint metadata (cumulative_metrics stored only in cumulative_metrics.json)
            checkpoint_data = {
                "checkpoint_timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "original_log_file": self.log_file,
                "total_entries": len(log_entries),
                "agent_step_count": agent_step_count,
                "log_entries": log_entries
            }
            
            # Add map stitcher data if available via callback
            if hasattr(self, '_map_stitcher_callback') and self._map_stitcher_callback:
                try:
                    self._map_stitcher_callback(checkpoint_data)
                except Exception as e:
                    logger.debug(f"Failed to save map stitcher to checkpoint: {e}")
            
            # Save to checkpoint file
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"LLM checkpoint saved: {checkpoint_file} ({len(log_entries)} entries)")
            
        except Exception as e:
            logger.error(f"Failed to save LLM checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str = None) -> Optional[int]:
        """Load LLM interaction history from checkpoint file
        
        Args:
            checkpoint_file: Path to load the checkpoint from (defaults to cache folder)
            
        Returns:
            Last agent step count from the checkpoint, or None if not found
        """
        try:
            # Use cache folder by default
            if checkpoint_file is None or checkpoint_file == "checkpoint_llm.txt":
                from utils.data_persistence.run_data_manager import get_cache_path
                checkpoint_file = str(get_cache_path("checkpoint_llm.txt"))
            
            if not os.path.exists(checkpoint_file):
                logger.info(f"No checkpoint file found at {checkpoint_file}")
                return None
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            log_entries = checkpoint_data.get("log_entries", [])

            # Cumulative metrics are loaded from cumulative_metrics.json only (not from checkpoint)

            # Restore log entries to current log file
            with open(self.log_file, 'w', encoding='utf-8') as f:
                for entry in log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Try to get step count from checkpoint metadata first
            last_step = checkpoint_data.get("agent_step_count")
            
            # If not in metadata, find the last agent step from log entries
            if last_step is None:
                for entry in reversed(log_entries):
                    if entry.get("type") == "step_start" and "step_number" in entry:
                        last_step = entry["step_number"]
                        break
            
            logger.info(f"LLM checkpoint loaded: {checkpoint_file} ({len(log_entries)} entries, step {last_step})")
            
            # Load map stitcher data if available via callback
            if hasattr(self, '_map_stitcher_load_callback') and self._map_stitcher_load_callback:
                try:
                    self._map_stitcher_load_callback(checkpoint_data)
                except Exception as e:
                    logger.debug(f"Failed to load map stitcher from checkpoint: {e}")
            
            return last_step
            
        except Exception as e:
            logger.error(f"Failed to load LLM checkpoint: {e}")
            return None

# Global logger instance
_llm_logger = None

def get_llm_logger() -> LLMLogger:
    """Get the global LLM logger instance
    
    Returns:
        The global LLM logger instance
    """
    global _llm_logger
    if _llm_logger is None:
        # Check for session_id from environment (set by client for multiprocess consistency)
        session_id = os.environ.get("LLM_SESSION_ID")
        _llm_logger = LLMLogger(session_id=session_id)
    return _llm_logger

def setup_map_stitcher_checkpoint_integration(memory_reader):
    """Set up map stitcher integration with checkpoint system"""
    logger = get_llm_logger()
    
    def save_callback(checkpoint_data):
        if hasattr(memory_reader, '_map_stitcher') and memory_reader._map_stitcher:
            memory_reader._map_stitcher.save_to_checkpoint(checkpoint_data)
    
    def load_callback(checkpoint_data):
        if hasattr(memory_reader, '_map_stitcher') and memory_reader._map_stitcher:
            memory_reader._map_stitcher.load_from_checkpoint(checkpoint_data)
    
    logger._map_stitcher_callback = save_callback
    logger._map_stitcher_load_callback = load_callback

def log_llm_interaction(interaction_type: str, prompt: str, response: str, 
                       metadata: Optional[Dict[str, Any]] = None,
                       duration: Optional[float] = None,
                       model_info: Optional[Dict[str, Any]] = None,
                       step_number: Optional[int] = None):
    """Convenience function to log an LLM interaction
    
    Args:
        interaction_type: Type of interaction
        prompt: Input prompt
        response: LLM response
        metadata: Additional metadata
        duration: Time taken
        model_info: Model information
        step_number: Optional step number for per-step tracking
    """
    logger = get_llm_logger()
    logger.log_interaction(interaction_type, prompt, response, metadata, duration, model_info, step_number)


def log_llm_error(interaction_type: str, prompt: str, error: str, 
                 metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to log an LLM error
    
    Args:
        interaction_type: Type of interaction that failed
        prompt: Input prompt
        error: Error message
        metadata: Additional metadata
    """
    logger = get_llm_logger()
    logger.log_error(interaction_type, prompt, error, metadata)

def increment_action_count(count: int = 1):
    """Convenience function to increment action count from anywhere
    
    Args:
        count: Number of actions to add
    """
    logger = get_llm_logger()
    logger.increment_action_count(count)

def log_milestone_completion(milestone_id: str, step_number: int, timestamp: float = None):
    """Convenience function to log milestone completion
    
    Args:
        milestone_id: ID of the milestone
        step_number: Current agent step number
        timestamp: Optional timestamp
    """
    logger = get_llm_logger()
    logger.log_milestone_completion(milestone_id, step_number, timestamp)


def log_objective_completion(
    objective_id: str,
    category: str,
    objective_index: int,
    step_number: int,
    timestamp: Optional[float] = None,
):
    """Convenience function to log direct objective completion.

    Args:
        objective_id: ID of the completed objective.
        category: Category (e.g. story, battling, dynamics, legacy).
        objective_index: Index of the objective within its sequence.
        step_number: Current agent step number.
        timestamp: Optional timestamp.
    """
    get_llm_logger().log_objective_completion(
        objective_id, category, objective_index, step_number, timestamp
    ) 