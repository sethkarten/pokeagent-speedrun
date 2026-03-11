"""
Reader for Codex CLI session JSONL files (analogous to claude_jsonl_reader, gemini_session_reader).

Reads session files under agent_memory_dir/sessions/ (Codex stores under ~/.codex/sessions/).
Supports:
- event_msg with payload.type "token_count" (input_tokens, cached_input_tokens, output_tokens, reasoning_tokens)
- turn_context (model, turn metadata)
- turn.completed-like usage in stream-written session logs

Best-effort parsing: gracefully skips unknown shapes, never raises.
Ref: https://github.com/openai/codex/issues/9660, ccusage/codex, takopi exec-json-cheatsheet
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSIONS_REL = "sessions"


def find_session_files(data_path: Path) -> list[Path]:
    """Return JSONL files under data_path/sessions/, sorted by mtime (most recent first).
    Codex stores sessions in nested dirs like sessions/2026/03/10/rollout-*.jsonl."""
    sessions_dir = Path(data_path).resolve() / SESSIONS_REL
    if not sessions_dir.is_dir():
        # Fallback: look for jsonl directly under data_path
        if data_path.is_dir():
            files = list(Path(data_path).rglob("*.jsonl"))
            return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        return []
    files = list(sessions_dir.rglob("*.jsonl")) + list(sessions_dir.rglob("*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _parse_timestamp(ts: Any) -> datetime | None:
    """Parse ISO-8601 or UNIX timestamp to UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            pass
    return None


def _create_step_hash(
    session_key: str,
    cumulative_total: int | None,
    parsed_ts: datetime | None,
    line_idx: int,
    compaction_offset: int = 0,
) -> str:
    """Stable dedup key for one logical Codex usage step (a bit more complex due to how session rollout log emits data).

    If cumulative_total decreases (e.g. after context compaction), compaction_offset
    increments so post-compaction steps get fresh hashes and are not deduplicated.
    """
    if cumulative_total is not None:
        return f"{session_key}:usage:{cumulative_total}:compact:{compaction_offset}"
    ts = parsed_ts.isoformat() if parsed_ts is not None else f"line-{line_idx}"
    return f"{session_key}:usage-ts:{ts}:compact:{compaction_offset}"


def _normalize_tool_args(raw_args: Any) -> dict[str, Any]:
    """Best-effort normalization for tool-call args."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def extract_usage_snapshot(entry: dict) -> tuple[dict[str, Any] | None, int | None]:
    """Extract step token usage and cumulative total from a Codex entry.

    Token semantics (OpenAI/OpenRouter): cached_input_tokens is a SUBSET of
    input_tokens. total_tokens = input_tokens + output_tokens (reasoning is
    included in output). We use the API's total_tokens when available to avoid
    double-counting cached tokens.
    """
    payload = entry.get("payload")
    if isinstance(payload, dict) and payload.get("type") == "token_count":
        info = payload.get("info") or {}
        total_usage = info.get("total_token_usage") or {}
        usage = info.get("last_token_usage") or total_usage or payload
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        reasoning = int(usage.get("reasoning_output_tokens", 0) or 0)
        # Prefer API total (cached is subset of input; adding cached double-counts)
        api_total = usage.get("total_tokens") or total_usage.get("total_tokens")
        total = int(api_total) if api_total is not None else (inp + out + reasoning)
        if total == 0 and inp == 0 and out == 0:
            return None, None
        cumulative_total = total_usage.get("total_tokens")
        if cumulative_total is None:
            cumulative_total = usage.get("total_tokens")
        cumulative_total_int = int(cumulative_total or total)
        return (
            {
                "prompt": inp,
                "completion": out + reasoning,
                "cached": cached,
                "cache_write": 0,
                "total": total,
                "cost": 0.0,
            },
            cumulative_total_int,
        )

    usage = entry.get("usage")
    if isinstance(usage, dict):
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        reasoning = int(usage.get("reasoning_output_tokens", 0) or 0)
        api_total = usage.get("total_tokens")
        total = int(api_total) if api_total is not None else (inp + out + reasoning)
        if total == 0 and inp == 0 and out == 0:
            return None, None
        cumulative_total = int(usage.get("total_tokens", 0) or total)
        return (
            {
                "prompt": inp,
                "completion": out + reasoning,
                "cached": cached,
                "cache_write": 0,
                "total": total,
                "cost": 0.0,
            },
            cumulative_total,
        )

    return None, None


def extract_tokens_from_entry(entry: dict) -> dict[str, Any] | None:
    """Extract token counts from a Codex session entry.

    Supports:
    - event_msg with payload.type token_count (input_tokens, cached_input_tokens, output_tokens, reasoning_tokens)
    - usage object (input_tokens, output_tokens, cached_input_tokens) from turn.completed
    """
    tokens, _ = extract_usage_snapshot(entry)
    return tokens


def extract_tool_calls_from_entry(entry: dict) -> list[dict[str, Any]]:
    """Extract tool/MCP calls from entry if present. Best-effort."""
    tool_calls: list[dict[str, Any]] = []
    payload = entry.get("payload")
    if isinstance(payload, dict) and payload.get("type") == "mcp_tool_call":
        tool = payload.get("tool")
        args = _normalize_tool_args(payload.get("arguments"))
        if tool:
            tool_calls.append({"name": tool, "args": args})
    elif (
        entry.get("type") == "response_item"
        and isinstance(payload, dict)
        and payload.get("type") == "function_call"
    ):
        tool = payload.get("name")
        args = _normalize_tool_args(payload.get("arguments"))
        if tool:
            tool_calls.append({"name": tool, "args": args})
    return tool_calls


def load_new_usage_entries(
    data_path: Path,
    processed_hashes: set[str],
) -> tuple[list[dict], set[str]]:
    """Load Codex session entries that have not been processed.

    Scans data_path/sessions/ for JSONL files. Parses event_msg (token_count),
    usage-bearing entries. Deduplicates by hash. Returns entries with _tokens,
    _tool_calls, _parsed_timestamp, _model for append_cli_step compatibility.

    Compaction handling: if cumulative_total decreases (e.g. after context
    compaction), we increment compaction_offset and include it in the hash so
    post-compaction steps are not treated as duplicates.

    Best-effort: skips malformed lines, never raises.
    """
    new_entries: list[dict] = []
    updated_hashes = set(processed_hashes)

    for session_path in find_session_files(data_path):
        session_key = str(session_path.resolve())
        current_model = "gpt-5-codex"
        pending_tool_calls: list[dict[str, Any]] = []
        last_cumulative_total: int | None = None
        compaction_offset = 0
        try:
            with open(session_path, "r", encoding="utf-8") as fh:
                for line_idx, raw_line in enumerate(fh):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        entry = json.loads(raw_line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed JSONL line in %s", session_path)
                        continue

                    payload = entry.get("payload")
                    if entry.get("type") == "turn_context" and isinstance(payload, dict):
                        current_model = payload.get("model") or current_model

                    tool_calls = extract_tool_calls_from_entry(entry)
                    if tool_calls:
                        pending_tool_calls.extend(tool_calls)

                    tokens, cumulative_total = extract_usage_snapshot(entry)
                    if tokens is None:
                        continue

                    # Potential compaction check for robust deduplication: cumulative_total may decreases 
                    # HOWEVER: based on https://github.com/openai/codex/pull/3446#pullrequestreview-3214048488
                    # we have reason to believe that cumulative_total won't decrease, which means we are in a strictly
                    # monotonic case and this logic is redundant.
                    if (
                        last_cumulative_total is not None
                        and cumulative_total is not None
                        and cumulative_total < last_cumulative_total
                    ):
                        compaction_offset += 1
                        logger.debug(
                            "Codex compaction detected via cumulative_total decrease: cumulative_total %d -> %d, offset=%d",
                            last_cumulative_total,
                            cumulative_total,
                            compaction_offset,
                        )
                    if cumulative_total is not None:
                        last_cumulative_total = cumulative_total

                    parsed_ts = _parse_timestamp(
                        entry.get("timestamp") or entry.get("ts") or entry.get("created_at")
                    )
                    uid = _create_step_hash(
                        session_key, cumulative_total, parsed_ts, line_idx, compaction_offset
                    )
                    if uid in updated_hashes:
                        # Re-reading old lines should not leak already-consumed tool calls into
                        # the next new step, especially when the file contains duplicate snapshots.
                        pending_tool_calls = []
                        continue

                    new_entries.append({
                        "_tokens": tokens,
                        "_tool_calls": pending_tool_calls.copy(),
                        "_parsed_timestamp": parsed_ts,
                        "_model": current_model,
                    })
                    pending_tool_calls = []
                    updated_hashes.add(uid)

        except OSError as exc:
            logger.warning("Could not read %s: %s", session_path, exc)

    return new_entries, updated_hashes
