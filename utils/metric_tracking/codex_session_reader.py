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


def _create_unique_hash(entry: dict, line_idx: int, file_path: str) -> str:
    """Stable dedup key for a session log entry."""
    # Prefer explicit ids when present
    session_id = entry.get("session_id") or entry.get("thread_id", "")
    turn_id = entry.get("turn_id") or entry.get("turn_index")
    ts = entry.get("timestamp") or entry.get("ts")
    payload = entry.get("payload") or {}
    if isinstance(payload, dict) and payload.get("type") == "token_count":
        # Use cumulative fields to avoid duplicate counts
        inp = payload.get("input_tokens", 0)
        out = payload.get("output_tokens", 0)
        key = f"{session_id}:{turn_id}:{ts}:{inp}:{out}"
    else:
        key = f"{session_id}:{turn_id}:{ts}:{line_idx}:{file_path}"
    return key or f"line_{line_idx}_{file_path}"


def extract_tokens_from_entry(entry: dict) -> dict[str, Any] | None:
    """Extract token counts from a Codex session entry.

    Supports:
    - event_msg with payload.type token_count (input_tokens, cached_input_tokens, output_tokens, reasoning_tokens)
    - usage object (input_tokens, output_tokens, cached_input_tokens) from turn.completed
    """
    # event_msg with token_count payload
    payload = entry.get("payload")
    if isinstance(payload, dict) and payload.get("type") == "token_count":
        # Codex nests counts under payload.info.last_token_usage or payload.info.total_token_usage
        info = payload.get("info") or {}
        usage = info.get("last_token_usage") or info.get("total_token_usage") or payload
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        reasoning = int(usage.get("reasoning_output_tokens", 0) or 0)
        total = inp + out + cached + reasoning
        if total == 0:
            return None
        return {
            "prompt": inp,
            "completion": out + reasoning,
            "cached": cached,
            "cache_write": 0,
            "total": total,
            "cost": 0.0,
        }

    # usage object (e.g. from turn.completed written to session)
    usage = entry.get("usage")
    if isinstance(usage, dict):
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        total = inp + out + cached
        if total == 0:
            return None
        return {
            "prompt": inp,
            "completion": out,
            "cached": cached,
            "cache_write": 0,
            "total": total,
            "cost": 0.0,
        }

    return None


def extract_tool_calls_from_entry(entry: dict) -> list[dict[str, Any]]:
    """Extract tool/MCP calls from entry if present. Best-effort."""
    tool_calls: list[dict[str, Any]] = []
    payload = entry.get("payload")
    if isinstance(payload, dict) and payload.get("type") == "mcp_tool_call":
        tool = payload.get("tool")
        args = payload.get("arguments") or {}
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

    Best-effort: skips malformed lines, never raises.
    """
    new_entries: list[dict] = []
    updated_hashes = set(processed_hashes)

    for session_path in find_session_files(data_path):
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

                    tokens = extract_tokens_from_entry(entry)
                    if tokens is None:
                        continue

                    uid = _create_unique_hash(entry, line_idx, str(session_path))
                    if uid in updated_hashes:
                        continue

                    parsed_ts = _parse_timestamp(
                        entry.get("timestamp") or entry.get("ts") or entry.get("created_at")
                    )
                    tool_calls = extract_tool_calls_from_entry(entry)
                    model = (
                        (entry.get("turn_context") or {}).get("model")
                        if isinstance(entry.get("turn_context"), dict)
                        else entry.get("model", "gpt-5-codex")
                    )

                    new_entries.append({
                        "_tokens": tokens,
                        "_tool_calls": tool_calls,
                        "_parsed_timestamp": parsed_ts,
                        "_model": model,
                    })
                    updated_hashes.add(uid)

        except OSError as exc:
            logger.warning("Could not read %s: %s", session_path, exc)

    return new_entries, updated_hashes
