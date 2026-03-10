"""
Reader for Gemini CLI session JSON files (analogous to claude_jsonl_reader).

Reads session-*.json from tmp/workspace/chats/ under the agent memory directory.
Each gemini message has tokens and toolCalls; we extract these for cumulative_metrics.
Replaces telemetry.jsonl-based tracking to save space (telemetry grows much faster).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHATS_REL = "tmp/workspace/chats"


def find_session_files(data_path: Path) -> list[Path]:
    """Return all session-*.json files under data_path/tmp/workspace/chats, sorted by mtime."""
    chats_dir = Path(data_path).resolve() / CHATS_REL
    if not chats_dir.is_dir():
        return []
    files = list(chats_dir.glob("session-*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime)


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


def extract_tokens_from_message(msg: dict) -> dict[str, Any] | None:
    """Extract token counts from a Gemini session message (type=gemini).

    Returns dict with prompt, completion, cached, cache_write, total, cost.
    Maps: input->prompt, output+thoughts+tool->completion, cached->cached.

    Token accounting (Gemini vs Claude):
    - Claude: prompt, cached, cache_write are ADDITIVE; total = prompt + completion + cached + cache_write.
    - Gemini: cached is a SUBSET of prompt (input); total = prompt + completion. We store cached
      separately for analytics/billing but do NOT add it to total to avoid double-counting.
    - cache_write: Gemini uses IMPLICIT CACHING (automatic, no explicit cache creation API).
      The session format does not expose cache_write; we return None. Cost of cache creation
      is already included in the first request's input tokens. For plotting/analytics,
      cache_write_tokens will be null in step entries—this is expected for Gemini runs.
    """
    tokens = msg.get("tokens")
    if not isinstance(tokens, dict):
        return None
    inp = int(tokens.get("input", 0) or 0)
    out = int(tokens.get("output", 0) or 0)
    cached = int(tokens.get("cached", 0) or 0)
    thoughts = int(tokens.get("thoughts", 0) or 0)
    tool = int(tokens.get("tool", 0) or 0)
    # Session format may add cache_write/cacheWrite in future; until then, None for Gemini
    cache_write_raw = tokens.get("cache_write") or tokens.get("cacheWrite")
    cache_write = int(cache_write_raw) if cache_write_raw is not None else None
    total = int(tokens.get("total", 0) or 0)
    if total == 0:
        # Cached is subset of input; total = prompt + completion (do not add cached again)
        total = inp + out + thoughts + tool
    return {
        "prompt": inp,
        "completion": out + thoughts + tool,
        "cached": cached,
        "cache_write": cache_write,
        "total": total,
        "cost": 0.0,
    }


def extract_tool_calls_from_message(msg: dict) -> list[dict[str, Any]]:
    """Extract toolCalls from a Gemini message as {name, args} list."""
    tool_calls = msg.get("toolCalls")
    if not isinstance(tool_calls, list):
        return []
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict) and tc.get("name"):
            result.append({"name": tc["name"], "args": tc.get("args", {})})
    return result


def load_new_usage_entries(
    data_path: Path,
    processed_hashes: set[str],
) -> tuple[list[dict], set[str]]:
    """Load gemini messages from session JSON files that have not been processed.

    Returns (new_entries, updated_processed_hashes). Each entry has _tokens, _tool_calls,
    _parsed_timestamp, and uses message id as dedup key.

    Multi-session / subagent robustness:
    - Processes ALL session-*.json files under tmp/workspace/chats/ (sorted by mtime).
    - Deduplication by message id ensures no double-counting across files.
    - Multiple session files (e.g. main agent + subagent /ask) are all included;
      message ids are globally unique, so interleaving is safe.
    - To exclude subagent sessions, callers would need to filter by sessionId or
      pass a primary session file path; not implemented by default.
    """
    new_entries: list[dict] = []
    updated_hashes = set(processed_hashes)

    for session_path in find_session_files(data_path):
        try:
            raw = session_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s: %s", session_path, exc)
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug("Skipping unparseable session JSON %s: %s", session_path, exc)
            continue

        messages = data.get("messages")
        if not isinstance(messages, list):
            continue

        for msg in messages:
            if msg.get("type") != "gemini":
                continue
            msg_id = msg.get("id")
            if not msg_id:
                continue
            if msg_id in updated_hashes:
                continue

            tokens = extract_tokens_from_message(msg)
            if tokens is None or tokens["total"] == 0:
                continue

            tool_calls = extract_tool_calls_from_message(msg)
            parsed_ts = _parse_timestamp(msg.get("timestamp"))

            entry = {
                "_tokens": tokens,
                "_tool_calls": tool_calls,
                "_parsed_timestamp": parsed_ts,
                "_model": msg.get("model", "gemini"),
            }
            new_entries.append(entry)
            updated_hashes.add(msg_id)

    return new_entries, updated_hashes
