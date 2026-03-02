"""
Minimal reader for Claude Code JSONL session logs.

Reimplements only the token extraction and deduplication logic needed to record
per-API-call usage into cumulative_metrics.json.  Inspired by (but not dependent on)
Maciek-roboblog/Claude-Code-Usage-Monitor – we need only ~150 lines of the reader,
not the full TUI/Rich/pydantic stack.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def find_jsonl_files(data_path: Path) -> list[Path]:
    """Return all *.jsonl files under data_path, sorted by path for reproducibility."""
    if not data_path.is_dir():
        return []
    return sorted(data_path.rglob("*.jsonl"))


def _parse_timestamp(ts: Any) -> datetime | None:
    """Parse a timestamp value into a timezone-aware UTC datetime.

    Accepts ISO-8601 strings (with or without Z/offset) and UNIX floats.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if isinstance(ts, str):
        try:
            # Replace Z sentinel with explicit UTC offset before fromisoformat
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            logger.debug("Could not parse timestamp string: %r", ts)
    return None


def _create_unique_hash(entry: dict) -> str | None:
    """Derive a dedup key from message id + requestId (or uuid as fallback).

    Mirrors the logic used by Claude-Code-Usage-Monitor._create_unique_hash so
    that the same entries are not double-counted if we re-read files.
    """
    msg = entry.get("message")
    message_id = msg.get("id") if isinstance(msg, dict) else None
    request_id = entry.get("requestId")
    if message_id and request_id:
        return f"{message_id}:{request_id}"
    uid = entry.get("uuid")
    return uid if uid else None


def extract_tokens_from_entry(entry: dict) -> dict[str, int] | None:
    """Extract token counts from a Claude Code JSONL assistant entry.

    Returns a dict with keys (prompt, completion, cached, cache_write, total)
    or None when no usage data is present.  Handles both the primary location
    (entry.message.usage) and the top-level fallback (entry.usage).
    """
    usage: dict | None = None
    msg = entry.get("message")
    if isinstance(msg, dict):
        usage = msg.get("usage")
    if not isinstance(usage, dict):
        usage = entry.get("usage")
    if not isinstance(usage, dict):
        return None

    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    # Anthropic reports cache buckets separately from input_tokens
    cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    total = input_tokens + output_tokens + cache_write + cache_read

    return {
        "prompt": input_tokens,
        "completion": output_tokens,
        "cached": cache_read,
        "cache_write": cache_write,
        "total": total,
    }


def extract_tool_calls_from_entry(entry: dict) -> list[dict[str, Any]]:
    """Extract tool_use blocks from message.content as {name, args} dicts."""
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name")
            args = block.get("input", {})
            if name:
                tool_calls.append({"name": name, "args": args})
    return tool_calls


def load_new_usage_entries(
    data_path: Path,
    processed_hashes: set[str],
    since_timestamp: datetime | None = None,
) -> tuple[list[dict], set[str]]:
    """Load assistant JSONL entries that have not yet been processed.

    Only returns entries of type "assistant" that have usable token data and a
    dedup hash not already in processed_hashes.  Entries are returned in file
    order; callers should sort by _parsed_timestamp before creating steps.

    The returned processed_hashes is the *union* of the input set and all new
    hashes, suitable for passing back on the next poll.

    Each returned entry has three extra keys injected:
      _tokens          – dict from extract_tokens_from_entry
      _tool_calls      – list from extract_tool_calls_from_entry
      _parsed_timestamp – datetime | None
    """
    # best_by_hash: keep the entry with the highest total token count for each uid.
    # Claude Code emits two entries per API call: a streaming chunk (output_tokens=0)
    # and a final entry with complete usage.  Taking the last/best avoids under-counting.
    best_by_hash: dict[str, dict] = {}

    for jsonl_path in find_jsonl_files(data_path):
        try:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        entry = json.loads(raw_line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed JSONL line in %s", jsonl_path)
                        continue

                    if entry.get("type") != "assistant":
                        continue

                    tokens = extract_tokens_from_entry(entry)
                    if tokens is None:
                        continue

                    uid = _create_unique_hash(entry)
                    if uid is None:
                        logger.debug("No dedup key for entry in %s, skipping", jsonl_path)
                        continue

                    if uid in processed_hashes:
                        continue

                    parsed_ts = _parse_timestamp(entry.get("timestamp"))
                    if since_timestamp is not None and parsed_ts is not None:
                        if parsed_ts < since_timestamp:
                            continue

                    entry["_tokens"] = tokens
                    entry["_tool_calls"] = extract_tool_calls_from_entry(entry)
                    entry["_parsed_timestamp"] = parsed_ts

                    existing = best_by_hash.get(uid)
                    if existing is None or tokens["total"] > existing["_tokens"]["total"]:
                        best_by_hash[uid] = entry

        except OSError as exc:
            logger.warning("Could not read %s: %s", jsonl_path, exc)

    new_entries = list(best_by_hash.values())
    new_hashes = set(best_by_hash.keys())
    return new_entries, processed_hashes | new_hashes
