"""
Reader for Gemini CLI telemetry logs (OpenTelemetry file-based output).

Parses `gemini_cli.api_response` events from the telemetry outfile to extract
per-API-request usage (tokens, cost, model, duration).  Each api_response event
maps to one "step" in our cumulative_metrics, matching the granularity we get
from Claude Code JSONL parsing.

Telemetry is enabled by writing a settings.json into the agent's ~/.gemini/ with:
  { "telemetry": { "enabled": true, "target": "local", "outfile": "<path>" } }

The outfile is a JSONL file with OpenTelemetry LogRecord-style entries.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The telemetry event name emitted for each completed Gemini API response.
API_RESPONSE_EVENT = "gemini_cli.api_response"

TELEMETRY_OUTFILE_DEFAULT = "telemetry.jsonl"


def find_telemetry_files(data_path: Path) -> list[Path]:
    """Return all telemetry log/jsonl files under data_path, sorted by path."""
    if not data_path.is_dir():
        return []
    files: list[Path] = []
    for ext in ("*.jsonl", "*.log", "*.json"):
        files.extend(data_path.rglob(ext))
    return sorted(set(files))


def _parse_otel_timestamp(ts: Any) -> datetime | None:
    """Parse an OTEL timestamp (ISO-8601 string or UNIX epoch) to UTC datetime."""
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


def _extract_api_response(record: dict) -> dict[str, Any] | None:
    """Extract token usage from a gemini_cli.api_response telemetry record.

    Returns a dict compatible with llm_logger.append_cli_step(token_usage=...):
      { prompt, completion, cached, cache_write, total, cost,
        _model, _duration_ms, _timestamp, _prompt_id }
    or None if the record is not an api_response event.
    """
    if not isinstance(record, dict):
        return None

    attrs = record.get("attributes") or {}
    body = (
        record.get("body")
        or record.get("name")
        or record.get("event")
        or attrs.get("event.name")
        or ""
    )
    if body != API_RESPONSE_EVENT:
        return None

    if not attrs:
        resource = record.get("resource")
        if isinstance(resource, dict):
            attrs = resource.get("attributes") or {}
    if not attrs:
        return None

    input_tokens = int(attrs.get("input_token_count", 0) or 0)
    output_tokens = int(attrs.get("output_token_count", 0) or 0)
    cached_tokens = int(attrs.get("cached_content_token_count", 0) or 0)
    thoughts_tokens = int(attrs.get("thoughts_token_count", 0) or 0)
    tool_tokens = int(attrs.get("tool_token_count", 0) or 0)
    total_tokens = int(attrs.get("total_token_count", 0) or 0)

    # If total_token_count wasn't provided, compute it
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens + thoughts_tokens + tool_tokens + cached_tokens

    # Map to our standard token_usage dict.
    # Gemini doesn't have a separate "cache_write" concept in its telemetry;
    # "cached_content_token_count" is read-cache (analogous to Claude's cache_read).
    # thoughts_tokens and tool_tokens go into completion for cost calculation purposes.
    return {
        "prompt": input_tokens,
        "completion": output_tokens + thoughts_tokens + tool_tokens,
        "cached": cached_tokens,
        "cache_write": 0,
        "total": total_tokens,
        "cost": 0.0,  # Gemini telemetry doesn't include explicit cost; llm_logger pricing handles it
        "_model": str(attrs.get("model", "")),
        "_duration_ms": int(attrs.get("duration_ms", 0) or 0),
        "_timestamp": attrs.get("event.timestamp") or record.get("timestamp") or record.get("timeUnixNano"),
        "_prompt_id": str(attrs.get("prompt_id", "")),
        "_status_code": attrs.get("status_code"),
    }


def _make_dedup_hash(entry: dict) -> str:
    """Create a stable dedup hash for a telemetry entry.

    Uses prompt_id + timestamp + model + token counts to prevent duplicates
    across polls while being robust to re-reads after restarts.
    """
    key_parts = [
        entry.get("_prompt_id", ""),
        str(entry.get("_timestamp", "")),
        entry.get("_model", ""),
        str(entry.get("prompt", 0)),
        str(entry.get("completion", 0)),
    ]
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_new_gemini_usage(
    data_path: Path,
    processed_hashes: set[str],
    processed_offsets: dict[str, int] | None = None,
) -> tuple[list[dict], set[str], dict[str, int]]:
    """Load new api_response entries from Gemini telemetry files.

    Uses file byte offsets for efficient incremental reads (only reads new data
    appended since the last poll).

    Args:
        data_path: Directory containing telemetry logs (typically agent_memory_dir).
        processed_hashes: Set of dedup hashes already seen.
        processed_offsets: Dict mapping str(file_path) -> last read byte offset.

    Returns:
        (new_entries, updated_hashes, updated_offsets)
        Each entry in new_entries has the standard token_usage keys plus
        _model, _duration_ms, _timestamp, _prompt_id, _hash.
    """
    if processed_offsets is None:
        processed_offsets = {}

    new_entries: list[dict] = []
    updated_hashes = set(processed_hashes)
    updated_offsets = dict(processed_offsets)

    for log_file in find_telemetry_files(data_path):
        str_path = str(log_file)
        offset = updated_offsets.get(str_path, 0)

        try:
            current_size = log_file.stat().st_size
            if current_size < offset:
                offset = 0  # file truncated/rotated
            if current_size == offset:
                continue

            with open(log_file, "r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue

                    entry = _extract_api_response(record)
                    if entry is None:
                        continue
                    if entry["total"] == 0:
                        continue

                    h = _make_dedup_hash(entry)
                    if h in updated_hashes:
                        continue

                    entry["_hash"] = h
                    # Parse timestamp for sorting
                    entry["_parsed_timestamp"] = _parse_otel_timestamp(entry["_timestamp"])
                    new_entries.append(entry)
                    updated_hashes.add(h)

                updated_offsets[str_path] = f.tell()

        except Exception as e:
            logger.warning("Error reading Gemini telemetry %s: %s", log_file, e)

    return new_entries, updated_hashes, updated_offsets
