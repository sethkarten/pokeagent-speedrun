"""
Reader for Hermes session state stored in ~/.hermes/sessions/*.json.

Hermes persists a continuously updated session JSON log for each run. We treat
those logs as the source of truth for assistant turns and tool calls, then join
them with wrapper-written usage events to produce append_cli_step-compatible
entries for cumulative metrics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_DB_NAME = "state.db"
USAGE_EVENTS_NAME = "usage_events.jsonl"

def _parse_timestamp(ts: Any) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _normalize_tool_name(name: Any) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    for prefix in ("mcp_pokemon_emerald_", "mcp__pokemon-emerald__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name.split("__")[-1] if "__" in name else name


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        arguments = item.get("arguments")
        function = item.get("function")
        if isinstance(function, dict):
            name = function.get("name", name)
            arguments = function.get("arguments", arguments)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        normalized_name = _normalize_tool_name(name)
        if normalized_name:
            normalized.append({"name": normalized_name, "args": arguments})
    return normalized


def find_session_logs(data_path: Path) -> list[Path]:
    logs_dir = Path(data_path).resolve() / "sessions"
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _extract_usage_tokens(raw: dict[str, Any]) -> dict[str, Any]:
    prompt = int(raw.get("prompt_tokens", 0) or raw.get("input_tokens", 0) or 0)
    completion = int(raw.get("completion_tokens", 0) or raw.get("output_tokens", 0) or 0)
    total = int(raw.get("total_tokens", 0) or (prompt + completion))
    prompt_details = _coerce_mapping(raw.get("prompt_tokens_details"))
    cached = int(
        raw.get("cached_tokens", 0)
        or prompt_details.get("cached_tokens", 0)
        or raw.get("cache_read_input_tokens", 0)
        or 0
    )
    cache_write = int(
        raw.get("cache_write_tokens", 0)
        or prompt_details.get("cache_write_tokens", 0)
        or raw.get("cache_creation_input_tokens", 0)
        or 0
    )
    cost = float(raw.get("cost_usd", 0.0) or raw.get("cost", 0.0) or raw.get("total_cost_usd", 0.0) or 0.0)
    return {
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "cache_write": cache_write,
        "total": total,
        "cost": cost,
    }


def _load_usage_events(data_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    usage_path = Path(data_path).resolve() / USAGE_EVENTS_NAME
    if not usage_path.exists():
        return {}

    events_by_session: dict[str, dict[int, dict[str, Any]]] = {}
    try:
        lines = usage_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read Hermes usage events %s: %s", usage_path, exc)
        return {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue

        session_id = raw.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        api_call_index = int(raw.get("api_call_index", 0) or 0)
        if api_call_index <= 0:
            continue

        event = {
            "timestamp": _parse_timestamp(raw.get("timestamp")),
            "tokens": _extract_usage_tokens(raw),
        }
        events_by_session.setdefault(session_id, {})[api_call_index] = event
    return events_by_session


def get_latest_session_id(data_path: Path) -> str | None:
    logs = find_session_logs(data_path)
    if not logs:
        return None
    latest = max(logs, key=lambda p: p.stat().st_mtime)
    stem = latest.stem
    return stem[len("session_") :] if stem.startswith("session_") else stem


def load_new_usage_entries(
    data_path: Path,
    processed_hashes: set[str],
    last_seen_totals: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], set[str], dict[str, dict[str, Any]]]:
    """
    Load new Hermes usage entries from session JSON plus wrapper usage events.
    """
    root = Path(data_path).resolve()
    previous_state = dict(last_seen_totals or {})
    updated_hashes = set(processed_hashes)
    usage_events = _load_usage_events(root)
    next_state: dict[str, dict[str, Any]] = dict(previous_state)
    new_entries: list[dict[str, Any]] = []

    for log_path in find_session_logs(root):
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read Hermes session log %s: %s", log_path, exc)
            continue

        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            stem = log_path.stem
            session_id = stem[len("session_") :] if stem.startswith("session_") else stem

        messages = payload.get("messages")
        if not isinstance(messages, list):
            continue

        session_state = dict(next_state.get(session_id, {}))
        last_assistant_index = int(session_state.get("assistant_index", 0) or 0)
        session_start = _parse_timestamp(payload.get("session_start")) or datetime.now(timezone.utc)
        model = payload.get("model") or "hermes-agent"
        session_usage = usage_events.get(session_id, {})

        assistant_index = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue

            assistant_index += 1
            if assistant_index <= last_assistant_index:
                continue

            tool_calls = _normalize_tool_calls(message.get("tool_calls"))
            usage_event = session_usage.get(assistant_index, {})
            tokens = usage_event.get(
                "tokens",
                {
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                    "cache_write": 0,
                    "total": 0,
                    "cost": 0.0,
                },
            )
            parsed_ts = usage_event.get("timestamp") or (session_start + timedelta(milliseconds=assistant_index))

            snapshot_hash = f"json:{session_id}:assistant:{assistant_index}"
            if snapshot_hash in updated_hashes:
                continue

            new_entries.append(
                {
                    "_tokens": tokens,
                    "_tool_calls": tool_calls,
                    "_parsed_timestamp": parsed_ts,
                    "_model": model,
                    "_session_id": session_id,
                }
            )
            updated_hashes.add(snapshot_hash)

        next_state[session_id] = {
            "assistant_index": assistant_index,
        }

    new_entries.sort(key=lambda e: (e.get("_parsed_timestamp") or datetime.min.replace(tzinfo=timezone.utc)))
    return new_entries, updated_hashes, next_state
