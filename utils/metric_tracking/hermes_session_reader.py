"""
Reader for Hermes session state stored in ~/.hermes/state.db.

Hermes persists durable session metadata in SQLite rather than JSONL. We poll
the state database in read-only mode, compute deltas from the last observed
session totals, and translate them into append_cli_step-compatible entries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_DB_NAME = "state.db"


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        check_same_thread=False,
        timeout=2.0,
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        # Read-only handles may reject journal changes on some platforms.
        pass
    return conn


def _with_retries(db_path: Path, fn, retries: int = 5, base_sleep: float = 0.05):
    last_exc: Exception | None = None
    for attempt in range(retries):
        conn: sqlite3.Connection | None = None
        try:
            conn = _connect_readonly(db_path)
            return fn(conn)
        except sqlite3.OperationalError as exc:
            last_exc = exc
            message = str(exc).lower()
            if "locked" not in message and "busy" not in message:
                raise
            time.sleep(base_sleep * (2 ** attempt))
        finally:
            if conn is not None:
                conn.close()
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("read-only Hermes DB query failed without exception")


def _parse_timestamp(ts: Any) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


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
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        if isinstance(name, str) and name:
            normalized.append({"name": name, "args": arguments})
    return normalized


def find_session_logs(data_path: Path) -> list[Path]:
    logs_dir = Path(data_path).resolve() / "sessions"
    if not logs_dir.is_dir():
        return []
    return sorted(logs_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime)


def get_latest_session_id(data_path: Path) -> str | None:
    db_path = Path(data_path).resolve() / STATE_DB_NAME
    if db_path.exists():
        try:
            row = _with_retries(
                db_path,
                lambda conn: conn.execute(
                    """
                    SELECT id
                    FROM sessions
                    ORDER BY COALESCE(ended_at, started_at) DESC, started_at DESC
                    LIMIT 1
                    """
                ).fetchone(),
            )
            if row and row["id"]:
                return str(row["id"])
        except Exception as exc:
            logger.warning("Could not read latest Hermes session id: %s", exc)

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
    Load new Hermes usage deltas from state.db.

    Because Hermes stores cumulative token counters on the `sessions` table, we
    derive a delta step each time those cumulative counts advance.
    """

    db_path = Path(data_path).resolve() / STATE_DB_NAME
    if not db_path.exists():
        return [], set(processed_hashes), dict(last_seen_totals or {})

    previous_state = dict(last_seen_totals or {})
    updated_hashes = set(processed_hashes)

    def _query(conn: sqlite3.Connection):
        session_rows = conn.execute(
            """
            SELECT id, model, started_at, ended_at, tool_call_count, input_tokens, output_tokens
            FROM sessions
            ORDER BY started_at ASC
            """
        ).fetchall()

        messages_by_session: dict[str, list[sqlite3.Row]] = {}
        for row in session_rows:
            sid = row["id"]
            message_rows = conn.execute(
                """
                SELECT id, timestamp, tool_calls
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (sid,),
            ).fetchall()
            messages_by_session[sid] = message_rows
        return session_rows, messages_by_session

    session_rows, messages_by_session = _with_retries(db_path, _query)

    new_entries: list[dict[str, Any]] = []
    next_state: dict[str, dict[str, Any]] = {}
    for row in session_rows:
        session_id = str(row["id"])
        messages = messages_by_session.get(session_id, [])
        previous = previous_state.get(
            session_id,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "last_message_id": 0,
                "tool_call_count": 0,
            },
        )

        current_input = int(row["input_tokens"] or 0)
        current_output = int(row["output_tokens"] or 0)
        last_message_id = max((int(msg["id"]) for msg in messages), default=0)

        delta_input = max(current_input - int(previous.get("input_tokens", 0) or 0), 0)
        delta_output = max(current_output - int(previous.get("output_tokens", 0) or 0), 0)
        new_messages = [
            msg
            for msg in messages
            if int(msg["id"]) > int(previous.get("last_message_id", 0) or 0)
        ]

        tool_calls: list[dict[str, Any]] = []
        latest_ts = _parse_timestamp(row["ended_at"]) or _parse_timestamp(row["started_at"])
        for msg in new_messages:
            parsed = _normalize_tool_calls(msg["tool_calls"])
            if parsed:
                tool_calls.extend(parsed)
            msg_ts = _parse_timestamp(msg["timestamp"])
            if msg_ts is not None:
                latest_ts = msg_ts

        snapshot_hash = (
            f"session:{session_id}:input:{current_input}:output:{current_output}:"
            f"messages:{last_message_id}:tools:{int(row['tool_call_count'] or 0)}"
        )
        if snapshot_hash not in updated_hashes and (delta_input > 0 or delta_output > 0 or tool_calls):
            entry = {
                "_tokens": {
                    "prompt": delta_input,
                    "completion": delta_output,
                    "cached": 0,
                    "cache_write": 0,
                    "total": delta_input + delta_output,
                    "cost": 0.0,
                },
                "_tool_calls": tool_calls,
                "_parsed_timestamp": latest_ts,
                "_model": row["model"] or "hermes-agent",
                "_session_id": session_id,
            }
            new_entries.append(entry)
            updated_hashes.add(snapshot_hash)

        next_state[session_id] = {
            "input_tokens": current_input,
            "output_tokens": current_output,
            "last_message_id": last_message_id,
            "tool_call_count": int(row["tool_call_count"] or 0),
        }

    return new_entries, updated_hashes, next_state
