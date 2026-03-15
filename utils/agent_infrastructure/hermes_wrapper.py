#!/usr/bin/env python3
"""
Hermes wrapper that adapts AIAgent callbacks to JSONL stdout events.

The CLI harness expects one JSON object per stdout line so it can stream
events through CliAgentBackend.run_stream_reader(). Hermes does not provide
that natively, so this wrapper exposes a small, stable event contract:

- system: session/model metadata
- thinking: reasoning text for UI streaming
- tool_use: MCP/native tool invocation preview
- result: final session metrics
- error: fatal wrapper/runtime failures
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _read_text(path: str | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _build_initial_prompt(
    directive_text: str,
    server_url: str,
    is_resume: bool,
) -> str:
    """Build initial user message. Matches base _build_bootstrap_content runtime context."""
    runtime_context = (
        "Runtime context:\n"
        f"- Pokemon server URL: {server_url}\n"
        "- You are running in a long-lived interactive session.\n"
        "- Act autonomously and continuously, using MCP tools directly as needed.\n"
        "- Poll game state on your own via MCP tools; do not wait for additional operator prompts.\n"
        "- Continue until externally terminated by the orchestrator when completion condition is met.\n"
    )
    if is_resume:
        return (
            "Continue the current autonomous Pokemon Emerald session.\n\n"
            f"{runtime_context}"
        )
    if directive_text.strip():
        # directive_text from backend already includes directive + runtime context; use as-is
        return directive_text.rstrip()
    return (
        "Start the autonomous Pokemon Emerald session.\n\n"
        f"{runtime_context}"
    )


def _extract_tool_reasoning(arguments: dict[str, Any]) -> str:
    for key in ("reasoning", "reason", "thought", "notes"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="JSONL bridge for Hermes AIAgent.")
    parser.add_argument("--directive-path", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--hermes-home", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--resume-session-id", default="")
    args = parser.parse_args()

    hermes_home = Path(args.hermes_home).resolve()
    hermes_home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(hermes_home)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # Import Hermes only after HERMES_HOME is set so it picks up the run-local config/state.
    try:
        from hermes_state import SessionDB
        from run_agent import AIAgent
    except ImportError as exc:
        _emit(
            {
                "type": "error",
                "message": (
                    "Failed to import Hermes internals. Ensure the container/image has "
                    "hermes-agent installed and exposes run_agent.py + hermes_state.py."
                ),
                "error": str(exc),
            }
        )
        return 1

    stop_requested = False

    def _handle_signal(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        _emit(
            {
                "type": "error",
                "message": f"Hermes wrapper interrupted by signal {signum}",
                "signal": signum,
            }
        )
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    directive_text = _read_text(args.directive_path)
    resume_session_id = args.resume_session_id.strip() or None

    session_db = SessionDB(hermes_home / "state.db")
    conversation_history: list[dict[str, Any]] | None = None
    if resume_session_id:
        try:
            conversation_history = session_db.get_messages_as_conversation(resume_session_id)
        except Exception:
            conversation_history = None

    model = args.model.strip() or os.environ.get("HERMES_MODEL", "").strip() or "anthropic/claude-sonnet-4.5"
    provider = args.provider.strip() or os.environ.get("HERMES_PROVIDER", "").strip() or None
    base_url = args.base_url.strip() or os.environ.get("HERMES_BASE_URL", "").strip() or None
    api_key = None
    api_key_env = args.api_key_env.strip() or os.environ.get("HERMES_API_KEY_ENV", "").strip()
    if api_key_env:
        api_key = os.environ.get(api_key_env) or None

    tool_event_counter = 0
    start_time = time.time()

    def tool_progress_callback(tool_name: str, _preview: str, arguments: dict[str, Any]) -> None:
        nonlocal tool_event_counter
        tool_event_counter += 1
        normalized_args = arguments if isinstance(arguments, dict) else {}
        _emit(
            {
                "type": "tool_use",
                "tool_use_id": f"{resume_session_id or 'session'}-tool-{tool_event_counter}",
                "name": tool_name,
                "tool_name": tool_name,
                "input": normalized_args,
                "arguments": normalized_args,
                "parameters": normalized_args,
            }
        )

    def reasoning_callback(reasoning_text: str) -> None:
        if isinstance(reasoning_text, str) and reasoning_text.strip():
            _emit({"type": "thinking", "content": reasoning_text.strip()})

    try:
        os.chdir(args.working_dir)
        agent = AIAgent(
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            model=model,
            quiet_mode=True,
            session_id=resume_session_id,
            session_db=session_db,
            enabled_toolsets=["mcp-pokemon-emerald"],
            tool_progress_callback=tool_progress_callback,
            reasoning_callback=reasoning_callback,
            pass_session_id=True,
        )

        _emit(
            {
                "type": "system",
                "session_id": agent.session_id,
                "model": agent.model,
                "mcp_servers": ["pokemon-emerald"],
                "tools": [tool.get("function", {}).get("name", "") for tool in (agent.tools or [])],
            }
        )

        user_message = _build_initial_prompt(
            directive_text=directive_text,
            server_url=args.server_url,
            is_resume=bool(resume_session_id and conversation_history),
        )

        result = agent.run_conversation(
            user_message=user_message,
            conversation_history=conversation_history,
            persist_user_message=user_message,
        )
        duration_ms = int((time.time() - start_time) * 1000)

        _emit(
            {
                "type": "result",
                "session_id": agent.session_id,
                "model": agent.model,
                "content": result.get("final_response") if isinstance(result, dict) else None,
                "num_turns": int(getattr(agent, "session_api_calls", 0) or 0),
                "duration_ms": duration_ms,
                "duration_api_ms": 0,
                "total_cost_usd": 0.0,
                "is_error": bool(result.get("failed")) if isinstance(result, dict) else False,
                "error": result.get("error", "") if isinstance(result, dict) else "",
                "usage": {
                    "input_tokens": int(getattr(agent, "session_prompt_tokens", 0) or 0),
                    "output_tokens": int(getattr(agent, "session_completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(agent, "session_total_tokens", 0) or 0),
                },
            }
        )
        return 0 if not (isinstance(result, dict) and result.get("failed")) else 1
    except SystemExit:
        raise
    except Exception as exc:
        _emit(
            {
                "type": "error",
                "message": "Hermes wrapper failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "stopped": stop_requested,
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
