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
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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


def _normalize_tool_name(name: str) -> str:
    for prefix in ("mcp_pokemon_emerald_", "mcp__pokemon-emerald__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name.split("__")[-1] if "__" in name else name


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


# #region agent log
def _debug_log(
    log_path: Path,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    run_id: str = "",
) -> None:
    import time as _t
    try:
        payload = {
            "sessionId": "b377a9",
            "id": f"log_{int(_t.time()*1000)}",
            "timestamp": int(_t.time() * 1000),
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "__dict__"):
        return {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _extract_usage_snapshot(response: Any) -> dict[str, Any]:
    response_map = _coerce_mapping(response)
    usage_map = _coerce_mapping(getattr(response, "usage", None))
    if not usage_map and isinstance(response_map.get("usage"), dict):
        usage_map = response_map["usage"]

    prompt_details = _coerce_mapping(usage_map.get("prompt_tokens_details"))
    prompt_tokens = int(usage_map.get("prompt_tokens", 0) or usage_map.get("input_tokens", 0) or 0)
    completion_tokens = int(usage_map.get("completion_tokens", 0) or usage_map.get("output_tokens", 0) or 0)
    total_tokens = int(usage_map.get("total_tokens", 0) or (prompt_tokens + completion_tokens))
    cached_tokens = int(prompt_details.get("cached_tokens", 0) or usage_map.get("cache_read_input_tokens", 0) or 0)
    cache_write_tokens = int(prompt_details.get("cache_write_tokens", 0) or usage_map.get("cache_creation_input_tokens", 0) or 0)
    total_cost_usd = float(
        response_map.get("total_cost_usd", 0.0)
        or response_map.get("cost", 0.0)
        or usage_map.get("cost_usd", 0.0)
        or usage_map.get("cost", 0.0)
        or 0.0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "total_cost_usd": total_cost_usd,
        "_usage_map": usage_map,
    }


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
    usage_events_path = hermes_home / "usage_events.jsonl"
    debug_log_path = Path("/data3/tu8435/thesis-remote/pokeagent-speedrun/.cursor/debug-b377a9.log")
    _disable_multimodal = os.environ.get("HERMES_DISABLE_MULTIMODAL", "").lower() in ("1", "true", "yes")
    conversation_history: list[dict[str, Any]] | None = None
    if resume_session_id:
        try:
            conversation_history = session_db.get_messages_as_conversation(resume_session_id)
        except Exception:
            conversation_history = None

    model = args.model.strip() or os.environ.get("HERMES_MODEL", "").strip() or "google/gemini-3-flash-preview"
    provider = args.provider.strip() or os.environ.get("HERMES_PROVIDER", "").strip() or None
    base_url = args.base_url.strip() or os.environ.get("HERMES_BASE_URL", "").strip() or None
    api_key = None
    api_key_env = args.api_key_env.strip() or os.environ.get("HERMES_API_KEY_ENV", "").strip()
    if api_key_env:
        api_key = os.environ.get(api_key_env) or None

    tool_event_counter = 0
    start_time = time.time()
    usage_state = {
        "last_snapshot": None,
        "api_call_index": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "total_cost_usd": 0.0,
    }
    multimodal_store: dict[str, dict[str, Any]] = {}
    multimodal_counter = 0

    def _capture_usage_snapshot(snapshot: dict[str, Any], session_id: str, model_name: str) -> None:
        usage_state["api_call_index"] += 1
        usage_state["prompt_tokens"] += int(snapshot.get("prompt_tokens", 0) or 0)
        usage_state["completion_tokens"] += int(snapshot.get("completion_tokens", 0) or 0)
        usage_state["total_tokens"] += int(snapshot.get("total_tokens", 0) or 0)
        usage_state["cached_tokens"] += int(snapshot.get("cached_tokens", 0) or 0)
        usage_state["cache_write_tokens"] += int(snapshot.get("cache_write_tokens", 0) or 0)
        usage_state["total_cost_usd"] += float(snapshot.get("total_cost_usd", 0.0) or 0.0)
        _append_jsonl(
            usage_events_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "api_call_index": usage_state["api_call_index"],
                "model": model_name,
                "prompt_tokens": int(snapshot.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(snapshot.get("completion_tokens", 0) or 0),
                "total_tokens": int(snapshot.get("total_tokens", 0) or 0),
                "cached_tokens": int(snapshot.get("cached_tokens", 0) or 0),
                "cache_write_tokens": int(snapshot.get("cache_write_tokens", 0) or 0),
                "total_cost_usd": float(snapshot.get("total_cost_usd", 0.0) or 0.0),
            },
        )

    def _patch_mcp_image_bridge() -> None:
        import tools.mcp_tool as mcp_tool

        # #region agent log
        _debug_log(
            debug_log_path,
            "H1",
            "hermes_wrapper.py:_patch_mcp_image_bridge",
            "mcp_tool handler names",
            {"has_make_tool_handler": hasattr(mcp_tool, "_make_tool_handler"), "has_make_call_tool_handler": hasattr(mcp_tool, "_make_call_tool_handler")},
        )
        # #endregion

    def _build_multimodal_registry_handler(server_name: str, tool_name: str, tool_timeout: float):
        import tools.mcp_tool as mcp_tool

        def _handler(args: dict, **kwargs) -> str:
            nonlocal multimodal_counter

            # #region agent log
            _debug_log(
                debug_log_path,
                "H1",
                "hermes_wrapper.py:_build_multimodal_registry_handler",
                "patched_handler_invoked",
                {"tool_name": tool_name},
            )
            _mcp_start = time.time()
            # #endregion

            with mcp_tool._lock:
                server = mcp_tool._servers.get(server_name)
            if not server or not server.session:
                return json.dumps({"error": f"MCP server '{server_name}' is not connected"})

            async def _call() -> str:
                nonlocal multimodal_counter
                result = await server.session.call_tool(tool_name, arguments=args)
                if result.isError:
                    error_text = ""
                    for block in (result.content or []):
                        if hasattr(block, "text"):
                            error_text += block.text
                    return json.dumps(
                        {
                            "error": mcp_tool._sanitize_error(
                                error_text or "MCP tool returned an error"
                            )
                        }
                    )

                text_parts: list[str] = []
                multimodal_parts: list[dict[str, Any]] = []
                image_count = 0
                block_types: list[str] = []
                for block in (result.content or []):
                    if hasattr(block, "text") and block.text:
                        text_parts.append(block.text)
                        multimodal_parts.append({"type": "text", "text": block.text})
                        block_types.append("text")
                    elif hasattr(block, "data") and hasattr(block, "mimeType"):
                        image_count += 1
                        multimodal_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                            }
                        )
                        block_types.append("image_data")
                    elif hasattr(block, "blob") and hasattr(block, "mimeType"):
                        image_count += 1
                        blob_data = getattr(block, "blob")
                        if isinstance(blob_data, bytes):
                            import base64

                            encoded = base64.b64encode(blob_data).decode("ascii")
                        else:
                            encoded = str(blob_data)
                        multimodal_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{block.mimeType};base64,{encoded}"},
                            }
                        )
                        block_types.append("image_blob")
                    else:
                        block_types.append(
                            f"{type(block).__name__}(has_data={hasattr(block,'data')},has_blob={hasattr(block,'blob')},has_mimeType={hasattr(block,'mimeType')})"
                        )

                # #region agent log
                _debug_log(
                    debug_log_path,
                    "H2",
                    "hermes_wrapper.py:_build_multimodal_registry_handler",
                    "tool_result_blocks",
                    {"tool_name": tool_name, "block_types": block_types, "image_count": image_count, "block_count": len(result.content or [])},
                )
                # #endregion

                payload: dict[str, Any] = {"result": "\n".join(text_parts) if text_parts else ""}
                if image_count:
                    multimodal_counter += 1
                    ref = f"{server_name}:{tool_name}:{multimodal_counter}"
                    if tool_name == "get_game_state":
                        to_evict = [k for k in multimodal_store if k.startswith(f"{server_name}:get_game_state:")]
                        for k in to_evict:
                            del multimodal_store[k]
                    multimodal_store[ref] = {
                        "parts": multimodal_parts,
                        "tool_name": tool_name,
                        "image_count": image_count,
                    }
                    payload["_pokeagent_multimodal_ref"] = ref
                    payload["_pokeagent_tool_name"] = tool_name
                    payload["_pokeagent_image_count"] = image_count
                return json.dumps(payload, ensure_ascii=False)

            try:
                # #region agent log
                _debug_log(
                    debug_log_path,
                    "H4",
                    "hermes_wrapper.py:_build_multimodal_registry_handler",
                    "mcp_call_start",
                    {"tool_name": tool_name},
                )
                # #endregion
                out = mcp_tool._run_on_mcp_loop(_call(), timeout=tool_timeout)
                # #region agent log
                _debug_log(
                    debug_log_path,
                    "H4",
                    "hermes_wrapper.py:_build_multimodal_registry_handler",
                    "mcp_call_done",
                    {"tool_name": tool_name, "duration_s": round(time.time() - _mcp_start, 2)},
                )
                # #endregion
                return out
            except Exception as exc:
                return json.dumps(
                    {
                        "error": mcp_tool._sanitize_error(
                            f"MCP call failed: {type(exc).__name__}: {exc}"
                        )
                    }
                )

        return _handler

    def _patch_registered_mcp_handlers(agent: Any) -> None:
        if _disable_multimodal:
            return
        from tools.registry import registry

        patched_tools: list[str] = []
        for tool in agent.tools or []:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            registered_name = function.get("name")
            if not isinstance(registered_name, str):
                continue
            prefix = "mcp_pokemon_emerald_"
            if not registered_name.startswith(prefix):
                continue
            entry = registry._tools.get(registered_name)
            if entry is None:
                continue
            raw_tool_name = registered_name[len(prefix) :]
            entry.handler = _build_multimodal_registry_handler(
                "pokemon-emerald",
                raw_tool_name,
                120.0,
            )
            patched_tools.append(registered_name)

        # #region agent log
        _debug_log(
            debug_log_path,
            "H1",
            "hermes_wrapper.py:_patch_registered_mcp_handlers",
            "patched_registry_handlers",
            {"count": len(patched_tools), "tools": patched_tools},
        )
        # #endregion

    def _patch_agent_runtime() -> None:
        original_create_client = AIAgent._create_openai_client
        original_build_assistant_message = AIAgent._build_assistant_message
        original_build_api_kwargs = AIAgent._build_api_kwargs

        def _wrap_openai_client(client: Any) -> Any:
            if getattr(client, "_pokeagent_usage_wrapped", False):
                return client

            _VISION_TIMEOUT = float(os.environ.get("HERMES_VISION_TIMEOUT", "10"))

            def _wrap_create(call):
                def _count_images(msgs):
                    payload_bytes = 0
                    image_count = 0
                    if isinstance(msgs, (list, tuple)):
                        for m in msgs:
                            if not isinstance(m, dict):
                                continue
                            c = m.get("content")
                            if isinstance(c, str):
                                payload_bytes += len(c)
                            elif isinstance(c, list):
                                for p in c:
                                    if isinstance(p, dict) and p.get("type") == "image_url":
                                        image_count += 1
                                        url = (p.get("image_url") or {}).get("url") or ""
                                        payload_bytes += len(url)
                                    elif isinstance(p, dict) and p.get("type") == "text":
                                        payload_bytes += len(str(p.get("text", "")))
                    return image_count, payload_bytes

                def _strip_vision_messages(msgs):
                    """Remove synthetic user messages that carry image_url parts."""
                    if not isinstance(msgs, (list, tuple)):
                        return msgs
                    stripped = []
                    for m in msgs:
                        if not isinstance(m, dict):
                            stripped.append(m)
                            continue
                        c = m.get("content")
                        if (
                            m.get("role") == "user"
                            and isinstance(c, list)
                            and any(isinstance(p, dict) and p.get("type") == "image_url" for p in c)
                        ):
                            continue
                        stripped.append(m)
                    return stripped

                def _wrapped(*call_args, **call_kwargs):
                    _api_start = time.time()
                    _api_idx = usage_state.get("api_call_index", 0) + 1
                    usage_state["api_call_index"] = _api_idx
                    msgs = call_kwargs.get("messages") or (call_args[0] if call_args else None)
                    msg_count = len(msgs) if isinstance(msgs, (list, tuple)) else 0
                    image_count, payload_bytes = _count_images(msgs)
                    _debug_log(
                        debug_log_path,
                        "H7",
                        "hermes_wrapper.py:_wrap_create",
                        "api_call_start",
                        {"api_call_index": _api_idx, "msg_count": msg_count, "payload_bytes": payload_bytes, "image_count": image_count},
                    )

                    original_timeout = call_kwargs.get("timeout")
                    if image_count > 0 and _VISION_TIMEOUT > 0:
                        call_kwargs["timeout"] = _VISION_TIMEOUT

                    try:
                        response = call(*call_args, **call_kwargs)
                    except Exception as vision_exc:
                        exc_name = type(vision_exc).__name__.lower()
                        is_timeout = "timeout" in exc_name or "timeout" in str(vision_exc).lower()
                        if image_count > 0 and is_timeout:
                            _debug_log(
                                debug_log_path,
                                "H9",
                                "hermes_wrapper.py:_wrap_create",
                                "vision_timeout_fallback",
                                {"api_call_index": _api_idx, "timeout_s": _VISION_TIMEOUT, "image_count": image_count},
                            )
                            stripped = _strip_vision_messages(msgs)
                            if "messages" in call_kwargs:
                                call_kwargs["messages"] = stripped
                            elif call_args:
                                call_args = (stripped, *call_args[1:])
                            if original_timeout is not None:
                                call_kwargs["timeout"] = original_timeout
                            else:
                                call_kwargs.pop("timeout", None)
                            response = call(*call_args, **call_kwargs)
                        else:
                            raise

                    _debug_log(
                        debug_log_path,
                        "H5",
                        "hermes_wrapper.py:_wrap_create",
                        "api_call_done",
                        {"api_call_index": _api_idx, "duration_s": round(time.time() - _api_start, 2)},
                    )
                    snapshot = _extract_usage_snapshot(response)
                    usage_map = snapshot.get("_usage_map") or {}
                    if usage_map:
                        try:
                            response.usage = _to_namespace(usage_map)
                        except Exception:
                            pass
                    usage_state["last_snapshot"] = {
                        key: value
                        for key, value in snapshot.items()
                        if not key.startswith("_")
                    }
                    return response

                return _wrapped

            try:
                client.chat.completions.create = _wrap_create(client.chat.completions.create)
            except Exception:
                pass
            try:
                client.responses.create = _wrap_create(client.responses.create)
            except Exception:
                pass
            client._pokeagent_usage_wrapped = True
            return client

        def _patched_create_openai_client(self, *client_args, **client_kwargs):
            client = original_create_client(self, *client_args, **client_kwargs)
            return _wrap_openai_client(client)

        def _patched_build_assistant_message(self, assistant_message, finish_reason: str):
            message = original_build_assistant_message(self, assistant_message, finish_reason)
            snapshot = usage_state.pop("last_snapshot", None)
            if snapshot is not None:
                _capture_usage_snapshot(snapshot, self.session_id, self.model)
            return message

        def _patched_build_api_kwargs(self, api_messages: list[dict[str, Any]]) -> dict:
            transformed: list[dict[str, Any]] = []
            for msg in api_messages:
                if not isinstance(msg, dict) or msg.get("role") != "tool" or not isinstance(msg.get("content"), str):
                    transformed.append(msg)
                    continue

                try:
                    parsed = json.loads(msg["content"])
                except json.JSONDecodeError:
                    transformed.append(msg)
                    continue

                if not isinstance(parsed, dict):
                    transformed.append(msg)
                    continue

                ref = parsed.get("_pokeagent_multimodal_ref")
                stored = None if _disable_multimodal else (multimodal_store.get(ref) if isinstance(ref, str) else None)
                # #region agent log
                _debug_log(
                    debug_log_path,
                    "H3",
                    "hermes_wrapper.py:_patched_build_api_kwargs",
                    "build_api_kwargs_tool_msg",
                    {"has_ref": ref is not None, "ref": ref, "stored_found": stored is not None, "multimodal_disabled": _disable_multimodal},
                )
                # #endregion
                if not stored:
                    transformed.append(msg)
                    continue

                tool_text = parsed.get("result", "")
                tool_name = parsed.get("_pokeagent_tool_name") or stored.get("tool_name") or "tool"
                tool_msg = dict(msg)
                tool_msg["content"] = json.dumps({"result": tool_text}, ensure_ascii=False)
                transformed.append(tool_msg)

                image_parts = [
                    part
                    for part in stored.get("parts", [])
                    if isinstance(part, dict) and part.get("type") == "image_url"
                ]
                # #region agent log
                _debug_log(
                    debug_log_path,
                    "H3",
                    "hermes_wrapper.py:_patched_build_api_kwargs",
                    "multimodal_inject",
                    {"image_parts_count": len(image_parts), "tool_name": tool_name},
                )
                # #endregion
                if image_parts:
                    transformed.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Visual observation from the previous `{_normalize_tool_name(str(tool_name))}` "
                                        "tool result. Use the attached image when reasoning about the current game state. "
                                        "The tool result text is already in the preceding tool message."
                                    ),
                                },
                                *image_parts,
                            ],
                        }
                    )

            return original_build_api_kwargs(self, transformed)

        AIAgent._create_openai_client = _patched_create_openai_client
        AIAgent._build_assistant_message = _patched_build_assistant_message
        AIAgent._build_api_kwargs = _patched_build_api_kwargs

    _patch_mcp_image_bridge()
    _patch_agent_runtime()

    def tool_progress_callback(tool_name: str, _preview: str, arguments: dict[str, Any]) -> None:
        nonlocal tool_event_counter
        tool_event_counter += 1
        normalized_args = arguments if isinstance(arguments, dict) else {}
        normalized_name = _normalize_tool_name(tool_name)
        _emit(
            {
                "type": "tool_use",
                "tool_use_id": f"{resume_session_id or 'session'}-tool-{tool_event_counter}",
                "name": normalized_name,
                "tool_name": normalized_name,
                "raw_tool_name": tool_name,
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
        _patch_registered_mcp_handlers(agent)

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
                "num_turns": int(usage_state["api_call_index"] or getattr(agent, "session_api_calls", 0) or 0),
                "duration_ms": duration_ms,
                "duration_api_ms": 0,
                "total_cost_usd": float(usage_state["total_cost_usd"] or 0.0),
                "is_error": bool(result.get("failed")) if isinstance(result, dict) else False,
                "error": result.get("error", "") if isinstance(result, dict) else "",
                "usage": {
                    "input_tokens": int(usage_state["prompt_tokens"] or getattr(agent, "session_prompt_tokens", 0) or 0),
                    "output_tokens": int(usage_state["completion_tokens"] or getattr(agent, "session_completion_tokens", 0) or 0),
                    "total_tokens": int(usage_state["total_tokens"] or getattr(agent, "session_total_tokens", 0) or 0),
                    "cached_tokens": int(usage_state["cached_tokens"] or 0),
                    "cache_write_tokens": int(usage_state["cache_write_tokens"] or 0),
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
