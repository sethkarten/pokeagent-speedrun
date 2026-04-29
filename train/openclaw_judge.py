"""Online per-step judge reward via Gemini-3-flash-preview.

Used as an additional reward function in GRPO (OpenClaw-RL style).
Per-step, per-completion: send (image, state summary, model_response,
teacher reference) to Gemini and parse a single scalar score in [0,1].

Reuses GeminiJudge but:
- default model is gemini-3-flash-preview for latency + cost
- simplified single-metric rubric that directly maps to GRPO reward
- image is loaded from image_path (the dataset stores paths, not b64)

The reward is a weighted combination of:
  0.5 * action_correctness   ("did the model pick a sensible action?")
  0.3 * reasoning_fidelity   ("does it match the teacher's reasoning?")
  0.2 * format_compliance    ("did it follow the harness tool-call format?")

Parallelized via a ThreadPoolExecutor so one training step doesn't serialize
all N completion judge calls.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("openclaw_judge")


_ENV_LOADED = False


def _load_env():
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    _ENV_LOADED = True


def _completion_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content") or ""
    return str(c)


def _parse_json_scores(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _coerce_01(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return max(0.0, min(1.0, x))


_IMAGE_LOCK = threading.Lock()
_IMAGE_CACHE: dict[str, bytes] = {}
_IMAGE_CACHE_MAX = 2000


def _load_image_bytes(image_path: Optional[str], image_b64: Optional[str]) -> Optional[bytes]:
    if image_b64:
        try:
            return base64.b64decode(image_b64)
        except Exception:
            pass
    if not image_path:
        return None
    with _IMAGE_LOCK:
        b = _IMAGE_CACHE.get(image_path)
    if b is not None:
        return b
    try:
        p = Path(image_path)
        if not p.is_absolute():
            # resolve relative to repo root (the dataset stores run_data/... paths)
            repo_root = Path(__file__).resolve().parent.parent
            p = repo_root / p
        if not p.exists():
            return None
        b = p.read_bytes()
    except Exception:
        return None
    with _IMAGE_LOCK:
        if len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
            _IMAGE_CACHE.clear()
        _IMAGE_CACHE[image_path] = b
    return b


# ── Gemini singleton ─────────────────────────────────────────────────

_JUDGE = None
_JUDGE_LOCK = threading.Lock()


_SESSION: Optional[Any] = None
_SESSION_LOCK = threading.Lock()


def _get_session():
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
    import requests as _req
    s = _req.Session()
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        s.proxies = {"https": proxy, "http": proxy}
    with _SESSION_LOCK:
        _SESSION = s
    return s


def _get_model(model_name: str):
    global _JUDGE
    with _JUDGE_LOCK:
        if _JUDGE is not None and _JUDGE.model_name == model_name:
            return _JUDGE
    _load_env()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    class _Model:
        def __init__(self, name):
            self.model_name = name
            self._api_key = api_key
            self._url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"models/{name}:generateContent?key={api_key}"
            )

        def generate(self, parts, *, timeout_s=90.0):
            api_parts: list[dict] = []
            for p in parts:
                if isinstance(p, str):
                    api_parts.append({"text": p})
                elif isinstance(p, (bytes, bytearray)):
                    api_parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(bytes(p)).decode("ascii"),
                        }
                    })
            payload = {
                "contents": [{"parts": api_parts}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 2048,
                },
            }
            try:
                s = _get_session()
                r = s.post(self._url, json=payload, timeout=timeout_s)
                r.raise_for_status()
                data = r.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    for p in content.get("parts", []):
                        if p.get("text"):
                            return p["text"]
            except Exception as e:
                logger.warning("gemini judge error: %s", e)
            return None

    with _JUDGE_LOCK:
        _JUDGE = _Model(model_name)
    return _JUDGE


# ── prompt builder ───────────────────────────────────────────────────

_JUDGE_PROMPT = """You are a process reward model (PRM) grading a Pokemon-playing agent's action in the context of its trajectory.

You are given:
- the current screenshot of the game (attached as an image)
- the game state summary and the teacher's trajectory context
- a structured RECENT TRAJECTORY window of recent actions/locations (use this to detect oscillation, regression, or backtracking)
- the CURRENT STORY OBJECTIVE from the Pokemon Red walkthrough (authoritative; tells you exactly what the agent should be doing right now)
- the student model's response (reasoning + chosen tool call)

Score four metrics in [0.0, 1.0] and return ONE JSON object:

1. action_correctness: Given what is visible on the screen + the state summary,
   is the chosen tool/button sequence a reasonable next action?
   Use the image to catch things the text can't describe — e.g. walking into a
   wall, ignoring a visible NPC/item, missing a clear exit.
   1.0 = clearly correct; 0.7 = plausible; 0.3 = wrong; 0.0 = irrelevant.

2. trajectory_progress: Does this action advance the agent toward the CURRENT STORY OBJECTIVE's completion_condition?
   1.0 = action clearly progresses toward the navigation_hint / target_location.
   0.5 = plausibly useful (e.g. moving in the right direction).
   0.2 = interacts with something relevant but not progressing toward the condition.
   0.0 = repeats previous actions, wrong direction, or **undoes recent progress visible in the RECENT TRAJECTORY window** (e.g. re-entering a building the agent just exited, walking back through a warp the agent already crossed).
   IMPORTANT: ground this score in the CURRENT STORY OBJECTIVE and use the RECENT TRAJECTORY to penalize regressions.

3. reasoning_quality: Does the model's reasoning correctly describe the current situation and form a coherent plan that aligns with the current objective?
   1.0 = accurate state description + plan matches navigation_hint; 0.5 = partial; 0.0 = hallucinated or off-task.

4. format_compliance: Did the model produce a parseable tool call in EITHER accepted format?
   Accepted: `[tool_name]` bracket OR `call:tool_name(args)` OR `ACTION: call:tool_name(...)` plus ANALYZE/PLAN sections.
   1.0 = full format (tool call + ANALYZE + PLAN); 0.5 = partial (tool call only);
   0.0 = no parseable tool call.

Your entire output must be a single JSON object. Example:
{{"action_correctness": 0.7, "trajectory_progress": 0.2, "reasoning_quality": 0.5, "format_compliance": 1.0}}

{objective_block}
{trajectory_block}
# State
{state}

# Teacher reference (one-line hint)
{teacher}

# Model-under-test response
{model}
"""


def _build_state_summary(pre_state_raw: Any) -> str:
    """Compact game-state summary fed to PRM + teacher.

    Includes structured map exits (`warp_events`) and interactables
    (`bg_events`) so Gemini can derive "next action that gets closer to
    a relevant exit/object" without per-objective coordinate hardcoding.
    Falls back to terse summary when the field isn't present.
    """
    if isinstance(pre_state_raw, str):
        try:
            d = json.loads(pre_state_raw)
        except Exception:
            return pre_state_raw[:500]
    elif isinstance(pre_state_raw, dict):
        d = pre_state_raw
    else:
        return ""

    out: dict[str, Any] = {
        "location": d.get("location"),
        "is_in_battle": d.get("is_in_battle"),
        "dialog_active": d.get("dialog_active"),
        "coords": d.get("player_coords") or {"x": d.get("x"), "y": d.get("y")},
    }
    # Structured world info: map exits + interactables on the current map.
    # The agent's prompt has these but the PRM/teacher prompt didn't see
    # them, leaving Gemini guessing at door coordinates from the
    # screenshot alone.
    map_data = d.get("map") or d.get("map_data") or {}
    if isinstance(map_data, dict):
        warps = map_data.get("warp_events")
        if warps:
            out["warp_events"] = [
                {"x": w.get("x"), "y": w.get("y"), "dest_map": w.get("dest_map")}
                for w in warps[:8]
            ]
        bg = map_data.get("bg_events")
        if bg:
            out["bg_events"] = [
                {"x": e.get("x"), "y": e.get("y"),
                 "symbol": e.get("symbol"), "script": e.get("script")}
                for e in bg[:8]
            ]
    # Fall back: top-level warp_events / bg_events in pre_state.
    if "warp_events" not in out and d.get("warp_events"):
        out["warp_events"] = [
            {"x": w.get("x"), "y": w.get("y"), "dest_map": w.get("dest_map")}
            for w in d["warp_events"][:8]
        ]
    if "bg_events" not in out and d.get("bg_events"):
        out["bg_events"] = [
            {"x": e.get("x"), "y": e.get("y"),
             "symbol": e.get("symbol"), "script": e.get("script")}
            for e in d["bg_events"][:8]
        ]
    return json.dumps(out)


def _score_one(
    model_text: str,
    teacher_text: str,
    state_summary: str,
    image_bytes: Optional[bytes],
    judge_model: str,
    objective_block: str = "",
    trajectory_block: str = "",
) -> tuple[float, dict]:
    if not model_text.strip():
        return 0.0, {"action_correctness": 0.0, "trajectory_progress": 0.0, "reasoning_quality": 0.0, "format_compliance": 0.0}
    traj_text = (trajectory_block or "").strip()
    traj_section = f"# Recent trajectory window\n{traj_text}\n" if traj_text else ""
    prompt = _JUDGE_PROMPT.format(
        state=state_summary,
        teacher=(teacher_text or "")[:2000],
        model=model_text[:1600],
        objective_block=(objective_block or "").strip(),
        trajectory_block=traj_section,
    )
    parts: list[Any] = [prompt]
    if image_bytes:
        parts.append(image_bytes)
    m = _get_model(judge_model)
    raw = None
    for attempt in range(3):
        raw = m.generate(parts, timeout_s=90.0)
        if raw is not None:
            break
        time.sleep(1 + attempt * 2)
    if raw is None:
        return 0.0, {}
    parsed = _parse_json_scores(raw)
    ac = _coerce_01(parsed.get("action_correctness")) or 0.0
    tp = _coerce_01(parsed.get("trajectory_progress")) or 0.0
    rq = _coerce_01(parsed.get("reasoning_quality")) or 0.0
    fc = _coerce_01(parsed.get("format_compliance")) or 0.0
    # PRM weighting: trajectory_progress is the dominant signal
    reward = 0.4 * tp + 0.3 * ac + 0.2 * rq + 0.1 * fc
    return reward, {"action_correctness": ac, "trajectory_progress": tp, "reasoning_quality": rq, "format_compliance": fc}


# ── GRPO reward function ─────────────────────────────────────────────

_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        return _EXECUTOR


def openclaw_judge_reward(
    prompts: list,
    completions: list,
    completion_ids: list | None = None,
    **kwargs: Any,
) -> list[float]:
    """Per-completion reward from Gemini-3-flash-preview online judge.

    Expects kwargs to contain per-row context:
      - gemini_response (list[str]) : teacher reference
      - pre_state (list[str|dict])  : JSON state dict or already-decoded
      - image_path (list[str]) or image_b64 (list[str])
    GRPO flattens these to one value per completion.
    """
    judge_model = os.environ.get("OPENCLAW_JUDGE_MODEL", "gemini-3-flash-preview")

    teacher = kwargs.get("gemini_response") or kwargs.get("raw_response") or [""] * len(completions)
    pre_state = kwargs.get("pre_state") or [{}] * len(completions)
    image_paths = kwargs.get("image_path") or [None] * len(completions)
    image_b64s = kwargs.get("image_b64") or [None] * len(completions)

    def _task(i):
        model_text = _completion_text(completions[i])
        st = _build_state_summary(pre_state[i] if i < len(pre_state) else {})
        img = _load_image_bytes(
            image_paths[i] if i < len(image_paths) else None,
            image_b64s[i] if i < len(image_b64s) else None,
        )
        reward, detail = _score_one(model_text, teacher[i] if i < len(teacher) else "",
                                    st, img, judge_model)
        return i, reward, detail

    rewards = [0.0] * len(completions)
    ex = _get_executor()
    futures = [ex.submit(_task, i) for i in range(len(completions))]
    try:
        for f in concurrent.futures.as_completed(futures, timeout=300):
            try:
                i, r, _ = f.result()
                rewards[i] = r
            except Exception as e:
                logger.warning("judge task failed: %s", e)
    except concurrent.futures.TimeoutError:
        logger.warning("judge batch timed out after 300s; %d/%d completions scored",
                       sum(1 for r in rewards if r > 0), len(rewards))
    return rewards
