"""Reward functions for offline GRPO training.

Each function follows TRL's GRPOTrainer calling convention:
    def reward_func(prompts, completions, completion_ids, **kwargs) -> list[float]

The ``**kwargs`` contain every extra column from the HF Dataset, plus
``trainer_state``.  We pass ``gemini_response`` (the teacher's raw output)
and ``pre_state`` (JSON-encoded game state dict) through the dataset so
reward functions can compare the model's completion against the reference.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── tool name extraction ─────────────────────────────────────────────

# Same regex as vlm_backends.py:2642
_BRACKET_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]\s*")
_CALLING_RE = re.compile(r"(?:Calling|Tool:)\s*([a-zA-Z_]\w*)")
# Gemma4 function-call format from SFT: "call:tool_name args:{...}"
_CALL_COLON_RE = re.compile(r"^\s*call:([a-zA-Z_]\w*)", re.MULTILINE)


def _extract_tool_name(text: str) -> str | None:
    """Return the first tool name found in *text*, or None.

    Supports all formats produced by the SFT model:
    - ``[tool_name] ANALYZE: ...`` (bracket format, majority)
    - ``Calling tool_name(args)`` (prose format)
    - ``Tool: name`` (prose format)
    - ``call:tool_name args:{...}`` (Gemma4 function-call format)
    """
    m = _BRACKET_RE.search(text)
    if m:
        return m.group(1)
    m = _CALLING_RE.search(text)
    if m:
        return m.group(1)
    m = _CALL_COLON_RE.search(text)
    if m:
        return m.group(1)
    return None


# ── button list extraction ───────────────────────────────────────────

_BTN_NAMES = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R"}
_JSON_LIST_RE = re.compile(r'\[(["\'][^]]+)\]')
_PLAN_RE = re.compile(r'(?:PLAN|Action|buttons?)[:\s]+(.+?)(?:\.|$)', re.IGNORECASE)


def _extract_buttons(text: str) -> list[str]:
    """Extract a list of button presses from *text*."""
    # Try JSON-like list: ["A", "DOWN"]
    m = _JSON_LIST_RE.search(text)
    if m:
        try:
            import ast
            buttons = ast.literal_eval(m.group(0))
            if isinstance(buttons, list):
                return [b.upper() for b in buttons if str(b).upper() in _BTN_NAMES]
        except Exception:
            pass

    # Fallback: "Press LEFT x5" or comma-separated names
    m = _PLAN_RE.search(text)
    region = m.group(1) if m else text
    buttons: list[str] = []
    # Handle "LEFT x5" → 5 copies
    for token in re.split(r"[,\s]+", region):
        token = token.strip().strip("'\"").upper()
        if token in _BTN_NAMES:
            buttons.append(token)
    return buttons


def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two lists (treated as multisets)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0


# ── state checking ───────────────────────────────────────────────────

def _parse_pre_state(raw: Any) -> dict:
    """Safely parse pre_state which arrives as a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


# ── completion text extraction ───────────────────────────────────────

def _completion_text(completion: Any) -> str:
    """Extract plain text from a completion (may be str or list of dicts)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Conversational format: [{role: "assistant", content: "..."}]
        parts = []
        for msg in completion:
            c = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
        return " ".join(parts)
    return str(completion)


# =====================================================================
# Reward functions
# =====================================================================


def tool_match_reward(
    prompts: list,
    completions: list,
    completion_ids: list | None = None,
    **kwargs: Any,
) -> list[float]:
    """1.0 if the model picked the same tool as Gemini, else 0.0."""
    gemini_responses = kwargs.get("gemini_response", [])
    rewards = []
    for comp, ref in zip(completions, gemini_responses, strict=False):
        comp_text = _completion_text(comp)
        model_tool = _extract_tool_name(comp_text)
        ref_tool = _extract_tool_name(ref) if isinstance(ref, str) else None
        if model_tool and ref_tool and model_tool == ref_tool:
            rewards.append(1.0)
        elif model_tool is None and ref_tool is None:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def action_similarity_reward(
    prompts: list,
    completions: list,
    completion_ids: list | None = None,
    **kwargs: Any,
) -> list[float]:
    """Jaccard similarity of button lists for press_buttons, arg-key overlap otherwise."""
    gemini_responses = kwargs.get("gemini_response", [])
    rewards = []
    for comp, ref in zip(completions, gemini_responses, strict=False):
        comp_text = _completion_text(comp)
        ref_text = ref if isinstance(ref, str) else ""
        model_tool = _extract_tool_name(comp_text)
        ref_tool = _extract_tool_name(ref_text)

        if model_tool == "press_buttons" and ref_tool == "press_buttons":
            model_btns = _extract_buttons(comp_text)
            ref_btns = _extract_buttons(ref_text)
            rewards.append(_jaccard(model_btns, ref_btns))
        elif model_tool and ref_tool and model_tool == ref_tool:
            # Same non-press_buttons tool — partial credit
            rewards.append(0.5)
        elif model_tool and ref_tool:
            # Different tools
            rewards.append(0.0)
        else:
            # At least one side has no tool
            rewards.append(0.0)
    return rewards


def state_accuracy_reward(
    prompts: list,
    completions: list,
    completion_ids: list | None = None,
    **kwargs: Any,
) -> list[float]:
    """Does the model's reasoning mention the correct game state?

    Scores 0.0–1.0 based on:
    - 0.5 for mentioning the correct location name
    - 0.25 for correctly identifying battle status
    - 0.25 for correctly identifying dialog status
    """
    pre_states = kwargs.get("pre_state", [])
    rewards = []
    for comp, ps_raw in zip(completions, pre_states, strict=False):
        comp_text = _completion_text(comp).lower()
        ps = _parse_pre_state(ps_raw)
        score = 0.0

        # Location match (0.5)
        location = ps.get("location", "")
        if location:
            # Normalize: "ROUTE 104 MR BRINEYS HOUSE ALT" → check substrings
            loc_words = [w.lower() for w in location.split() if len(w) > 2]
            if loc_words:
                matched = sum(1 for w in loc_words if w in comp_text)
                score += 0.5 * min(matched / max(len(loc_words), 1), 1.0)

        # Battle status (0.25)
        is_battle = ps.get("is_in_battle", False)
        battle_words = {"battle", "fight", "opponent", "wild", "trainer"}
        text_mentions_battle = any(w in comp_text for w in battle_words)
        if is_battle == text_mentions_battle:
            score += 0.25

        # Dialog status (0.25)
        dialog_active = ps.get("dialog_active", False)
        dialog_words = {"dialog", "dialogue", "talking", "says", "text box", "prompt"}
        text_mentions_dialog = any(w in comp_text for w in dialog_words)
        if dialog_active == text_mentions_dialog:
            score += 0.25

        rewards.append(score)
    return rewards


def format_reward(
    prompts: list,
    completions: list,
    completion_ids: list | None = None,
    **kwargs: Any,
) -> list[float]:
    """1.0 if the completion has a parseable tool call + ANALYZE/PLAN structure."""
    rewards = []
    for comp in completions:
        comp_text = _completion_text(comp)
        has_tool = _extract_tool_name(comp_text) is not None
        has_analyze = bool(re.search(r"ANALYZE:", comp_text, re.IGNORECASE))
        if has_tool and has_analyze:
            rewards.append(1.0)
        elif has_tool:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards
