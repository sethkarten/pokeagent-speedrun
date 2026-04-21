"""Heuristic scorers that do not require an LLM judge.

Mirrors metrics used in ``data/eval_*.json``:
- tool_format (0/1)
- grounding (0-1) — substring heuristic inspired by
  ``train.reward_functions.state_accuracy_reward``
- degenerate (0/1) — does the response echo training context?
- actionable (0/1) — does the response contain a concrete executable action?
"""

from __future__ import annotations

import re
from typing import Iterable

# Reuse canonical tool-name regex from train.reward_functions
from train.reward_functions import _extract_tool_name, _extract_buttons  # noqa: E402


# ── tool_format ──────────────────────────────────────────────────────


def score_tool_format(response: str) -> float:
    """1.0 if the response contains a parseable tool name in canonical format."""
    if not response:
        return 0.0
    return 1.0 if _extract_tool_name(response) is not None else 0.0


# ── grounding ────────────────────────────────────────────────────────


def score_grounding(response: str, pre_state: dict) -> float:
    """0-1 score based on how well the response matches pre_state.

    Mirrors state_accuracy_reward:
    - 0.5 for mentioning the location name (word-level fraction match)
    - 0.25 for correctly identifying battle status
    - 0.25 for correctly identifying dialog status
    """
    if not response:
        return 0.0
    text = response.lower()
    score = 0.0

    location = pre_state.get("location") or ""
    if location:
        loc_words = [w.lower() for w in re.split(r"[\s_]+", location) if len(w) > 2]
        if loc_words:
            matched = sum(1 for w in loc_words if w in text)
            score += 0.5 * min(matched / max(len(loc_words), 1), 1.0)
        else:
            # Fallback for single-token locations (e.g. Red game: "Route4")
            if location.lower() in text:
                score += 0.5
    else:
        # No location info — give the 0.5 for free (can't penalize)
        score += 0.5

    is_battle = bool(pre_state.get("is_in_battle", False))
    battle_words = ("battle", "fight", "opponent", "wild", "trainer", "versus")
    mentions_battle = any(w in text for w in battle_words)
    if is_battle == mentions_battle:
        score += 0.25

    dialog = bool(pre_state.get("dialog_active", False))
    dialog_words = ("dialog", "dialogue", "talking", "speak", "text box", "textbox", "prompt")
    mentions_dialog = any(w in text for w in dialog_words)
    if dialog == mentions_dialog:
        score += 0.25

    return round(score, 6)


# ── degenerate ───────────────────────────────────────────────────────

_DEGENERATE_MARKERS: tuple[str, ...] = (
    "action history",
    "recent actions",
    "tool schemas",
    "# tool schemas",
    "📋 results from previous turn",
    "mind map results query log",
    "subagent registry",
    "# subagent registry",
    "### action history",
    "tools:\n  -",
    "## current direct objective",
    "# strategic guidance",
    "📖 story objective",
)


def score_degenerate(response: str, prompt: str | None = None) -> float:
    """1.0 if the response looks like it's echoing training context.

    Checks:
    - Known marker strings from the harness prompt template
    - If ``prompt`` is passed, also checks for long verbatim substring overlaps
      (≥120 char contiguous substring of prompt appearing in response).
    """
    if not response:
        return 0.0
    lower = response.lower()
    for marker in _DEGENERATE_MARKERS:
        if marker in lower:
            return 1.0

    if prompt and len(response) > 150 and len(prompt) > 500:
        # Sample a few windows of the prompt and check for long contiguous copies
        step = max(len(prompt) // 20, 400)
        for start in range(0, len(prompt) - 150, step):
            chunk = prompt[start : start + 120]
            if len(chunk.strip()) < 80:
                continue
            if chunk in response:
                return 1.0
    return 0.0


# ── actionable ───────────────────────────────────────────────────────

_ACTIONABLE_TOOL_KEYWORDS = (
    "press_buttons",
    "navigate_to_coords",
    "execute_custom_subagent",
    "run_code",
    "run_skill",
    "process_memory",
    "process_skill",
    "process_subagent",
    "replan_objectives",
    "complete_direct_objective",
    "mouse_click",
    "hold_key",
)


def score_actionable(response: str) -> float:
    """1.0 if response picks a concrete tool OR contains a button sequence.

    Avoids rewarding pure prose descriptions that do not commit to an action.
    """
    if not response:
        return 0.0
    if _extract_tool_name(response) is not None:
        return 1.0
    lower = response.lower()
    for kw in _ACTIONABLE_TOOL_KEYWORDS:
        if kw in lower:
            return 1.0
    # Concrete button list (e.g. "Press LEFT, LEFT, A")
    btns = _extract_buttons(response)
    if len(btns) >= 2:
        return 1.0
    return 0.0


# ── aggregation helpers ──────────────────────────────────────────────


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
