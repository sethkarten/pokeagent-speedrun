"""Gemini-based LLM judge for eval metrics that require semantic comparison.

Metrics scored here:
- action_relevance (0-1): does the model's chosen tool fit the state?
- reasoning_similarity (0-1): does the model's reasoning align with the teacher's?
- hallucination (0-1): does the response fabricate visual details?

The judge sends a short prompt to Gemini and parses a JSON response.
If Gemini fails twice in a row, returns None for that metric (caller should
treat None as "metric skipped" and not factor it into aggregates).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class JudgeScores:
    action_relevance: Optional[float] = None
    reasoning_similarity: Optional[float] = None
    hallucination: Optional[float] = None


_JSON_BLOB_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_json_scores(text: str) -> dict:
    """Extract a JSON object with score fields from model output."""
    if not text:
        return {}
    # Fast path: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: pick the first {...} blob
    for m in _JSON_BLOB_RE.finditer(text):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return {}


def _coerce_float_0_1(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return max(0.0, min(1.0, x))


class GeminiJudge:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. `uv add google-generativeai`."
            ) from e

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set (check .env)")

        genai.configure(api_key=api_key)
        self.genai = genai
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def _call(self, parts: list[Any], *, max_retries: int = 1, timeout_s: float = 45.0) -> Optional[str]:
        """Call Gemini with (text + optional image) parts, return plain text or None on failure."""
        from google.api_core import exceptions as gac_exceptions  # type: ignore

        for attempt in range(max_retries + 1):
            try:
                resp = self.model.generate_content(
                    parts,
                    generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
                    request_options={"timeout": timeout_s},
                )
                text = getattr(resp, "text", None)
                if text:
                    return text
                # fallback extraction
                if getattr(resp, "candidates", None):
                    for cand in resp.candidates:
                        content = getattr(cand, "content", None)
                        if content and getattr(content, "parts", None):
                            for p in content.parts:
                                if getattr(p, "text", None):
                                    return p.text
                return None
            except gac_exceptions.ResourceExhausted as e:
                logger.warning("Gemini 429, retrying after backoff: %s", e)
                time.sleep(2 + attempt * 3)
            except Exception as e:
                logger.warning("Gemini judge error (attempt %d): %s", attempt, e)
                time.sleep(1 + attempt)
        return None

    @staticmethod
    def _image_part(image_b64: Optional[str]) -> Optional[dict]:
        if not image_b64:
            return None
        try:
            base64.b64decode(image_b64[:16])  # sanity
        except Exception:
            return None
        return {"mime_type": "image/png", "data": base64.b64decode(image_b64)}

    def score(
        self,
        *,
        model_response: str,
        teacher_response: str,
        pre_state: dict,
        image_b64: Optional[str] = None,
    ) -> JudgeScores:
        """Return all three judge scores in a single prompt call."""
        if not model_response or not model_response.strip():
            # Empty response → action_relevance 0, reasoning_similarity 0, hallucination 0
            return JudgeScores(action_relevance=0.0, reasoning_similarity=0.0, hallucination=0.0)

        state_summary = json.dumps(
            {
                "location": pre_state.get("location"),
                "is_in_battle": pre_state.get("is_in_battle"),
                "dialog_active": pre_state.get("dialog_active"),
                "coords": pre_state.get("player_coords"),
            }
        )

        prompt_text = f"""You are grading a Pokemon-playing agent's single-step output.

You will be given:
- The current game state (location, battle flag, dialog flag).
- A reference response from a strong teacher model (Gemini 3 Pro).
- The model-under-test's response.
- Optionally, the screenshot the agent saw.

Score three metrics, each in [0.0, 1.0]:

1. action_relevance: Does the model's chosen tool/action make sense for this state?
   - 1.0 if the tool choice is appropriate for the situation (battle → fight/bag/run related; overworld → movement, interaction, navigation; dialog → press A).
   - 0.5 if plausible but suboptimal.
   - 0.0 if irrelevant, missing, or harmful.

2. reasoning_similarity: Semantic alignment between the model's reasoning and the teacher's reasoning.
   - 1.0 if the model describes the same situation, objective, and plan.
   - 0.5 if it gets the situation but differs on plan.
   - 0.0 if unrelated or contradicts the teacher.

3. hallucination: Binary flag for severe, state-contradicting fabrication.
   SCORE 0.0 BY DEFAULT. Only raise this if the response describes something that
   DIRECTLY CONTRADICTS the declared game state — e.g.:
     - says "dialog is up" when dialog_active=false and there is clearly no dialog
     - claims a battle is happening when is_in_battle=false
     - names a Pokemon, NPC, or item that is impossible in this location
   Do NOT flag:
     - detailed but plausible descriptions of the scene
     - strategic claims (cursor position, HP estimates) — those are reasoning errors, not hallucinations
     - matching the teacher's description closely
   - 0.0 = default, no clear contradiction.
   - 1.0 = severe contradiction with declared state.
   Partial scores only if multiple minor contradictions accumulate.

Return JSON ONLY, exactly like:
{{"action_relevance": 0.7, "reasoning_similarity": 0.5, "hallucination": 0.0}}

# State
{state_summary}

# Teacher response
{teacher_response[:2000]}

# Model-under-test response
{model_response[:2000]}
"""

        parts: list[Any] = [prompt_text]
        img_part = self._image_part(image_b64)
        if img_part is not None:
            parts.append(img_part)

        raw = self._call(parts)
        if raw is None:
            return JudgeScores()
        parsed = _parse_json_scores(raw)
        return JudgeScores(
            action_relevance=_coerce_float_0_1(parsed.get("action_relevance")),
            reasoning_similarity=_coerce_float_0_1(parsed.get("reasoning_similarity")),
            hallucination=_coerce_float_0_1(parsed.get("hallucination")),
        )
