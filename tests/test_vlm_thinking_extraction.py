"""Tests for multi-part agent-thinking extraction (parallel tool calls in one VLM turn)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.agent_infrastructure.vlm_backends import (
    _extract_thinking_from_gemini_like_response,
    _openai_text_part,
    _openai_tool_call_part,
    _thinking_from_content_parts,
)


def _adapter_from_parts(parts):
    content = type("Content", (), {"parts": parts})()
    candidate = type("Candidate", (), {"content": content})()
    return type("ResponseAdapter", (), {"candidates": [candidate]})()


class TestThinkingFromContentParts:
    def test_multiple_function_calls_newline_separated(self):
        parts = [
            _openai_tool_call_part("tool_a", {"reasoning": "do a"}),
            _openai_tool_call_part("tool_b", {"foo": 1, "bar": 2}),
        ]
        text = _thinking_from_content_parts(parts)
        lines = text.split("\n")
        assert len(lines) == 2
        assert "[tool_a]" in lines[0]
        assert "tool_b" in lines[1]

    def test_interleaved_text_and_calls_preserves_order(self):
        parts = [
            _openai_text_part("planning"),
            _openai_tool_call_part("x", {}),
        ]
        text = _thinking_from_content_parts(parts)
        assert text.split("\n") == ["planning", "Calling x()"]

    def test_empty_parts_fallback(self):
        assert _thinking_from_content_parts([]) == "[Executing function call]"


class TestExtractThinkingFromGeminiLikeResponse:
    def test_adapter_with_two_calls(self):
        adapter = _adapter_from_parts(
            [
                _openai_tool_call_part("first", {"reasoning": "r1"}),
                _openai_tool_call_part("second", {"reasoning": "r2"}),
            ]
        )
        out = _extract_thinking_from_gemini_like_response(adapter)
        assert "[first]" in out and "[second]" in out
        assert out.count("\n") == 1

    def test_missing_candidates(self):
        assert _extract_thinking_from_gemini_like_response(None) == "[Executing function call]"
        assert _extract_thinking_from_gemini_like_response(type("R", (), {"candidates": []})()) == (
            "[Executing function call]"
        )
