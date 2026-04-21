"""Evaluation pipeline for Pokemon Gemma4 agent models.

Entry point: ``python -m eval.run_eval`` (or ``uv run eval/run_eval.py``).

Modules:
- ``data_loader`` — load SFT JSONL shards, resolve screenshots, sample diverse examples.
- ``scorers`` — heuristic metrics (tool_format, grounding, degenerate, actionable).
- ``judge`` — Gemini-based LLM judge for action_relevance, reasoning_similarity, hallucination.
- ``run_eval`` — orchestrator: Ollama inference + scoring + aggregation + JSON/Markdown output.
"""
