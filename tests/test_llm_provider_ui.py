"""Tests for OpenRouter / native backend UI provider bucketing."""

from utils.llm_provider_ui import infer_llm_provider_family


def test_native_prefixes():
    assert infer_llm_provider_family("gemini_planner", "gemini-2.0-flash") == "gemini"
    assert infer_llm_provider_family("vertex_planner", "gemini-2.5-pro") == "gemini"
    assert infer_llm_provider_family("openai_planner", "gpt-5") == "openai"
    assert infer_llm_provider_family("anthropic_planner", "claude-sonnet-4-5") == "anthropic"


def test_openrouter_slugs():
    assert infer_llm_provider_family("openrouter_planner", "google/gemini-2.5-flash") == "gemini"
    assert infer_llm_provider_family("openrouter_x", "anthropic/claude-3.5-sonnet") == "anthropic"
    assert infer_llm_provider_family("openrouter_x", "openai/gpt-4o") == "openai"


def test_openrouter_heuristic_and_other():
    assert infer_llm_provider_family("openrouter_x", "meta-llama/llama-3.3-70b") == "other"
    assert infer_llm_provider_family("openrouter_x", "mistralai/mistral-small") == "other"


def test_metadata_fallback():
    assert infer_llm_provider_family("unknown", "gpt-4o", metadata_backend="openai") == "openai"
