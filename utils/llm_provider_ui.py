"""Map LLM log rows to UI color buckets (Gemini / OpenAI / Anthropic / other).

OpenRouter uses provider slugs in model ids (``google/gemini-...``, ``anthropic/claude-...``).
Native backends encode the API in ``interaction_type`` (``gemini_planner``, etc.).
"""


def _family_from_model_heuristic(model_lower: str) -> str:
    if "gemini" in model_lower or "palm" in model_lower:
        return "gemini"
    if "claude" in model_lower:
        return "anthropic"
    if any(x in model_lower for x in ("gpt", "o3", "o4", "o5", "codex")):
        return "openai"
    return "other"


def _family_from_openrouter_model(model_lower: str) -> str:
    if "/" in model_lower:
        org = model_lower.split("/", 1)[0]
        if org in ("google", "google-ai-platform", "gemini"):
            return "gemini"
        if org == "anthropic":
            return "anthropic"
        if org == "openai":
            return "openai"
    return _family_from_model_heuristic(model_lower)


def infer_llm_provider_family(
    interaction_type: str,
    model: str = "",
    metadata_backend: str | None = None,
) -> str:
    """Return one of: ``gemini``, ``openai``, ``anthropic``, ``other``."""
    it = (interaction_type or "").strip()
    model_l = (model or "").strip().lower()
    mb = (metadata_backend or "").strip().lower() if metadata_backend else ""

    if it:
        prefix = it.split("_", 1)[0].lower()
        if prefix in ("gemini", "vertex"):
            return "gemini"
        if prefix == "openai":
            return "openai"
        if prefix == "anthropic":
            return "anthropic"
        if prefix == "openrouter":
            return _family_from_openrouter_model(model_l)

    if mb in ("gemini", "vertex"):
        return "gemini"
    if mb == "openai":
        return "openai"
    if mb == "anthropic":
        return "anthropic"
    if mb == "openrouter":
        return _family_from_openrouter_model(model_l)

    return _family_from_model_heuristic(model_l)
