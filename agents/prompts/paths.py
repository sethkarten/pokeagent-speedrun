"""Canonical prompt paths used across the agents package."""

from pathlib import Path


PROMPTS_ROOT = "agents/prompts"
CLI_AGENT_DIRECTIVE_PATH = f"{PROMPTS_ROOT}/cli-agent-directives/pokemon_directive.md"
POKEAGENT_PROMPT_PATH = f"{PROMPTS_ROOT}/pokeagent-directives/POKEAGENT.md"
POKEAGENT_SYSTEM_PROMPT_PATH = f"{PROMPTS_ROOT}/pokeagent-directives/system_prompt.md"
SLAM_INSTRUCTIONS_PATH = f"{PROMPTS_ROOT}/pokeagent-directives/SLAM_INSTRUCTIONS.md"

# AutoEvolve directory (renamed from prompt-optimization/)
POKEAGENT_BASE_PROMPT_PATH = f"{PROMPTS_ROOT}/pokeagent-directives/auto-evolve/orchestrator_policy.md"
AUTOEVOLVE_BASE_SYSTEM_PROMPT_PATH = f"{PROMPTS_ROOT}/pokeagent-directives/auto-evolve/base_system_prompt.md"


def resolve_repo_path(relative_path: str) -> Path:
    """Resolve a repository-root-relative path to an absolute Path."""
    return Path(__file__).resolve().parents[2] / relative_path
