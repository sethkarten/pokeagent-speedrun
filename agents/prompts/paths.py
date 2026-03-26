"""Canonical prompt paths used across the agents package."""

from pathlib import Path
import os


PROMPTS_ROOT = "agents/prompts"
game_type = os.environ.get("GAME_TYPE", "emerald").lower()  # game type: red or emerald (for now)

# ---------------------------------------------------------------------------
# Template rendering: prompts that differ only in game name use {game_name}
# ---------------------------------------------------------------------------
GAME_NAMES = {
    "emerald": "Pokemon Emerald",
    "red": "Pokemon Red",
}
GAME_NAME = GAME_NAMES.get(game_type, "Pokemon Emerald")


def render_prompt(content: str) -> str:
    """Substitute {game_name} in prompt content based on GAME_TYPE env var.

    Uses plain str.replace() instead of str.format_map() so that literal
    curly braces in prompt files (JSON examples, code blocks) don't cause
    ValueError.
    """
    game_name = GAME_NAMES.get(game_type, "Pokemon Emerald")
    return content.replace("{game_name}", game_name)


# ---------------------------------------------------------------------------
# Separate files: genuinely different content per game
# ---------------------------------------------------------------------------
_optimization_enabled_prompts = {
    "red": f"{PROMPTS_ROOT}/pokeagent-directives/system_prompt_red.md",
    "emerald": f"{PROMPTS_ROOT}/pokeagent-directives/system_prompt.md",
}
_default_system_prompts = {
    "red": f"{PROMPTS_ROOT}/pokeagent-directives/POKEAGENT_RED.md",
    "emerald": f"{PROMPTS_ROOT}/pokeagent-directives/POKEAGENT.md",
}

POKEAGENT_PROMPT_PATH = _default_system_prompts[game_type]
POKEAGENT_SYSTEM_PROMPT_PATH = _optimization_enabled_prompts[game_type]
SLAM_INSTRUCTIONS_PATH = f"{PROMPTS_ROOT}/pokeagent-directives/SLAM_INSTRUCTIONS.md"

_no_builtins_prompts = {
    "red": f"{PROMPTS_ROOT}/pokeagent-directives/POKEAGENT_NO_BUILTINS_RED.md",
    "emerald": f"{PROMPTS_ROOT}/pokeagent-directives/POKEAGENT_NO_BUILTINS.md",
}
POKEAGENT_NO_BUILTINS_PROMPT_PATH = _no_builtins_prompts[game_type]
# ---------------------------------------------------------------------------
# Templated files: single file with {game_name}, rendered at load time
# ---------------------------------------------------------------------------
CLI_AGENT_DIRECTIVE_PATH = f"{PROMPTS_ROOT}/cli-agent-directives/pokemon_directive.md"

# AutoEvolve directory (renamed from prompt-optimization/)
POKEAGENT_BASE_PROMPT_PATH = f"{PROMPTS_ROOT}/pokeagent-directives/auto-evolve/orchestrator_policy.md"
_autoevolve_base_system_prompts = {
    "red": f"{PROMPTS_ROOT}/pokeagent-directives/auto-evolve/base_system_prompt_red.md",
    "emerald": f"{PROMPTS_ROOT}/pokeagent-directives/auto-evolve/base_system_prompt.md",
}
AUTOEVOLVE_BASE_SYSTEM_PROMPT_PATH = _autoevolve_base_system_prompts[game_type]


def resolve_repo_path(relative_path: str) -> Path:
    """Resolve a repository-root-relative path to an absolute Path."""
    return Path(__file__).resolve().parents[2] / relative_path
