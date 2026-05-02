#!/bin/bash
# Test commands for the three scaffolds

# AutoEvolve scaffold (H_auto: empty registry + harness evolution)
uv run python run.py --game red --backend gemini --model-name gemini-3.1-pro-preview \
  --port 2778 --agent-auto --scaffold autoevolve \
  --backup-state PokemonRed-GBC/red_init.zip \
  --enable-prompt-optimization --optimization-window-length 50 \
  --direct-objectives autonomous_objective_creation \
  --direct-objectives-start 4 --run-name ae_autoevolve

# Simple scaffold (H_min: empty registry, no evolution)
uv run python run.py --game red --backend gemini --model-name gemini-3.1-pro-preview \
  --port 2878 --agent-auto --scaffold simple \
  --backup-state PokemonRed-GBC/red_init.zip \
  --direct-objectives autonomous_objective_creation \
  --direct-objectives-start 4 --run-name ae_simple

# PokeAgent scaffold (H_expert: full built-in subagents + walkthrough)
uv run python run.py --game red --backend gemini --model-name gemini-3.1-pro-preview \
  --port 2978 --agent-auto --scaffold pokeagent \
  --backup-state PokemonRed-GBC/red_init.zip \
  --direct-objectives autonomous_objective_creation \
  --direct-objectives-start 4 --run-name ae_pokeagent
