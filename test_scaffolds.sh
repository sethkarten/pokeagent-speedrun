#!/bin/bash
# Test commands for the three scaffolds

# AutoEvolve scaffold (H_auto: empty registry + harness evolution)
uv run python run.py --backend gemini --model-name gemini-3.1-pro-preview \
  --backup-state Emerald-GBAdvance/auto-evolve_init.zip \
  --port 8778 --agent-auto --scaffold autoevolve \
  --enable-prompt-optimization --optimization-frequency 50 \
  --direct-objectives autonomous_objective_creation \
  --direct-objectives-start 6 --record --run-name ae_autoevolve

# Simple scaffold (H_min: empty registry, no evolution)
uv run python run.py --backend gemini --model-name gemini-3-flash-preview \
  --backup-state Emerald-GBAdvance/auto-evolve_init.zip \
  --port 8009 --agent-auto --scaffold simple \
  --direct-objectives autonomous_objective_creation \
  --direct-objectives-start 6 --record --run-name ae_simple

# PokeAgent scaffold (H_expert: full built-in subagents + walkthrough)
uv run python run.py --backend gemini --model-name gemini-3-flash-preview \
  --backup-state Emerald-GBAdvance/auto-evolve_init.zip \
  --port 8011 --agent-auto --scaffold pokeagent \
  --direct-objectives autonomous_objective_creation \
  --direct-objectives-start 6 --record --run-name ae_pokeagent
