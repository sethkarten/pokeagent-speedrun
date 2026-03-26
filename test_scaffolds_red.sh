#!/bin/bash
# Test commands for the three scaffolds

# AutoEvolve scaffold (H_auto: empty registry + harness evolution)
uv run python run.py --game red --backend gemini --model-name gemini-3-flash-preview \
  --port 2778 --agent-auto --scaffold autoevolve \
  --enable-prompt-optimization --optimization-frequency 50 \
  --direct-objectives autonomous_objective_creation \
  --run-name ae_autoevolve

# Simple scaffold (H_min: empty registry, no evolution)
uv run python run.py --game red --backend gemini --model-name gemini-3-flash-preview \
  --port 2779 --agent-auto --scaffold simple \
  --direct-objectives autonomous_objective_creation \
  --run-name ae_simple

# PokeAgent scaffold (H_expert: full built-in subagents + walkthrough)
uv run python run.py --game red --backend gemini --model-name gemini-3-flash-preview \
  --port 2780 --agent-auto --scaffold pokeagent \
  --direct-objectives autonomous_objective_creation \
  --run-name ae_pokeagent
