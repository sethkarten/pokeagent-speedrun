# SFT Quality Evaluation: Base vs Fine-tuned Gemma4 26B

**Date**: 2026-04-12
**Dataset**: emerald_v3 (20 orchestrator samples, diverse locations)
**Models**: gemma4:26b (base Q4_K_M), gemma4-emerald:26b (SFT LoRA r=256 merged, Q4_K_M)
**Eval prompt**: Simplified (screenshot + game state + "choose best action")

## Summary Scores

| Metric | Base (gemma4:26b) | SFT (gemma4-emerald:26b) | Delta |
|--------|-------------------|-------------------------|-------|
| tool_format | 0.05 | **0.55** | +0.50 |
| grounding | **0.78** | 0.75 | -0.03 |
| action_relevance | 0.53 | **0.68** | +0.15 |
| reasoning_similarity | 0.40 | **0.61** | +0.21 |
| tok/s | 190 | 183 | -7 |

## By Game State

### Overworld (n=10)

| Metric | Base | SFT | Delta |
|--------|------|-----|-------|
| tool_format | 0.00 | **0.40** | +0.40 |
| grounding | **0.70** | 0.55 | **-0.15** |
| action_relevance | 0.50 | **0.65** | +0.15 |

### Battle (n=10)

| Metric | Base | SFT | Delta |
|--------|------|-----|-------|
| tool_format | 0.10 | **0.70** | +0.60 |
| grounding | 0.85 | **0.95** | +0.10 |
| action_relevance | 0.55 | **0.70** | +0.15 |

## Key Findings

### What SFT improved
1. **Tool call format (+0.50)**: SFT model produces `[press_buttons]` bracket-format tool calls 55% of the time vs 5% for base. The base model writes generic prose descriptions instead.
2. **Battle grounding (+0.10)**: SFT model correctly identifies battle state, recognizes Pokemon names, HP values, and battle menus more reliably.
3. **Reasoning quality (+0.21)**: SFT responses use game-specific vocabulary (route names, Pokemon, tool names) matching the Gemini teacher more closely.
4. **Action relevance (+0.15)**: SFT picks contextually appropriate tools more often.

### What SFT degraded
1. **Overworld grounding (-0.15)**: The SFT model hallucinates dialogue boxes in overworld scenes. It says "The screenshot shows a prompt asking 'Do you want to take the POKÉ BALLS?'" when the screen shows a normal overworld. This is the critical bug we observed in the live eval.
2. **Some responses are degenerate**: The SFT model occasionally outputs fragments of its training context (action histories, tool schemas) instead of actual reasoning.

### Root cause analysis
The overworld hallucination problem likely stems from:
- **Training data bias**: The action history text in training prompts describes past game events. The model learned to parrot these descriptions rather than reading the screenshot.
- **Vision-text conflict**: When the text context mentions dialogue events, the model trusts text over visual input. The SFT didn't sufficiently teach the model to ground responses in the current screenshot.
- **No negative examples**: SFT only showed correct (state, action) pairs. The model never learned what happens when you describe the wrong screen state.

## Sample Comparisons

### Battle @ Route 109 (SFT wins)
**Teacher**: `[press_buttons] ANALYZE: Battle with Sailor on Route 109. Marshtomp at 23/57 HP...`
**Base**: *(empty/generic response)*
**SFT**: `[press_buttons] ANALYZE: The screenshot shows a battle against a Sailor trainer on Route 109. The battle menu is open with the cursor on FIGHT. Marshtomp is at 23/57 HP...` -- correctly reads HP, identifies trainer, picks FIGHT

### Overworld @ Dewford Town (SFT fails)
**Teacher**: `[press_buttons] ANALYZE: Overworld. Location: Dewford Town (8,11). Need to return to Gym...`
**Base**: *(correctly describes overworld scene)*
**SFT**: `ANALYZE: The player is currently in a dialogue with a Devon Researcher...` -- hallucinating a dialogue that doesn't exist on screen

## Performance
- Both models: ~180-195 tok/s generation on RTX 5090 (Q4_K_M)
- SFT responses are shorter on average (faster per-step)
- No significant speed difference between base and fine-tuned

## Implications for RL

The SFT model is a strong initialization for RL:
- It already knows the tool vocabulary and game domain
- Battle performance is genuinely good (0.95 grounding, 0.70 tool format)
- The overworld hallucination problem is exactly what RL with visual grounding reward can fix

**RL reward signals needed:**
1. **Visual grounding**: Compare model's state description to actual game state API (location, is_in_battle, dialog_active). Penalize mismatches.
2. **Game progression**: Reward milestone advancement, position change, new locations visited.
3. **Tool format**: Binary reward for producing parseable tool calls.
4. **Anti-repetition**: Penalize N consecutive identical actions with no state change.
