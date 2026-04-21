# Pokemon Gemma4 agent eval

- **Dataset**: emerald_v3 (20 samples)
- **Prompt mode**: real
- **Models**: gemma4-emerald:31b, gemma4-emerald-grpo:26b
- **Judge**: gemini-2.5-flash

## Overall Scores


| Metric | gemma4-emerald:31b | gemma4-emerald-grpo:26b |
|---|---|---|
| tool_format | 0.35 | 0.00 |
| actionable | 0.35 | 0.00 |
| grounding | 0.45 | 0.40 |
| action_relevance | 0.25 | 0.00 |
| reasoning_similarity | 0.05 | 0.00 |
| hallucination | 0.50 | 0.00 |
| degenerate | 0.10 | 0.00 |
| tok_s | 26.66 t/s | 26.27 t/s |

## Overworld


| Metric | gemma4-emerald:31b | gemma4-emerald-grpo:26b |
|---|---|---|
| tool_format | 0.60 | 0.00 |
| actionable | 0.60 | 0.00 |
| grounding | 0.57 | 0.55 |
| action_relevance | 0.42 | 0.00 |
| reasoning_similarity | 0.08 | 0.00 |
| hallucination | 0.67 | 0.00 |
| degenerate | 0.10 | 0.00 |
| tok_s | 32.09 t/s | 31.48 t/s |

## Battle


| Metric | gemma4-emerald:31b | gemma4-emerald-grpo:26b |
|---|---|---|
| tool_format | 0.10 | 0.00 |
| actionable | 0.10 | 0.00 |
| grounding | 0.33 | 0.25 |
| action_relevance | 0.00 | 0.00 |
| reasoning_similarity | 0.00 | 0.00 |
| hallucination | 0.25 | 0.00 |
| degenerate | 0.10 | 0.00 |
| tok_s | 21.24 t/s | 21.06 t/s |

## Sample Responses

### Example 1: overworld @ RUSTBORO CITY MART
- **Teacher**: `[press_buttons] ANALYZE: Location: Rustboro City Mart. Objective: Restock supplies. Situation: The cashier said "Welcome!". PLAN: Action: Press A. Reason: Advance the dialogue to open the shop menu. Expected result: The `
- **gemma4-emerald:31b**: `[press_buttons] ANALYZE: Location: Rustboro City Mart. Objective: Restock supplies. Situation: I am standing in front of the cashier, but the shop dialogue hasn't appeared yet. PLAN: Action: Press A. Reason: Initiate the`
- **gemma4-emerald-grpo:26b**: `<unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49>`

### Example 4: overworld @ Map_11_01
- **Teacher**: `[press_buttons] ANALYZE: [game screen: Player inside Pretty Petal Flower Shop at (2, 8). Objective: Obtain Wailmer Pail.] Situation: Need to talk to the NPCs inside the shop. The nearest NPC is to the north-east. PLAN: [`
- **gemma4-emerald:31b**: `[press_buttons] ANALYZE: [game screen: Player at (2, 8) in Petalburg Woods. Objective: Reach Rustboro City.] Situation: Just finished a battle. Previous subagent hit safety cap. Following Engineer's Protocol for manual m`
- **gemma4-emerald-grpo:26b**: `<unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49>`

### Example 7: overworld @ SLATEPORT CITY MART
- **Teacher**: `[press_buttons] ANALYZE: [game screen, location, objective, situation] Location: Slateport City Mart (3,7). Objective: Exit the Mart. Situation: Player is on the door mat (3,7). PLAN: Action: Press DOWN. Reason: Step on `
- **gemma4-emerald:31b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`
- **gemma4-emerald-grpo:26b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`

### Example 10: overworld @ SLATEPORT CITY OCEANIC MUSEUM 1F
- **Teacher**: `[complete_direct_objective] ANALYZE: Objective was to enter the Oceanic Museum. The location is now SLATEPORT CITY OCEANIC MUSEUM 1F at (9, 8). Evidence: Game text and screen show we are inside the museum. PLAN: Objectiv`
- **gemma4-emerald:31b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`
- **gemma4-emerald-grpo:26b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`

### Example 13: battle @ ROUTE 110
- **Teacher**: `[press_buttons] ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Poochyena is at critical HP. Zubat Lv 15 is healthy.] SITUATION: The objective is to finish the battle to gain EXP for Zubat. Poochyena's HP is low e`
- **gemma4-emerald:31b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`
- **gemma4-emerald-grpo:26b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`

### Example 16: battle @ ROUTE 102
- **Teacher**: `[execute_custom_subagent] ANALYZE: The battle transition animation is playing, and the game state indicates we are in Battle Mode against Trainer CASEY. Objective: Defeat the trainer's Pokemon to gain experience and mone`
- **gemma4-emerald:31b**: `<unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49>`
- **gemma4-emerald-grpo:26b**: `<unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49><unused49>`
