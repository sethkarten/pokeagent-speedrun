# Pokemon Gemma4 agent eval

- **Dataset**: red_v1 (20 samples)
- **Prompt mode**: real
- **Models**: gemma4-red:e4b
- **Judge**: gemini-2.5-flash

## Overall Scores


| Metric | gemma4-red:e4b |
|---|---|
| tool_format | 0.10 |
| actionable | 0.35 |
| grounding | 0.38 |
| action_relevance | 0.55 |
| reasoning_similarity | 0.45 |
| hallucination | 0.30 |
| degenerate | 0.00 |
| tok_s | 100.59 t/s |

## Overworld


| Metric | gemma4-red:e4b |
|---|---|
| tool_format | 0.20 |
| actionable | 0.40 |
| grounding | 0.50 |
| action_relevance | 0.75 |
| reasoning_similarity | 0.58 |
| hallucination | 0.17 |
| degenerate | 0.00 |
| tok_s | 119.35 t/s |

## Battle


| Metric | gemma4-red:e4b |
|---|---|
| tool_format | 0.00 |
| actionable | 0.30 |
| grounding | 0.25 |
| action_relevance | 0.25 |
| reasoning_similarity | 0.25 |
| hallucination | 0.50 |
| degenerate | 0.00 |
| tok_s | 81.82 t/s |

## Sample Responses

### Example 1: overworld @ ?
- **Teacher**: `[execute_custom_subagent] ANALYZE: The Sailor is about to send out his second Pikachu (Lv 21). Oddish (Lv 19) is active but could be paralyzed or outsped. Charmeleon (Lv 32) has a massive level advantage and Ground-type `
- **gemma4-red:e4b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`

### Example 4: overworld @ Route5
- **Teacher**: `[run_skill] ANALYZE: Player is at (12, 24) on Route 5. Objective: Return to Cerulean City. Situation: Need to move north past several ledges to reach the Cerulean City transition. PLAN: Action: Use navigate_to_coords_v5_`
- **gemma4-red:e4b**: `<|tool_call>call: run_skill   skill_id: navigate_to_coords_v5_ultimate   args: {"x": 10.0, "y": 0.0}   reasoning: "ANALYZE: Player is at (12, 26) on Route 5. Objective: Reach the northern exit warp to Cerulean City (10, `

### Example 7: overworld @ MtMoonPokecenter
- **Teacher**: `[run_skill] 1. Current State: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2. Obstacle Audit: [UP at (3, 3) is WALKABLE. Path to counter is clear]. 3. Decimal Audit: [Verified: No decimals in arguments]. 4. Action: [Con`
- **gemma4-red:e4b**: `1.  **Current State**: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2.  **Obstacle Audit**: [UP at (3, 3) is clear. UP at (3, 2) is the counter. A is the interaction tile]. 3.  **Decimal Audit**: [No decimals]. 4.  **Ac`

### Example 10: overworld @ PewterCity
- **Teacher**: `[run_skill] Situation: [11, 16, Facing Down, Pewter City, Overworld]. Status: [Charmander 37/37 HP Lv.13, 1 Potion, $245]. Objective: [Navigate to the Pewter Gym at (16, 17) to challenge Brock]. Stuck Check: [No]. Syntax`
- **gemma4-red:e4b**: `<|tool_call>call:run_skill   args: {"skill_id": "navigate_to_coords", "x": 16.0, "y": 17.0}   reasoning: "Situation: [11, 16, Facing Down, Pewter City, Overworld].\nStatus: [Charmander 37/37 HP Lv.13`

### Example 13: battle @ Route3
- **Teacher**: `[execute_custom_subagent] 1. Context: [11, 6, Route 3, Battle]. 2. Status: [Charmander 34/42 HP/Lv.15, $166]. 3. Objective: [Defeat the Youngster at (10, 6)]. 4. Navigation Check: [N/A - Battle]. 5. Technical Check: [Fin`
- **gemma4-red:e4b**: `<|tool_call>call:press_buttons   args: {"buttons": ["A", "A"], "reasoning": "1. Context: [11, 6, Route 3, Battle]. 2. Status: [Charmander 34/42 HP/Lv.15, $166]. 3. Objective: [Defeat the Youngster trainer (Pidgey)]. 4. N`

### Example 16: battle @ ViridianForest
- **Teacher**: `[execute_custom_subagent] Current State: Battle Mode. Position: (4, 31) during navigation. Goal: Navigate through Viridian Forest. Situation: Wild Lv. 3 Caterpie encountered. Charmander is Lv. 7 (16/24 HP). Strategic gui`
- **gemma4-red:e4b**: `<|tool_call>call: run_skill   args: {"skill_id": "navigate_to_coords_v2_simple", "x": 1.0, "y": 10.0}   reasoning": "ANALYZE: Player is at (2, 31) in Viridian Forest. Objective: Navigate north to Pewter City. Situation: `
