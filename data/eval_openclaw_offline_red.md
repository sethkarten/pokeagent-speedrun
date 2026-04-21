# Pokemon Gemma4 agent eval

- **Dataset**: red_v1 (20 samples)
- **Prompt mode**: real
- **Models**: gemma4-openclaw-offline:26b
- **Judge**: gemini-2.5-flash

## Overall Scores


| Metric | gemma4-openclaw-offline:26b |
|---|---|
| tool_format | 0.50 |
| actionable | 0.50 |
| grounding | 0.40 |
| action_relevance | 0.55 |
| reasoning_similarity | 0.30 |
| hallucination | 0.60 |
| degenerate | 0.05 |
| tok_s | 96.61 t/s |

## Overworld


| Metric | gemma4-openclaw-offline:26b |
|---|---|
| tool_format | 0.60 |
| actionable | 0.60 |
| grounding | 0.57 |
| action_relevance | 0.75 |
| reasoning_similarity | 0.33 |
| hallucination | 0.50 |
| degenerate | 0.00 |
| tok_s | 115.15 t/s |

## Battle


| Metric | gemma4-openclaw-offline:26b |
|---|---|
| tool_format | 0.40 |
| actionable | 0.40 |
| grounding | 0.23 |
| action_relevance | 0.25 |
| reasoning_similarity | 0.25 |
| hallucination | 0.75 |
| degenerate | 0.10 |
| tok_s | 78.07 t/s |

## Sample Responses

### Example 1: overworld @ ?
- **Teacher**: `[execute_custom_subagent] ANALYZE: The Sailor is about to send out his second Pikachu (Lv 21). Oddish (Lv 19) is active but could be paralyzed or outsped. Charmeleon (Lv 32) has a massive level advantage and Ground-type `
- **gemma4-openclaw-offline:26b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`

### Example 4: overworld @ Route5
- **Teacher**: `[run_skill] ANALYZE: Player is at (12, 24) on Route 5. Objective: Return to Cerulean City. Situation: Need to move north past several ledges to reach the Cerulean City transition. PLAN: Action: Use navigate_to_coords_v5_`
- **gemma4-openclaw-offline:26b**: `[press_buttons] ANALYZE: Player is at (12, 24) on Route 5, standing on a grass tile. Objective: Return to Cerulean City. Situation: The path north is clear but the map transition is at (10, 0). PLAN: Action: Press UP. Re`

### Example 7: overworld @ MtMoonPokecenter
- **Teacher**: `[run_skill] 1. Current State: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2. Obstacle Audit: [UP at (3, 3) is WALKABLE. Path to counter is clear]. 3. Decimal Audit: [Verified: No decimals in arguments]. 4. Action: [Con`
- **gemma4-openclaw-offline:26b**: `[press_buttons] 1. Current State: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2. Obstacle Audit: [UP at (3, 3) is WALKABLE. Path to counter is clear]. 3. Decimal Audit: [No decimals]. 4. Action: [Press UP to reach (3, `

### Example 10: overworld @ PewterCity
- **Teacher**: `[run_skill] Situation: [11, 16, Facing Down, Pewter City, Overworld]. Status: [Charmander 37/37 HP Lv.13, 1 Potion, $245]. Objective: [Navigate to the Pewter Gym at (16, 17) to challenge Brock]. Stuck Check: [No]. Syntax`
- **gemma4-openclaw-offline:26b**: `[press_buttons] Situation: [11, 16, Facing Up, Pewter City, Overworld]. Status: [Charmander 37/37 HP Lv.13, 1 Potion, $245]. Objective: [Navigate to the Pewter Gym at (16, 17) to challenge Brock]. Stuck Check: [No]. Synt`

### Example 13: battle @ Route3
- **Teacher**: `[execute_custom_subagent] 1. Context: [11, 6, Route 3, Battle]. 2. Status: [Charmander 34/42 HP/Lv.15, $166]. 3. Objective: [Defeat the Youngster at (10, 6)]. 4. Navigation Check: [N/A - Battle]. 5. Technical Check: [Fin`
- **gemma4-openclaw-offline:26b**: `[press_buttons] 1. Context: [11, 6, Route 3, Dialog]. 2. Status: [Charmander 34/42 HP/Lv.15, $166]. 3. Objective: [Defeat the Youngster at (10, 6)]. 4. Navigation Check: [Adjacent to trainer, dialogue triggered]. 5. Tech`

### Example 16: battle @ ViridianForest
- **Teacher**: `[execute_custom_subagent] Current State: Battle Mode. Position: (4, 31) during navigation. Goal: Navigate through Viridian Forest. Situation: Wild Lv. 3 Caterpie encountered. Charmander is Lv. 7 (16/24 HP). Strategic gui`
- **gemma4-openclaw-offline:26b**: `[press_buttons] ANALYZE: Player is at (17, 40) in Viridian Forest. Objective: Navigate north to reach Pewter City. Situation: Just entered a grass patch. PLAN: Action: Move UP 5 times. Reason: Advance through the grass t`
