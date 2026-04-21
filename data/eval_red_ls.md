# Pokemon Gemma4 agent eval

- **Dataset**: red_v1 (20 samples)
- **Prompt mode**: real
- **Models**: gemma4-red:31b, gemma4-red-grpo:26b
- **Judge**: gemini-2.5-flash

## Overall Scores


| Metric | gemma4-red:31b | gemma4-red-grpo:26b |
|---|---|---|
| tool_format | 0.50 | 0.50 |
| actionable | 0.50 | 0.50 |
| grounding | 0.44 | 0.44 |
| action_relevance | 0.75 | 0.65 |
| reasoning_similarity | 0.65 | 0.50 |
| hallucination | 0.30 | 0.40 |
| degenerate | 0.00 | 0.05 |
| tok_s | 26.23 t/s | 88.95 t/s |

## Overworld


| Metric | gemma4-red:31b | gemma4-red-grpo:26b |
|---|---|---|
| tool_format | 0.60 | 0.60 |
| actionable | 0.60 | 0.60 |
| grounding | 0.57 | 0.62 |
| action_relevance | 0.92 | 0.75 |
| reasoning_similarity | 0.58 | 0.42 |
| hallucination | 0.00 | 0.17 |
| degenerate | 0.00 | 0.00 |
| tok_s | 31.57 t/s | 107.07 t/s |

## Battle


| Metric | gemma4-red:31b | gemma4-red-grpo:26b |
|---|---|---|
| tool_format | 0.40 | 0.40 |
| actionable | 0.40 | 0.40 |
| grounding | 0.30 | 0.25 |
| action_relevance | 0.50 | 0.50 |
| reasoning_similarity | 0.75 | 0.62 |
| hallucination | 0.75 | 0.75 |
| degenerate | 0.00 | 0.10 |
| tok_s | 20.88 t/s | 70.83 t/s |

## Sample Responses

### Example 1: overworld @ ?
- **Teacher**: `[execute_custom_subagent] ANALYZE: The Sailor is about to send out his second Pikachu (Lv 21). Oddish (Lv 19) is active but could be paralyzed or outsped. Charmeleon (Lv 32) has a massive level advantage and Ground-type `
- **gemma4-red:31b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`
- **gemma4-red-grpo:26b**: `ERROR: 400 Client Error: Bad Request for url: http://127.0.0.1:11435/v1/chat/completions`

### Example 4: overworld @ Route5
- **Teacher**: `[run_skill] ANALYZE: Player is at (12, 24) on Route 5. Objective: Return to Cerulean City. Situation: Need to move north past several ledges to reach the Cerulean City transition. PLAN: Action: Use navigate_to_coords_v5_`
- **gemma4-red:31b**: `[run_skill] ANALYZE: Player is at (12, 24) on Route 5. Objective: Return to Cerulean City. Situation: Need to navigate the long path north to the map transition. PLAN: Action: Use navigate_to_coords_v5_ultimate to reach `
- **gemma4-red-grpo:26b**: `[run_skill] ANALYZE: Player is at (12, 24) on Route 5. Objective: Return to Cerulean City. Situation: The player is in a narrow corridor. The path north to the Underground Path entrance at (10, 0) is blocked by a sign at`

### Example 7: overworld @ MtMoonPokecenter
- **Teacher**: `[run_skill] 1. Current State: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2. Obstacle Audit: [UP at (3, 3) is WALKABLE. Path to counter is clear]. 3. Decimal Audit: [Verified: No decimals in arguments]. 4. Action: [Con`
- **gemma4-red:31b**: `[run_skill] 1. Current State: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2. Obstacle Audit: [UP at (3, 3) is blocked by an NPC. RIGHT at (4, 4) and UP at (4, 3) are WALKABLE]. 3. Decimal Audit: [Verified all numbers a`
- **gemma4-red-grpo:26b**: `[run_code] 1. Current State: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2. Obstacle Audit: [UP at (3, 3) is WALKABLE. (3, 2) is the Counter (C) where Nurse Joy is located]. 3. Decimal Audit: [No decimals in tool call]`

### Example 10: overworld @ PewterCity
- **Teacher**: `[run_skill] Situation: [11, 16, Facing Down, Pewter City, Overworld]. Status: [Charmander 37/37 HP Lv.13, 1 Potion, $245]. Objective: [Navigate to the Pewter Gym at (16, 17) to challenge Brock]. Stuck Check: [No]. Syntax`
- **gemma4-red:31b**: `[run_skill] Situation: [11, 16, Facing Down, Pewter City, Overworld].  Status: [Charmander 37/37 HP Lv.13, 1 Potion, $245].  Objective: [Navigate to the Pewter Gym at (16, 17) to challenge Brock].  Stuck Check: [No].  Sy`
- **gemma4-red-grpo:26b**: `[run_skill] Situation: [11, 16, Facing Up, PewterCity, Overworld]. Status: [Charmander 37/37 HP Lv.13, 1 Potion, $245]. Objective: [Navigate to the Pewter Gym entrance at (16, 17) to challenge Brock]. Stuck Check: [No]. `

### Example 13: battle @ Route3
- **Teacher**: `[execute_custom_subagent] 1. Context: [11, 6, Route 3, Battle]. 2. Status: [Charmander 34/42 HP/Lv.15, $166]. 3. Objective: [Defeat the Youngster at (10, 6)]. 4. Navigation Check: [N/A - Battle]. 5. Technical Check: [Fin`
- **gemma4-red:31b**: `[execute_custom_subagent] 1. Context: [11, 6, Route 3, Battle].  2. Status: [Charmander 34/42 HP/Lv.15, $166].  3. Objective: [Defeat the Youngster at (10, 6)].  4. Navigation Check: [N/A - Battle transition].  5. Techni`
- **gemma4-red-grpo:26b**: `[run_skill] 1. Context: [11, 6, Route 3, Battle]. 2. Status: [Charmander 34/42 HP/Lv.15, $166]. 3. Objective: [Defeat the Youngster at (10, 6)]. 4. Navigation Check: [Battle triggered, navigate_to_coords is not applicabl`

### Example 16: battle @ ViridianForest
- **Teacher**: `[execute_custom_subagent] Current State: Battle Mode. Position: (4, 31) during navigation. Goal: Navigate through Viridian Forest. Situation: Wild Lv. 3 Caterpie encountered. Charmander is Lv. 7 (16/24 HP). Strategic gui`
- **gemma4-red:31b**: `[execute_custom_subagent] Current State: Battle Mode. Position: (2, 31) during navigation. Goal: Navigate through Viridian Forest. Situation: Wild Lv. 3 Caterpie encountered. Charmander is Lv. 7 (16/24 HP). Strategic gui`
- **gemma4-red-grpo:26b**: `[run_skill] Current State: Battle Mode. Position: (17, 45). Goal: Defeat the wild Lv. 3 Caterpie to gain experience and resources. Situation: Battle encounter triggered in the grass. Charmander is Lv. 7 and healthy (24/2`
