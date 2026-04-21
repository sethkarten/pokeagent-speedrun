# Pokemon Gemma4 agent eval

- **Dataset**: red_v1 (20 samples)
- **Prompt mode**: real
- **Models**: gemma4:26b, gemma4:31b, gemma4:e4b, gemma4:e2b, gemma4-red:e4b
- **Judge**: gemini-2.5-flash

## Overall Scores


| Metric | gemma4:26b | gemma4:31b | gemma4:e4b | gemma4:e2b | gemma4-red:e4b |
|---|---|---|---|---|---|
| tool_format | 0.05 | 0.05 | 0.10 | 0.00 | 0.00 |
| actionable | 0.25 | 0.35 | 0.45 | 0.15 | 0.00 |
| grounding | 0.23 | 0.31 | 0.40 | 0.09 | 0.40 |
| action_relevance | 0.33 | 0.42 | 0.33 | 0.03 | — |
| reasoning_similarity | 0.25 | 0.40 | 0.40 | 0.00 | — |
| hallucination | 0.05 | 0.00 | 0.25 | 0.20 | — |
| degenerate | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| tok_s | 162.41 t/s | 20.94 t/s | 189.26 t/s | 232.67 t/s | 0.00 t/s |

## Overworld


| Metric | gemma4:26b | gemma4:31b | gemma4:e4b | gemma4:e2b | gemma4-red:e4b |
|---|---|---|---|---|---|
| tool_format | 0.10 | 0.10 | 0.20 | 0.00 | 0.00 |
| actionable | 0.20 | 0.20 | 0.40 | 0.10 | 0.00 |
| grounding | 0.28 | 0.28 | 0.53 | 0.15 | 0.55 |
| action_relevance | 0.40 | 0.30 | 0.45 | 0.05 | — |
| reasoning_similarity | 0.25 | 0.35 | 0.40 | 0.00 | — |
| hallucination | 0.10 | 0.00 | 0.10 | 0.20 | — |
| degenerate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| tok_s | 163.92 t/s | 20.95 t/s | 189.97 t/s | 234.47 t/s | 0.00 t/s |

## Battle


| Metric | gemma4:26b | gemma4:31b | gemma4:e4b | gemma4:e2b | gemma4-red:e4b |
|---|---|---|---|---|---|
| tool_format | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| actionable | 0.30 | 0.50 | 0.50 | 0.20 | 0.00 |
| grounding | 0.17 | 0.35 | 0.28 | 0.03 | 0.25 |
| action_relevance | 0.25 | 0.55 | 0.20 | 0.00 | — |
| reasoning_similarity | 0.25 | 0.45 | 0.40 | 0.00 | — |
| hallucination | 0.00 | 0.00 | 0.40 | 0.20 | — |
| degenerate | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 |
| tok_s | 160.90 t/s | 20.94 t/s | 188.56 t/s | 230.88 t/s | 0.00 t/s |

## Sample Responses

### Example 1: overworld @ ?
- **Teacher**: `[execute_custom_subagent] ANALYZE: The Sailor is about to send out his second Pikachu (Lv 21). Oddish (Lv 19) is active but could be paralyzed or outsped. Charmeleon (Lv 32) has a massive level advantage and Ground-type `
- **gemma4:26b**: `ANALYZE: The experience gain dialogue ("ODDISH gained 369 EXP. Points!") is being cleared. After this, the game will either present the next Pokémon from the Sailor's team or return to the overworld if the battle is over`
- **gemma4:31b**: ``
- **gemma4:e4b**: `**THINKING:** ANALYZE: The previous actions (7063-7066) successfully defeated the Pikachu and cleared the associated experience gain dialogue. The battle is now in its final stages, either transitioning to the next oppon`
- **gemma4:e2b**: ``
- **gemma4-red:e4b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 4: overworld @ Route5
- **Teacher**: `[run_skill] ANALYZE: Player is at (12, 24) on Route 5. Objective: Return to Cerulean City. Situation: Need to move north past several ledges to reach the Cerulean City transition. PLAN: Action: Use navigate_to_coords_v5_`
- **gemma4:26b**: ``
- **gemma4:31b**: ````json {   "action": "run_skill",   "args": {     "skill_id": "navigate_to_coords_v5_ultimate",     "args": {       "x": 10.0,       "y": 0.0     },     "reasoning": "ANALYZE: Player is at (12, 24) on Route 5. Objective`
- **gemma4:e4b**: `**Plan:** 1.  **`
- **gemma4:e2b**: `Action executed: press_buttons   TOOLS:     - run_skill       args: {"skill_id": "navigate_to_coords_v5_ultimate", "args": {"x": 10.0, "y": 0.0}, "reasoning": "ANALYZE: The agent has successfully navigated to the exit po`
- **gemma4-red:e4b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 7: overworld @ MtMoonPokecenter
- **Teacher**: `[run_skill] 1. Current State: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2. Obstacle Audit: [UP at (3, 3) is WALKABLE. Path to counter is clear]. 3. Decimal Audit: [Verified: No decimals in arguments]. 4. Action: [Con`
- **gemma4:26b**: `1. **Current State**: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2. **Obstacle Audit**: [The path to the counter is clear. The player is moving towards the desk]. 3. **Decimal Audit**: [No decimals]. 4. **Action**: [C`
- **gemma4:31b**: ````json {   "reasoning": "1. Current State: [Coords (3, 4), MtMoonPokecenter, Over`
- **gemma4:e4b**: `1.  **Current State**: [Coords (3, 4), MtMoonPokecenter, Overworld]. 2.  **Obstacle Audit**: [The exit warp is available at the bottom of the building. The party needs to be fully healed before leaving to ensure readines`
- **gemma4:e2b**: `1. **Current State**: [Coords (3, 6), MtMoonPokecenter, Overworld]. 2. **Obstacle Audit**: [The exit path is clear. The goal is to leave the building and proceed to Route 3]. 3. **Decimal Audit**:`
- **gemma4-red:e4b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 10: overworld @ PewterCity
- **Teacher**: `[run_skill] Situation: [11, 16, Facing Down, Pewter City, Overworld]. Status: [Charmander 37/37 HP Lv.13, 1 Potion, $245]. Objective: [Navigate to the Pewter Gym at (16, 17) to challenge Brock]. Stuck Check: [No]. Syntax`
- **gemma4:26b**: ``
- **gemma4:31b**: ````json {   "reasoning": "Situation: [11, 16, Facing ?, Pewter City, Overworld].\nStatus: [Charmander 37/37 HP Lv.13, 1 Potion, $245].\nObjective: [`
- **gemma4:e4b**: `**Situation**: (11, 16), Facing Down, PewterCity, Overworld. **Status**: Charmander 37/37 HP Lv.13, 1 Potion, $245. **Objective**: Navigate to the Pewter Gym entrance at (16, 17) and challenge Brock. **Stuck Check**: No.`
- **gemma4:e2b**: ``
- **gemma4-red:e4b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 13: battle @ Route3
- **Teacher**: `[execute_custom_subagent] 1. Context: [11, 6, Route 3, Battle]. 2. Status: [Charmander 34/42 HP/Lv.15, $166]. 3. Objective: [Defeat the Youngster at (10, 6)]. 4. Navigation Check: [N/A - Battle]. 5. Technical Check: [Fin`
- **gemma4:26b**: ``
- **gemma4:31b**: ````json {   "reasoning": "1. Context: [11, 6, Route 3, Battle].\n2. Status: [Charmander 34/42 HP/Lv.15, $166].\n3.`
- **gemma4:e4b**: `**1. Context**: [11, 6, Route 3, Battle]. **2. Status**: [Charmander 34/42 HP/Lv.15, $166]. **3. Objective**: [Defeat the Youngster at (10, 6) and continue traversing Route 3]. **4. Navigation Check**: [N/A - In Battle].`
- **gemma4:e2b**: ``
- **gemma4-red:e4b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 16: battle @ ViridianForest
- **Teacher**: `[execute_custom_subagent] Current State: Battle Mode. Position: (4, 31) during navigation. Goal: Navigate through Viridian Forest. Situation: Wild Lv. 3 Caterpie encountered. Charmander is Lv. 7 (16/24 HP). Strategic gui`
- **gemma4:26b**: `Current State: Battle. Position: (2, 31). Goal: Resolve battle with Caterpie and continue navigating Viridian Forest. Action: execute_custom_subagent to handle the battle.`
- **gemma4:31b**: ````json {   "reasoning": "Current State: Battle. Position: (4, 31) (approx). Goal: Resolve wild encounter with Caterpie. Situation: A wild Lv. 3 Caterpie has appeared. Charmander is Lv. 7 and healthy enough to win quickl`
- **gemma4:e4b**: `Current State: Battle Mode. Position: (2, 31) in Viridian Forest. Goal: Defeat`
- **gemma4:e2b**: `Action executed: run_skill   TOOLS:     - run_skill       args: {"skill_id": "navigate_to_coords_v2_simple", "args": {"x": 2.0, "y": 0.0}, "reasoning": "ANALYZE: Player is at (2, 19) in Viridian Forest. Objective: Naviga`
- **gemma4-red:e4b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`
