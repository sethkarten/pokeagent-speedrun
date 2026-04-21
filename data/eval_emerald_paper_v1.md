# Pokemon Gemma4 agent eval

- **Dataset**: emerald_v3 (20 samples)
- **Prompt mode**: real
- **Models**: gemma4:26b, gemma4:31b, gemma4:e4b, gemma4:e2b, gemma4-emerald:26b, gemma4-emerald:e4b, gemma4-emerald:e2b, gemma4-emerald-grpo:26b
- **Judge**: gemini-2.5-flash

## Overall Scores


| Metric | gemma4:26b | gemma4:31b | gemma4:e4b | gemma4:e2b | gemma4-emerald:26b | gemma4-emerald:e4b | gemma4-emerald:e2b | gemma4-emerald-grpo:26b |
|---|---|---|---|---|---|---|---|---|
| tool_format | 0.00 | 0.00 | 0.00 | 0.00 | 0.95 | 0.00 | 0.00 | 0.00 |
| actionable | 0.20 | 0.55 | 0.55 | 0.40 | 0.95 | 0.40 | 0.20 | 0.00 |
| grounding | 0.15 | 0.39 | 0.54 | 0.28 | 0.72 | 0.52 | 0.33 | 0.40 |
| action_relevance | 0.15 | 0.51 | 0.38 | 0.15 | 0.50 | 0.35 | 0.07 | — |
| reasoning_similarity | 0.15 | 0.45 | 0.26 | 0.10 | 0.35 | 0.28 | 0.05 | — |
| hallucination | 0.05 | 0.05 | 0.30 | 0.05 | 0.55 | 0.30 | 0.25 | — |
| degenerate | 0.00 | 0.00 | 0.00 | 0.05 | 0.05 | 0.10 | 0.00 | 0.00 |
| tok_s | 153.92 t/s | 18.12 t/s | 188.24 t/s | 221.31 t/s | 165.86 t/s | 188.21 t/s | 220.58 t/s | 0.00 t/s |

## Overworld


| Metric | gemma4:26b | gemma4:31b | gemma4:e4b | gemma4:e2b | gemma4-emerald:26b | gemma4-emerald:e4b | gemma4-emerald:e2b | gemma4-emerald-grpo:26b |
|---|---|---|---|---|---|---|---|---|
| tool_format | 0.00 | 0.00 | 0.00 | 0.00 | 0.90 | 0.00 | 0.00 | 0.00 |
| actionable | 0.10 | 0.50 | 0.40 | 0.50 | 0.90 | 0.60 | 0.30 | 0.00 |
| grounding | 0.15 | 0.40 | 0.56 | 0.36 | 0.75 | 0.57 | 0.47 | 0.55 |
| action_relevance | 0.10 | 0.45 | 0.35 | 0.20 | 0.40 | 0.25 | 0.15 | — |
| reasoning_similarity | 0.10 | 0.35 | 0.20 | 0.10 | 0.20 | 0.25 | 0.00 | — |
| hallucination | 0.00 | 0.10 | 0.30 | 0.00 | 0.50 | 0.30 | 0.30 | — |
| degenerate | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 0.00 |
| tok_s | 146.11 t/s | 18.31 t/s | 189.31 t/s | 233.66 t/s | 168.38 t/s | 189.02 t/s | 232.11 t/s | 0.00 t/s |

## Battle


| Metric | gemma4:26b | gemma4:31b | gemma4:e4b | gemma4:e2b | gemma4-emerald:26b | gemma4-emerald:e4b | gemma4-emerald:e2b | gemma4-emerald-grpo:26b |
|---|---|---|---|---|---|---|---|---|
| tool_format | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| actionable | 0.30 | 0.60 | 0.70 | 0.30 | 1.00 | 0.20 | 0.10 | 0.00 |
| grounding | 0.15 | 0.38 | 0.51 | 0.20 | 0.70 | 0.46 | 0.18 | 0.25 |
| action_relevance | 0.20 | 0.57 | 0.40 | 0.10 | 0.60 | 0.45 | 0.00 | — |
| reasoning_similarity | 0.20 | 0.55 | 0.32 | 0.10 | 0.50 | 0.30 | 0.10 | — |
| hallucination | 0.10 | 0.00 | 0.30 | 0.10 | 0.60 | 0.30 | 0.20 | — |
| degenerate | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.20 | 0.00 | 0.00 |
| tok_s | 161.73 t/s | 17.93 t/s | 187.17 t/s | 208.95 t/s | 163.33 t/s | 187.39 t/s | 209.05 t/s | 0.00 t/s |

## Sample Responses

### Example 1: overworld @ RUSTBORO CITY MART
- **Teacher**: `[press_buttons] ANALYZE: Location: Rustboro City Mart. Objective: Restock supplies. Situation: The cashier said "Welcome!". PLAN: Action: Press A. Reason: Advance the dialogue to open the shop menu. Expected result: The `
- **gemma4:26b**: ``
- **gemma4:31b**: ````json {   "action": "press_buttons",   "args": {     "buttons": [       "A"     ],     "reasoning": "ANALYZE: Location: Rustboro City Mart. Objective: Restock supplies. Situation: The cashier has said \"Welcome!\". I n`
- **gemma4:e4b**: ``
- **gemma4:e2b**: `Action executed: press_buttons   TOOLS:     - run_skill       args: {"skill_id": "exit_building", "args": {}}       result: "{\n  \"success\": true, \"skill_id\": \"exit_building\",`
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: Location: Rustboro City Mart. Objective: Restock supplies. Situation: The shop menu is open, but the cursor is currently on 'SELL'. I need to move the cursor to 'BUY' to purchase Potions and Para`
- **gemma4-emerald:e4b**: `ANALYZE: Location: Rustboro City Mart. Objective: Restock supplies. Situation: The shop menu is open, and the cursor is on "BUY". PLAN: Action: Press A to select "BUY", then DOWN to scroll to "POTIONS", and A to select t`
- **gemma4-emerald:e2b**: ````json {   "tool_name": "execute_custom_subagent",   "subagent_id": "shop_transaction_handler",   "instructions": "1. Select the items you wish to purchase from the list. 2. Confirm the quantities. 3. Click the 'PAY' bu`
- **gemma4-emerald-grpo:26b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 4: overworld @ Map_11_01
- **Teacher**: `[press_buttons] ANALYZE: [game screen: Player inside Pretty Petal Flower Shop at (2, 8). Objective: Obtain Wailmer Pail.] Situation: Need to talk to the NPCs inside the shop. The nearest NPC is to the north-east. PLAN: [`
- **gemma4:26b**: ``
- **gemma4:31b**: ````json {   "action": "press_buttons",   "args": {     "buttons": [`
- **gemma4:e4b**: ````json {   "action": "press_buttons",   "args": {     "buttons": [       "UP",       "UP",       "UP",       "UP",       "UP",       "UP",       "UP",       "UP",       "UP",       "UP"     ],     "reasoning": "ANALYZE:`
- **gemma4:e2b**: `Action executed: execute_custom_subagent   TOOLS:     - execute_custom_subagent       args: {"reasoning": "AN`
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: [game screen: Player at (2, 8) in Petalburg Woods. North path blocked by trees at Y=7. Objective: Reach Rustboro City.] Situation: The path to the exit (14, 4) requires bypassing the tree line. M`
- **gemma4-emerald:e4b**: `ANALYZE: [game screen: Player at (2, 8) on Route 104, just exited Petalburg Woods. Pretty Petal Flower Shop is to the left.] Situation: Need to visit the shop for the Wailmer Pail. The shop is at (5, 17). The current pos`
- **gemma4-emerald:e2b**: `### PLAN  1.  **Navigate to Route 105**: Use the `route_navigator` subagent to navigate from Route 104 to Route 105. This is a long route, and the subagent is designed to handle it efficiently. 2.  **Navigate to Route 10`
- **gemma4-emerald-grpo:26b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 7: overworld @ SLATEPORT CITY MART
- **Teacher**: `[press_buttons] ANALYZE: [game screen, location, objective, situation] Location: Slateport City Mart (3,7). Objective: Exit the Mart. Situation: Player is on the door mat (3,7). PLAN: Action: Press DOWN. Reason: Step on `
- **gemma4:26b**: ``
- **gemma4:31b**: ``
- **gemma4:e4b**: ``
- **gemma4:e2b**: `Based on the strategic guidance and the current game state, the immediate objective`
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: [game screen: Slateport Mart (3,7). Player is facing the clerk at (2,3). Objective: Sell TMs to buy 10 Potions.] SITUATION: The previous attempt to move UP to (3,3) failed because (3,4) was block`
- **gemma4-emerald:e4b**: ``
- **gemma4-emerald:e2b**: ``
- **gemma4-emerald-grpo:26b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 10: overworld @ SLATEPORT CITY OCEANIC MUSEUM 1F
- **Teacher**: `[complete_direct_objective] ANALYZE: Objective was to enter the Oceanic Museum. The location is now SLATEPORT CITY OCEANIC MUSEUM 1F at (9, 8). Evidence: Game text and screen show we are inside the museum. PLAN: Objectiv`
- **gemma4:26b**: ``
- **gemma4:31b**: ``
- **gemma4:e4b**: `**THINKING:** ANALYZE: Location: Slateport City Oceanic Museum 1F. Objective: Enter the museum and pay the fee. Situation: I have successfully navigated to the first floor of the museum. The immediate goal is to find the`
- **gemma4:e2b**: ``
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: Location: Slateport City (30, 26). Objective: Enter Oceanic Museum. Situation: We are standing right in front of the museum door. PLAN: Action: Press UP 3 times. Reason: Step into the museum buil`
- **gemma4-emerald:e4b**: ``
- **gemma4-emerald:e2b**: `### 🎯 Objective: Enter the Oceanic Museum and find Capt. Stern.  ### 📝 Action Plan: 1.  **Interact with Grunts:** Stand in front of the grunts and press `A` repeatedly to initiate dialogue. This will either clear the lin`
- **gemma4-emerald-grpo:26b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 13: battle @ ROUTE 110
- **Teacher**: `[press_buttons] ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Poochyena is at critical HP. Zubat Lv 15 is healthy.] SITUATION: The objective is to finish the battle to gain EXP for Zubat. Poochyena's HP is low e`
- **gemma4:26b**: `Action executed: press_buttons   TOOLS:     - press_buttons       args: {"buttons": ["A", "B", "A", "B"], "reasoning": "ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Text \"It's sup\" (super effective) is visibl`
- **gemma4:31b**: ``
- **gemma4:e4b**: `**THINKING:** ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Main battle menu visible. Zubat Lv 15 at 37/39 HP.] SITUATION: The wild battle is ongoing, and the objective is to continue grinding EXP for Zubat. The`
- **gemma4:e2b**: ``
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Main battle menu visible. Zubat Lv 15 at 37/39 HP.] SITUATION: Continuing the Zubat grind. Poochyena is at low HP. Leech Life is super effective. OBJ`
- **gemma4-emerald:e4b**: `ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Main battle menu visible. Zubat Lv 15 at 37/39 HP.] SITUATION: Continuing the Zubat grind. Poochyena is at low HP. Leech Life is super effective. OBJECTIVE: Defeat P`
- **gemma4-emerald:e2b**: `## 📝 Agent Performance Review & Improvement Plan  `
- **gemma4-emerald-grpo:26b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`

### Example 16: battle @ ROUTE 102
- **Teacher**: `[execute_custom_subagent] ANALYZE: The battle transition animation is playing, and the game state indicates we are in Battle Mode against Trainer CASEY. Objective: Defeat the trainer's Pokemon to gain experience and mone`
- **gemma4:26b**: ````json {   "action": "execute_custom_subagent",   "args": {     "reasoning": "ANALYZE: The battle with Casey has just been initiated. The pre-battle dialogue has been cleared, and we are now entering the battle screen. `
- **gemma4:31b**: ````json {   "action": "execute_custom_subagent",   "args": {     "reasoning": "ANALYZE: We have entered a trainer battle with Casey on Route 102. Mudkip is at full health and ready to fight. The battle is currently in th`
- **gemma4:e4b**: `**THINKING:** ANALYZE: We are currently in a trainer battle with a Youngster on Route 102. The objective is to defeat the opponent's Pokemon to gain experience and money. The battle is active (phase_211). PLAN: Action: E`
- **gemma4:e2b**: `Action executed: press_buttons   TOOLS:     - press_buttons       args: {"buttons": ["LEFT", "LEFT", "LEFT", "LEFT", "LEFT"], "reasoning": "ANALYZE: We have just completed a trainer battle on Route 102 and need to contin`
- **gemma4-emerald:26b**: `[execute_custom_subagent] ANALYZE: The player has encountered a wild Pokemon (looks like a Poochyena) on Route 102. The battle transition is complete and the main battle menu is visible. Location: Route 102. Objective: D`
- **gemma4-emerald:e4b**: `ANALYZE: The battle has started with a trainer on Route 102. The opponent is a trainer with a Pokemon that is currently unknown. The battle screen is in the initialization phase. Objective: Win the battle. Situation: Nee`
- **gemma4-emerald:e2b**: ````json {   "tool_name": "pokemon_catcher",   "params": {     "target_pokemon": "rock_type",     "pokemon_type": "water"   } } ````
- **gemma4-emerald-grpo:26b**: `ERROR: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate`
