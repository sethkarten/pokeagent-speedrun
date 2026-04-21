# Pokemon Gemma4 agent eval

- **Dataset**: emerald_v3 (16 samples)
- **Prompt mode**: real
- **Models**: gemma4:26b, gemma4-emerald:26b
- **Judge**: gemini-2.5-flash

## Overall Scores


| Metric | gemma4:26b | gemma4-emerald:26b |
|---|---|---|
| tool_format | 0.00 | 1.00 |
| actionable | 0.19 | 1.00 |
| grounding | 0.12 | 0.80 |
| action_relevance | 0.16 | 0.53 |
| reasoning_similarity | 0.16 | 0.34 |
| hallucination | 0.06 | 0.50 |
| degenerate | 0.00 | 0.06 |
| tok_s | 162.48 t/s | 159.85 t/s |

## Overworld


| Metric | gemma4:26b | gemma4-emerald:26b |
|---|---|---|
| tool_format | 0.00 | 1.00 |
| actionable | 0.00 | 1.00 |
| grounding | 0.06 | 0.81 |
| action_relevance | 0.00 | 0.38 |
| reasoning_similarity | 0.00 | 0.19 |
| hallucination | 0.00 | 0.50 |
| degenerate | 0.00 | 0.12 |
| tok_s | 163.97 t/s | 162.69 t/s |

## Battle


| Metric | gemma4:26b | gemma4-emerald:26b |
|---|---|---|
| tool_format | 0.00 | 1.00 |
| actionable | 0.38 | 1.00 |
| grounding | 0.19 | 0.79 |
| action_relevance | 0.31 | 0.69 |
| reasoning_similarity | 0.31 | 0.50 |
| hallucination | 0.12 | 0.50 |
| degenerate | 0.00 | 0.00 |
| tok_s | 161.00 t/s | 157.01 t/s |

## Sample Responses

### Example 1: overworld @ RUSTBORO CITY MART
- **Teacher**: `[press_buttons] ANALYZE: Location: Rustboro City Mart. Objective: Restock supplies. Situation: The cashier said "Welcome!". PLAN: Action: Press A. Reason: Advance the dialogue to open the shop menu. Expected result: The `
- **gemma4:26b**: ``
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: Location: Rustboro City Mart. Objective: Restock supplies. Situation: The shop menu is open, but the cursor is currently on 'SELL'. I need to move the cursor to 'BUY' to purchase Potions and Para`

### Example 3: overworld @ OLDALE TOWN POKEMON CENTER 1F
- **Teacher**: `[press_buttons] ANALYZE: We are inside the Oldale Town Pokemon Center at (7, 8). We need to heal our Mudkip whose HP is 16/29. The nurse is at (7, 2), and the space to talk to her across the counter is at (7, 4). PLAN: A`
- **gemma4:26b**: ``
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: We are at (7, 8) in the Pokemon Center. The nurse is at (7, 2). We need to heal our Mudkip. PLAN: Action: Press UP x5. Reason: Move towards the nurse to initiate healing. Expected: Player moves t`

### Example 5: overworld @ RUSTBORO CITY CUTTERS HOUSE
- **Teacher**: `[press_buttons] ANALYZE: Location: Cutter's House (5, 8). Objective: Talk to the Cutter to get HM01. Situation: The Cutter is sitting at (7, 5). I can reach him by walking UP 3 times and RIGHT 1 time to (6, 5), then pres`
- **gemma4:26b**: ``
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: Location: Rustboro City (16, 39). Objective: Enter the Cutter's house to get HM01 Cut. Situation: The Cutter's house is directly left of the Pokemon Center, with its door at (9, 38). PLAN: Action`

### Example 7: overworld @ SLATEPORT CITY MART
- **Teacher**: `[press_buttons] ANALYZE: [game screen, location, objective, situation] Location: Slateport City Mart (3,7). Objective: Exit the Mart. Situation: Player is on the door mat (3,7). PLAN: Action: Press DOWN. Reason: Step on `
- **gemma4:26b**: ``
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: [game screen: Slateport Mart (3,7). Player is facing the clerk at (2,3). Objective: Sell TMs to buy 10 Potions.] SITUATION: The previous attempt to move UP to (3,3) failed because (3,4) was block`

### Example 9: battle @ DEWFORD TOWN GYM
- **Teacher**: `[press_buttons] ANALYZE: [game screen: Battle against Brawly. Marshtomp (Lv. 20) vs Makuhita (Lv. 19). Text: 'Foe MAKUHITA used VITAL THROW!'] Location: Dewford Gym. Objective: defeat_brawly_gym. Situation: Makuhita is a`
- **gemma4:26b**: ``
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: [game screen: Battle with Brawly. Marshtomp vs Makuhita. Text: 'Foe MAKUHITA used POUND!'] Location: Dewford Gym (4,4). Objective: defeat_brawly_gym. Situation: Makuhita is at roughly 40-50% HP. `

### Example 11: battle @ ROUTE 110
- **Teacher**: `[press_buttons] ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Poochyena is at critical HP. Zubat Lv 15 is healthy.] SITUATION: The objective is to finish the battle to gain EXP for Zubat. Poochyena's HP is low e`
- **gemma4:26b**: `Action executed: press_buttons   TOOLS:     - press_buttons       args: {"buttons": ["A", "B", "A", "B"], "reasoning": "ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Text \"It's sup\" (super effective) is visibl`
- **gemma4-emerald:26b**: `[press_buttons] ANALYZE: [game screen: Wild battle vs Poochyena Lv 12. Main battle menu visible. Zubat Lv 15 at 37/39 HP.] SITUATION: Continuing the Zubat grind. Poochyena is at low HP. Leech Life is super effective. OBJ`
