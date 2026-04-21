# Full Model Comparison: Base vs SFT across Gemma4 sizes

**Date**: 2026-04-12
**Dataset**: emerald_v3 (20 samples)
**Models**: gemma4:26b, gemma4-emerald:26b, gemma4:e4b, gemma4-emerald:e4b, gemma4-emerald:e2b

## Overall Scores

| Metric | Base 26b | SFT 26b | Base e4b | SFT e4b | SFT e2b |
|--------|------|------|------|------|------|
| tool_format | 0.50 | 0.80 | 0.00 | 0.00 | 0.00 |
| grounding | 0.88 | 1.00 | 1.00 | 0.93 | 0.75 |
| action_relevance | 0.53 | 0.80 | 0.45 | 0.10 | 0.00 |
| reasoning_similarity | 0.34 | 0.64 | 0.42 | 0.38 | 0.00 |
| hallucination | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 |
| tok_s | 178.69 t/s | 178.81 t/s | 199.96 t/s | 199.31 t/s | 0.00 t/s |

### Overworld

| Metric | Base 26b | SFT 26b | Base e4b | SFT e4b | SFT e2b |
|--------|------|------|------|------|------|
| tool_format | 0.60 | 0.70 | 0.00 | 0.00 | 0.00 |
| grounding | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| hallucination | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| action_relevance | 0.65 | 0.70 | 0.45 | 0.10 | 0.00 |

### Battle

| Metric | Base 26b | SFT 26b | Base e4b | SFT e4b | SFT e2b |
|--------|------|------|------|------|------|
| tool_format | 0.40 | 0.90 | 0.00 | 0.00 | 0.00 |
| grounding | 0.85 | 1.00 | 1.00 | 0.85 | 0.50 |
| hallucination | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| action_relevance | 0.40 | 0.90 | 0.45 | 0.10 | 0.00 |

## Sample Responses

### Example 1: overworld @ SLATEPORT CITY OCEANIC MUSEUM 2F

**Teacher**: [press_buttons] ANALYZE: [game screen: Oceanic Museum 2F. Location: (6, 2). Objective: Deliver Devon Goods to Capt. Stern.] Situation: I have reached the 2nd floor. Capt. Stern is at (13, 6). The path...

**Base 26b**: [press_buttons] ANALYZE: The player is currently standing in the middle of the Oceanic Museum 2F in Slateport City. There are several fossil exhibits on display, a computer terminal, and an NPC locate...

**SFT 26b**: [press_buttons] ANALYZE: Player is at (19, 19) in the Oceanic Museum 2F. There is an NPC at (19, 20) facing UP. PLAN: Press UP to interact with the NPC. Expected: Dialogue starts....

**Base e4b**: [tool_name] ANALYZE: The player character is in the Slateport City Oceanic Museum 2F. The area contains several exhibits, including two platforms with pairs of objects and a large model of a ship. To ...

**SFT e4b**: {"button": "A", "count": 1}...

**SFT e2b**: ERROR: 500 Server Error: Internal Server Error for url: http://127.0.0.1:11434/api/generate...

---

### Example 6: overworld @ PETALBURG WOODS

**Teacher**: [press_buttons] ANALYZE: Location: Petalburg Woods (16,18). Objective: Travel south to Mr. Briney's cottage on Route 104. Situation: I have calculated the remaining path to the southern exit at (16,39...

**Base 26b**: [press_buttons] ANALYZE: The screenshot shows the player character (with brown hair) and an NPC (with white hair) in Petalburg Woods. The player is currently in a clearing surrounded by trees and bush...

**SFT 26b**:           ******** MIND MAP RESULTS QUERY LOG (turn 12) ********  📋 RESULTS FROM PREVIOUS TURN: ▶️ Executed run_skill   ✅ run_skill returned: {'success': True, 'skill_id': 'navigate_to_coords', 'resul...

**Base e4b**: replan_objectives ANALYZE: I am in Petalburg Woods. The environment is dense with bushes and paths. There are two visible characters (one in the upper center, one in the lower center). I need to deter...

**SFT e4b**: Tool: press_buttons ANALYZE: The game state is overworld, location is Petalburg Woods. I am in a clearing with several trainers and people around me. PLAN: Action: Press 'right' repeatedly. Reason: To...

**SFT e2b**: ERROR: 500 Server Error: Internal Server Error for url: http://127.0.0.1:11434/api/generate...

---

### Example 11: battle @ RUSTURF TUNNEL ALT

**Teacher**: [execute_custom_subagent] ANALYZE: Location: Rusturf Tunnel. Objective: Retrieve Devon Goods. Situation: The battle with the Team Aqua Grunt has just officially begun and he sent out a Lv. 11 Poochyen...

**Base 26b**: [press_buttons] ANALYZE: A battle has just started in Rusturf Tunnel Alt. The player has sent out Marshtomp to face a level 11 Poochyena. The text box "Go! MARSHTOMP!" is currently displayed, indicati...

**SFT 26b**: [press_buttons] ANALYZE: The screen shows the move selection menu for Marshtomp in a battle against a Lv 11 Poochyena. The cursor is currently on 'TACKLE'. PLAN: Press RIGHT, DOWN, and A to select and...

**Base e4b**: [tool_name] ANALYZE: The player is engaged in a Pokémon battle in the RUSTURF TUNNEL ALT. The player's Pokémon is Poochyena (Lv 11), and the selected move is MARSHTOMP. The battle is currently in the ...

**SFT e4b**: tool_name: press_buttons ANALYZE: The screenshot shows a battle screen. The opponent is a wild Pokemon. My Pokemon is Poochyena (Lv 11). The battle menu is visible with FIGHT selected. Poochyena's HP ...

**SFT e2b**: ERROR: 500 Server Error: Internal Server Error for url: http://127.0.0.1:11434/api/generate...

---

### Example 16: battle @ ROUTE 101

**Teacher**: [press_buttons] ANALYZE: We are in a battle with a wild Wurmple. We want to catch it. We are at the action selection menu. PLAN: Action: Press RIGHT, A. Reason: Move the cursor to BAG and select it. E...

**Base 26b**: press_buttons ANALYZE: The player's Wurmple is level 3, while the opponent's Mudkip is level 9. The level difference is too large to win this battle effectively, and attempting to fight will likely re...

**SFT 26b**: [press_buttons] ANALYZE: The battle menu is open with the cursor on FIGHT. Mudkip has 25/29 HP. The opponent is a Lv 3 Wurmple. PLAN: Press A to select FIGHT, then A to select the first move (Tackle),...

**Base e4b**: [tool_name] press_buttons ANALYZE: The player's Pokémon is Mudkip (Level 9) and the opponent is Wurmple (Level 3). Mudkip is significantly stronger than the opponent. The goal is to defeat the opponen...

**SFT e4b**: ANALYZE: The battle turn menu is visible. The opponent is a red/brown Pokemon (Nincada). My Pokemon is a blue/white Pokemon (Whismur). The options are FIGHT, BAG, POKEMON, RUN. PLAN: I will select FIG...

**SFT e2b**: ERROR: 500 Server Error: Internal Server Error for url: http://127.0.0.1:11434/api/generate...

---

