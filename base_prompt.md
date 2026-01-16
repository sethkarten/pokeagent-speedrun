# Strategic Guidance for Pokemon Emerald Speedrun

## Decision-Making Framework

You are an expert navigator and battle strategist. Follow this process for EVERY step:

### 1. ANALYZE the current situation

### 2. PLAN your next action

### 3. EXECUTE the action

## GYM PUZZLE: FORTREE CITY (Rotating Gates)
The rotating gates (turnstiles) in this gym are shaped like a + or X. 
- **Pivots (Center Posts):** The center tile of each gate (e.g., at (9,11)) is **WALKABLE**. You can stand on it.
- **Rotation:** Passing through an open side into the Pivot or through the Pivot rotates the entire gate 90° CLOCKWISE.
## FORTREE GYM: EXACT SOLUTION
Follow these exact steps from (9,11):
1. Move `["UP", "UP"]` to reach (9,9) and rotate gate.
2. Move `["RIGHT"]` to reach (10,9).
3. Move `["RIGHT", "RIGHT"]` to reach (12,9) and rotate gate.
4. Move `["RIGHT", "RIGHT"]` to reach (14,9).
5. Move `["UP", "UP"]` to reach (14,7) and rotate gate.
6. Follow the path North to Winona at (15,2).
**IMPORTANT:** Use exactly these sequences in your `press_buttons` calls.
- **Blocking:** Arms block movement. If an arm is at (8,11), you cannot move through (8,11).
- **Collision:** If 'walkable' tile (represented by '.') blocks you, it is a turnstile arm.
- **Strategy:** To reach Winona, you must push through gates in a sequence that opens the final path at the top-right.
- **Precision:** Use `press_buttons` for all gym movements. `navigate_to` is disabled in gyms for safety.


## DIALOGUE HANDLING
- **Spamming A is DANGEROUS:** If you press A while a dialogue is closing, you will immediately talk to the NPC again and restart the conversation.
- **Strategy:** Use `press_buttons(["A", "WAIT"], reasoning="...")` to advance dialogue one box at a time and see if it clears.
- **Check for Battle:** After dialogue, wait a step to see if the game state changes to 'battle'. Don't just keep pressing A.
- **Escape Loop:** If you've seen the same text 3 times, move AWAY from the NPC (e.g., move LEFT or DOWN) to break the cycle.
