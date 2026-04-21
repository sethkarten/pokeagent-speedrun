# Real Harness Prompt Evaluation: All Models

**Date**: 2026-04-13
**Dataset**: emerald_v3 (16 samples with REAL harness prompts)
**Models**: gemma4:e2b, gemma4-emerald:e2b, gemma4:e4b, gemma4-emerald:e4b, gemma4:26b, gemma4-emerald:26b

Uses the actual harness system prompt + game state + action history from training data,
NOT a simplified prompt. This tests how models behave in the real agent loop.

## Overall Scores

| Metric | Base e2b | SFT e2b | Base e4b | SFT e4b | Base 26b | SFT 26b |
|--------|-------|-------|-------|-------|-------|-------|
| tool_format | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 | 0.94 |
| actionable | 0.81 | 0.44 | 0.69 | 0.19 | 0.44 | 0.44 |
| grounding | 0.91 | 0.88 | 0.84 | 0.81 | 0.78 | 0.81 |
| reasoning_similarity | 0.38 | 0.34 | 0.52 | 0.42 | 0.39 | 0.46 |
| hallucination | 0.00 | 0.00 | 0.06 | 0.06 | 0.06 | 0.12 |
| degenerate | 0.88 | 0.06 | 0.44 | 0.00 | 0.19 | 0.00 |
| tok_s | 233.37 t/s | 233.41 t/s | 184.79 t/s | 183.13 t/s | 163.70 t/s | 164.10 t/s |

## Sample Responses (Real Prompts)

### Example 1: overworld @ ROUTE 104 MR BRINEYS HOUSE ALT (prompt: 54396 chars)

**Teacher (Gemini)**: `[press_buttons] ANALYZE: Location: Mr. Briney's House. Objective: Talk to Mr. Briney. Situation: I am at (6,4), Mr. Briney is at (5,3) and Peeko is at (6,3). PLAN: Action: Press LEFT to reach (5,4), UP to face Mr. Briney, and spam A. Expected: Dialog`

**Base e2b**: `Action executed: navigate_to_coords   TOOLS:     - navigate_to_coords       args: {"x": 11, "y": 9, "reasoning": "Navigate to Dewford Town, where the Pokemon Center is located, to complete the healing`

**SFT e2b**: `# 📝 Summary of Strategic Guidance  This document provides critical debugging information and a manual navigation plan for catching Mr. Briney on Route 104.  ## 🚨 CRITICAL DEBUGGING FINDINGS  1.  **Nav`

**Base e4b**: `**THINKING:** ANALYZE: Location: Mr. Briney's House. Objective: Sail to Dewford Town. Situation: I have successfully positioned myself in Mr. Briney's path and pressed 'A' repeatedly to advance the di`

**SFT e4b**: `ANALYZE: Location: Mr. Briney's House. Objective: Talk to Mr. Briney to set sail for Dewford Town. Situation: He is continuing his story about his boat, "The Wanderer." I need to keep pressing 'A' to `

**Base 26b**: ``

**SFT 26b**: `[complete_direct_objective] ANALYZE: The objective was to sail to Dewford Town. The game state shows the player is currently in Dewford Town (location: DEWFORD TOWN, coordinates: 11, 9). Evidence: The`

---

### Example 2: overworld @ GRANITE CAVE B1F ALT (prompt: 55883 chars)

**Teacher (Gemini)**: `[press_buttons] ANALYZE: Location: Granite Cave B1F. Objective: Deliver letter to Steven. Situation: Standing at (20, 16) in the dark cave. I am following the exact path to the B2F ladder previously calculated by run_code. PLAN: Action: Execute the f`

**Base e2b**: `The sequence of actions successfully navigated you from Granite Cave B1F to the area leading to the B2F warp, positioning you at coordinates (20, 16).  You have successfully executed the calculated pa`

**SFT e2b**: ````json {   "action_type": "press_buttons",   "buttons_queued": [     "LEFT",     "LEFT",     "LEFT",     "LEFT",     "LEFT",     "LEFT",     "LEFT",     "LEFT",     "LEFT",     "LEFT",     "LEFT",   `

**Base e4b**: ````json {   "tool_name": "run_code",   "args": {     "reasoning": "I have successfully navigated through the dark areas of Granite Cave B1F and am now at (20, 16). My objective is to reach the warp po`

**SFT e4b**: `ANALYZE: The wild Pokemon battle has started. The intro text is displayed. Objective: Catch the wild Pokemon. Situation: Need to advance the text to reach the main battle menu. PLAN: Action: Press A t`

**Base 26b**: `ANALYZE: Location: Granite Cave B1F. Objective: Deliver letter to Steven. Situation: Standing at (20, 16). I have completed the first 14 steps of the calculated path. The remaining path is: RIGHT, DOW`

**SFT 26b**: `[press_buttons] ANALYZE: The POKEMON menu is open and the party swap is complete. Marshtomp is now in Slot 1 (the large box on the left). Objective: Deliver the letter to Steven in Granite Cave B2F/B3`

---

### Example 3: overworld @ RUSTBORO CITY MART (prompt: 45708 chars)

**Teacher (Gemini)**: `[press_buttons] ANALYZE: Location: Rustboro City Mart (3, 7). Objective: Catch a Flying or Bug type Pokemon on Route 116. Situation: I accidentally walked into the Poke Mart when trying to move away from the Rival. I need to exit the Mart to head tow`

**Base e2b**: `Action executed: press_buttons   TOOLS:     - navigate_to_coords       args: {"x": 16.0, "y": 45.0, "reasoning": "ANALYZE: Location: Rustboro City Mart (3, 7). Objective: Catch a Flying or Bug type Po`

**SFT e2b**: `# 🧩 Subagent Registry  This registry lists all currently registered subagents, categorized by their primary operational context.  ## ⚔️ Combat & Battle Management These subagents are designed to handl`

**Base e4b**: `**THINKING:** ANALYZE: Location: Rustboro City Mart (3, 7). Objective: Prepare for Rival battle by catching a Flying/Bug type Pokemon. Situation: I am currently inside the Poke Mart, which is a buildi`

**SFT e4b**: ``

**Base 26b**: `To prepare for the rematch with Rival May, I will take advantage of being inside the Rustboro City Mart to restock on Potions. This will help ensure Marshtomp can survive longer if I cannot immediatel`

**SFT 26b**: `[execute_custom_subagent] ANALYZE: The agent is currently in the Pokemon Center (3, 7). The objective is to catch a Flying or Bug type Pokemon on Route 116 or Route 104. The previous attempt to use th`

---

### Example 4: overworld @ ROUTE 102 (prompt: 43754 chars)

**Teacher (Gemini)**: `[press_buttons] ANALYZE: We are at (11, 5) on Route 102. The path west continues to be clear. We are safely north of a trainer who is looking away. Objective: Continue west to Petalburg City. PLAN: Action: Press LEFT x5. Reason: Move further west alo`

**Base e2b**: `Action executed: press_buttons   TOOLS:     - press_buttons       args: {"buttons": ["LEFT", "LEFT", "LEFT", "LEFT", "LEFT"], "reasoning": "ANALYZE: We are at (11, 5) on Route 102. Our objective is to`

**SFT e2b**: ````json {   "tool_name": "navigate_to_coords",   "params": {     "x": 10,     "y": 13,     "reasoning": "The objective is to travel west on Route 102 to Petalburg City. The path is mostly clear, but t`

**Base e4b**: `**ANALYSIS:** We are at (11, 5) on Route 102. Our long-term objective is to reach Petalburg City. The path west is clear at our current Y level. We must continue moving west until we reach the boundar`

**SFT e4b**: `ANALYZE: We are at (11, 5) on Route 102. The path continues west, but we are approaching a large group of tall trees (a forest/woodland) that are blocking our direct path. We need to find a gap in the`

**Base 26b**: `THINKING: ANALYZE: We are at (11, 5) on Route 102. The path West at our current Y level is clear. Objective: Continue west to Petalburg City. PLAN: Press LEFT x5. Reason: Move west along the path towa`

**SFT 26b**: `[press_buttons] ANALYZE: The player is at (11, 5) on Route 102. The path west is clear at this Y level. Objective: Continue west to Petalburg City. PLAN: Action: Press LEFT x5. Reason: Move west along`

---

