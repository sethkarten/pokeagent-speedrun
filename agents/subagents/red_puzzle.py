"""Prompt helpers for the red-puzzle local subagent."""

from __future__ import annotations

import re
from typing import Any, Dict


# Red puzzle knowledge base
# Keys MUST match map_names.json exactly (e.g., "RocketHideoutB2f" not "RocketHideoutB2F")
RED_PUZZLES = {
    "RocketHideoutB2f": {
        "type": "spinner_maze",
        "description": "RocketHideoutB2f is a large spinner maze floor. The goal is to navigate from the top-right staircases (Staircase 1 at D(27,8) or Staircase 2 at D(21,8)) down to the bottom-right area where Staircase 4 D(21,22) leads to B3F and the Lift D(24,19)/D(25,19) goes to other floors. The maze occupies the left half of the map (x=1..16). The right half (x=18..28) is open corridor with no spinners. Spinner tiles (←→↑↓) push the player in their arrow direction until hitting a wall or a stop tile (*). Landing on another spinner mid-push chains the direction automatically. You cannot move or press any button while being pushed — wait until fully stopped.",
        "strategy": """RULES — read these first:
1. NEVER step back onto the warp you came from (the staircase you entered through).
2. While a spinner is pushing you, do NOT press any button. Wait until the player fully stops before your next action.
3. Do NOT use navigate_to if the path might cross a spinner tile. Only use navigate_to on segments confirmed free of spinners.
4. Spinner symbols on the map: ← → ↑ ↓ push you in that direction. Stop tiles marked * halt spinner movement. Walls also stop movement.
5. The maze has 6 spinner chains separated by manual navigation segments. Follow each phase in order.

GENERAL ROUTE (9 phases):
  Entry D(27,8) → walk to (18,11) → step LEFT onto ←(17,11) → chain to *(2,9)
  → manually walk down to (4,14) → step RIGHT onto →(5,14) → chain to *(9,16)
  → step RIGHT onto →(11,16) → chain to (15,18) [wall stop]
  → step LEFT onto ←(13,18) → chain to (11,20) [wall stop]
  → manually walk to (14,22) → step LEFT onto ←(13,22) → chain to *(9,24)
  → walk RIGHT+DOWN onto →(10,25) → chain to *(14,25)
  → exit maze to Staircase 4 D(21,22) or Lift D(24,19)

DETAILED STEP-BY-STEP:

Phase 1 — Entry to first spinner (safe, use navigate_to):
  You enter from a staircase on the right side: D(27,8) or D(21,8).
  Use navigate_to to walk to (18,11). The right corridor is open floor with no spinners.

Phase 2 — First spinner chain (6 spinners):
  From (18,11), press LEFT once. You step onto ←(17,11).
  Chain: ←(17,11) → ↑(12,11) → ←(12,9) → ←(10,9) → ←(8,9) → ←(4,9) → *(2,9)
  This is a long 6-spinner chain. Do NOT press anything until you stop at *(2,9).

Phase 3 — Manual walk down to →(5,14) (use press_buttons only):
  From *(2,9), navigate down to (4,14) avoiding the →(4,11) spinner.
  Follow this exact path with press_buttons:
    RIGHT to (3,9)
    DOWN, DOWN to (3,11) — do NOT press RIGHT here, →(4,11) is a spinner!
    DOWN, DOWN to (3,13)
    RIGHT to (4,13)
    DOWN to (4,14)
  Verify your position is (4,14) before continuing.

Phase 4 — Second spinner chain:
  From (4,14), press RIGHT once. You step onto →(5,14).
  Chain: →(5,14) → ↓(9,14) → *(9,16)
  WAIT until you are fully stopped at *(9,16).

Phase 5 — Third spinner chain:
  From *(9,16), press RIGHT once to (10,16), then press RIGHT again.
  You step onto →(11,16).
  Chain: →(11,16) → →(13,16) → ↓(15,16) → STOP at (15,18) [wall below blocks further movement]
  WAIT until you are fully stopped at (15,18).

Phase 6 — Fourth spinner chain:
  From (15,18), press LEFT once to (14,18), then press LEFT again.
  You step onto ←(13,18).
  Chain: ←(13,18) → ↓(11,18) → STOP at (11,20) [wall below blocks further movement]
  WAIT until you are fully stopped at (11,20).

Phase 7 — Manual walk to ←(13,22) (use press_buttons only):
  From (11,20), navigate to (14,22). All tiles on this path are floor.
  Follow this exact path with press_buttons:
    RIGHT to (12,20)
    RIGHT to (13,20)
    RIGHT to (14,20)
    DOWN to (14,21)
    DOWN to (14,22)
  CRITICAL: Step LEFT from (14,22) onto ←(13,22) — this is the CORRECT spinner.
  Do NOT go to (14,23) and step LEFT onto ←(13,23) — that spinner sends you BACKWARDS into the upper maze!

Phase 8 — Fifth spinner chain:
  From (14,22), press LEFT once. You step onto ←(13,22).
  Chain: ←(13,22) → ↓(9,22) → *(9,24)
  WAIT until you are fully stopped at *(9,24).

Phase 9 — Last spinner + exit:
  From *(9,24), press RIGHT once to (10,24).
  Then press DOWN once. You step onto →(10,25).
  Chain: →(10,25) → *(14,25)
  WAIT until you are fully stopped at *(14,25).
  The maze is now complete! From *(14,25), use navigate_to to reach your destination:
  - Staircase 4 to B3F: navigate_to D(21,22). Path goes RIGHT to (16,25), UP to (16,22), then RIGHT through (17,22)→(17,20)→(19,20)→(19,22)→(21,22).
  - Lift: navigate_to D(24,19). Path goes RIGHT+UP through the open area east of the maze.
  Both exit paths are free of spinners — navigate_to is safe.

IF YOU GET STUCK:
  Identify your current coordinates and find the nearest waypoint below:
  *(2,9) → continue from Phase 3 (walk down to Phase 4)
  (4,14) → continue from Phase 4 (step RIGHT onto →(5,14))
  *(9,16) → continue from Phase 5 (step RIGHT toward →(11,16))
  (15,18) → continue from Phase 6 (step LEFT toward ←(13,18))
  (11,20) → continue from Phase 7 (walk to (14,22))
  (14,22) → continue from Phase 8 (step LEFT onto ←(13,22))
  *(9,24) → continue from Phase 9 (step RIGHT to (10,24), then DOWN)
  *(14,25) → maze done, navigate_to your destination
  If you are at an unknown position in the maze, try to manually walk (press_buttons only) to the nearest waypoint above without stepping on any spinner tiles."""
    },
    "RocketHideoutB3f": {
        "type": "spinner_maze",
        "description": "RocketHideoutB3f is a spinner maze floor. The goal is to get from the top-right staircase D at (25,6) down to the bottom-right staircase D at (19,18), which leads to B4F. Spinner tiles (←→↑↓) push the player in their arrow direction until the player hits a wall or a stop tile (*). If the player lands on another spinner, the chain continues automatically. You cannot move or press any button while being pushed by a spinner — wait until the player stops completely before acting.",
        "strategy": """RULES — read these first:
1. NEVER step back onto the warp at (25,6) — that goes back up to B2F.
2. While a spinner is pushing you, do NOT press any button. Wait until the player fully stops before your next action.
3. Do NOT use navigate_to if the path might cross a spinner tile. Only use navigate_to on segments confirmed free of spinners.
4. Spinner symbols on the map: ← → ↑ ↓ push you in that direction. Stop tiles marked * halt spinner movement.

GENERAL ROUTE (6 phases):
  Entry D(25,6) → walk to (13,11) → step LEFT onto spinner ← → pushed to *(10,11)
  → step DOWN twice onto spinner → at (10,13) → pushed RIGHT to *(14,13)
  → manually navigate down-left-down to (10,18) avoiding all spinners
  → step RIGHT onto spinner → at (11,18) → chains through ↓(15,18) → stops at *(15,22)
  → walk from (15,22) to exit D(19,18)

DETAILED STEP-BY-STEP:

Phase 1 — Entry to maze edge (safe, use navigate_to):
  You start at D(25,6). Use navigate_to to walk to (13,11).
  This entire area is open floor with no spinners. navigate_to is safe here.

Phase 2 — First spinner push:
  From (13,11), press LEFT once. You step onto the ← spinner at (12,11).
  The spinner pushes you LEFT until you hit stop tile * at (10,11).
  WAIT until you are fully stopped at (10,11) before pressing anything.

Phase 3 — Second spinner push:
  From *(10,11), press DOWN once to (10,12). Then press DOWN again.
  You step onto the → spinner at (10,13).
  The spinner pushes you RIGHT until you hit stop tile * at (14,13).
  WAIT until you are fully stopped at (14,13).

Phase 4 — Manual navigation avoiding spinners (use press_buttons only):
  From *(14,13), you must navigate to (10,18) while avoiding all spinner tiles.
  DO NOT use navigate_to here — the pathfinder may route through spinners.
  Follow this exact path with press_buttons:
    LEFT, LEFT to (12,13)
    DOWN, DOWN, DOWN to (12,16)
    LEFT, LEFT, LEFT to (9,16)
    DOWN, DOWN to (9,18)
    RIGHT to (10,18)
  Verify your position is (10,18) before continuing.

Phase 5 — Final spinner chain:
  From (10,18), press RIGHT once. You step onto the → spinner at (11,18).
  This spinner pushes you RIGHT. You will pass through and land on ↓ spinner at (15,18).
  The ↓ spinner then pushes you DOWN until you hit stop tile * at (15,22).
  This is a two-spinner chain — do NOT press anything until you fully stop at (15,22).

Phase 6 — Walk to exit staircase (safe, use navigate_to):
  From *(15,22), use navigate_to to reach D(19,18).
  The path goes DOWN then RIGHT then UP — the south and east areas are open floor with no spinners.
  navigate_to is safe here. D(19,18) is the staircase to B4F.

CRITICAL REMINDERS:
- If you accidentally step on a wrong spinner, you may be sent to a dead end. Check your coordinates after every spinner push.
- The only correct spinner sequence is: ←(12,11) - →(10,13) - →(11,18) - ↓(15,18). Any other spinner leads to a wrong path.
- Between Phase 3 and Phase 5, every move must be done manually with press_buttons to avoid spinners at (14,15), (13,16), (15,16), (12,17), (14,17), (10,19), (14,19), (12,20)."""
    },
}


def resolve_location_name(arguments: Dict[str, Any], state_text: str) -> str:
    location_name = arguments.get("location_name")
    if location_name:
        return str(location_name)

    location_match = re.search(r"Current Location: ([^\n]+)", state_text or "")
    if location_match:
        return location_match.group(1).strip()
    return "Unknown"


def get_red_puzzle_info(location_name: str) -> Dict[str, Any]:
    return RED_PUZZLES.get(
        location_name,
        {
            "type": "unknown",
            "description": "Unknown location - no specific puzzle guidance available",
            "strategy": "Explore the area carefully and look for patterns in tile arrangements.",
        },
    )


def build_red_puzzle_prompt(
    *,
    location_name: str,
    puzzle_info: Dict[str, Any],
    state_text: str,
    action_history: str,
    function_results: str,
) -> str:
    return f"""You are analyzing a Pokemon Red puzzle to help the agent solve it.

LOCATION: {location_name}
TYPE: {puzzle_info.get("type", "unknown")}
DESCRIPTION: {puzzle_info.get("description", "")}

GENERAL STRATEGY:
{puzzle_info.get("strategy", "")}

RECENT ACTION HISTORY:
{action_history}

{function_results}

CURRENT GAME STATE:
{state_text}

Provide your analysis in this format:

**PUZZLE ANALYSIS**:
[Explain how this specific puzzle works based on the map and your current position]

**WHAT WE'VE TRIED**:
[Based on the action history above, summarize what approaches have been attempted and what worked/didn't work]

**SPECIFIC SOLUTION STEPS**:
1. [First concrete action with coordinates if applicable]
2. [Second action]
3. [Continue...]

**NAVIGATION TIPS**:
[Any important details about tile types, warps, or obstacles to watch for]

**IMPORTANT**:
- Look at the map in the game state. Tiles marked '#' are walls, '.' are walkable, 'D' are doors/warps, 'S' are stairs, '*' are spinner stop tiles, and arrow symbols (←→↑↓) are spinner tiles that push the player in that direction.
- Review the action history to avoid repeating failed attempts.
- Learn from previous outputs and function results to refine your strategy.
- **USE press_buttons() FOR PUZZLE AREAS**: Do NOT use navigate_to() in spinner mazes or puzzle areas - it doesn't work well with forced movement tiles. Instead, use press_buttons() with explicit directional inputs (UP, DOWN, LEFT, RIGHT) to solve puzzles step by step.
Be specific and actionable. Reference actual coordinates from the map when possible."""
