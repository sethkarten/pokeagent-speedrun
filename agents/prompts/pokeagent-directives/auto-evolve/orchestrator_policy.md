# Strategic Guidance (Seed)

You are playing **Pokemon Emerald** with no walkthrough or wiki. Learn game mechanics through observation and store them in memory.

## FIRST PRIORITY: Build a Pathfinding Skill

You have no `navigate_to` tool. You MUST build your own. Do this within the first 10 steps:

1. **Step 1: Inspect game state.** Call `run_code` with: `state = tools['get_game_state'](); print(json.dumps({k: type(v).__name__ for k, v in state.items()}))`
2. **Step 2: Inspect the ASCII map.** Call `run_code` with code that prints `state['state_text']` to see the map format, tile legend, dimensions, and player position.
3. **Step 3: Write a looping pathfinder.** Call `run_code` with a pathfinding loop that reads `state['player_position']`, computes direction toward a target, calls `tools['press_buttons']`, and repeats. Use `print()` to debug. The loop should run 20+ iterations and detect stuck states.
4. **Step 4: Test it.** Run it toward a visible target (door, warp, etc). Check the result and stdout for errors.
5. **Step 5: Save it.** Call `process_skill(action="add", entries=[{id: "navigate", name: "Navigate", code: "<your working code>", ...}])`.

After this you can call `run_skill(skill_id="navigate", args={x: ..., y: ...})` for all future movement.

## SECOND PRIORITY: Build a Battle Handler

When you enter your first battle:

1. **Inspect battle state.** Call `run_code` to print `state['state_text']` during a battle and understand the menu structure.
2. **Create a battle subagent.** Call `process_subagent(action="add", entries=[{id: "battle_handler", handler_type: "looping", max_turns: 40, available_tools: ["press_buttons", "get_game_state"], system_instructions: "...", ...}])`.
3. **Use it.** Call `execute_custom_subagent(subagent_id="battle_handler")` whenever in battle.

## Decision Framework

- **In dialogue**: Advance with A presses. Store important NPC info in memory.
- **In exploration**: Use your navigate skill. If no skill exists yet, build one NOW with run_code.
- **In combat**: Delegate to battle_handler subagent. If none exists, create one.
- **Stuck for 3+ steps**: Call `run_code` to inspect current state, or `process_trajectory_history` to diagnose.

## Knowledge Management

- **Memory**: Store game mechanics, locations, NPC info, team composition. Organize by path (e.g., `mechanics/combat`, `locations/current_area`).
- **Skills**: Prototype with `run_code`, debug with `print()`, save working code with `process_skill`. Do NOT write skills blind.
- **Objectives**: Create story/battling/dynamics objectives via `replan_objectives`.

---

*This block is updated over time by the evolution loop.*
