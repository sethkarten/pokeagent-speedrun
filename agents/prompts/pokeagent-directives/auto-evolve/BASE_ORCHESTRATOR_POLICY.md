# Strategic Guidance (Seed)

You are playing **{game_name}** with no walkthrough or wiki. Learn game mechanics through observation and store them in memory.

## Decision Framework

**Every step, assess your context and act accordingly:**

- **Read the screenshot and game state text carefully.** Understand what mode you are in (exploration, combat, dialogue, menu) before acting.
- **In dialogue**: Advance with A presses. Store important information from dialogue in memory.
- **In exploration**: Move toward your objective. If you are doing manual movement repeatedly, build an executable skill to automate it (see "How to develop executable skills" in your system prompt).
- **In combat/battle**: If you have a battle subagent, delegate to it. Otherwise, observe the interface and learn the mechanics. Once you understand it, create a battle subagent.
- **In menus**: Navigate carefully with directional buttons and A/B.
- **Stuck for 3+ steps**: Use `run_code` to inspect current game state, or `process_trajectory_history` to diagnose.

## Building Your Toolkit

You start with NO tools for navigation or combat. Build them:

- **If you are repeatedly pressing directional buttons to move**: Stop and build a movement skill using `run_code`. Inspect the game state, write a loop, test it, save it.
- **If you encounter the same game mode multiple times** (combat, dialogue sequences, menus): Create a subagent to handle it.
- **If a skill or subagent fails**: Inspect the error with `run_code`, fix it, update it with `process_skill` or `process_subagent`.

## Knowledge Management

- **Memory**: Store game mechanics, locations, NPC info, team composition. Organize by path.
- **Skills**: Prototype with `run_code`, debug with `print()`, save working code with `process_skill`.
- **Objectives**: Create story/battling/dynamics objectives via `replan_objectives`.

---

*This block is updated over time by the evolution loop.*
