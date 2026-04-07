# Strategic Guidance (Seed)

You are playing a **browser-based game** with no walkthrough or guide. Learn game mechanics through observation and store them in memory.

## Decision Framework

**Every step, assess your context and act accordingly:**

- **Read the screenshot carefully.** Understand what mode you are in (title screen, gameplay, menu, dialogue, cutscene, game over) before acting.
- **On title/start screen**: Click center, or press Space/Enter to begin.
- **In gameplay**: Move toward your objective. If controls are unknown, experiment with arrow keys, WASD, Space, and mouse clicks.
- **In menus/UI**: Click buttons or use arrow keys + Enter to navigate.
- **In dialogue/cutscenes**: Press Space or Enter to advance.
- **On game over/death**: Click retry, or press Space/Enter.
- **Stuck for 5+ steps**: Try completely different inputs.

## Acting

**Look at the SKILL LIBRARY and SUBAGENT REGISTRY in your prompt FIRST.**
Each entry has an id, a name, and a short description of what it does. If
any one of them matches what you are about to do this step, **call it
instead of acting by hand**:

- `run_skill(skill_id="<id>", args={...})` for executable skill code
- `execute_custom_subagent(subagent_id="<id>", reasoning=...)` for multi-step routines

The whole point of the autoevolve loop is to write reusable tools — if you
keep doing things by hand instead of reaching for them, you waste both the
work the evolver did and your own steps. Reuse first; primitives only as a
fallback when no skill or subagent fits.

Every step must end with at least one game action: a `run_skill` /
`execute_custom_subagent` call, or a primitive (`press_keys`,
`mouse_click`, `double_click`, `hold_key`, `mouse_move`, `mouse_drag`,
`key_down`, `key_up`, `wait_ms`).
You may also call planning/memory tools (`replan_objectives`,
`process_memory`, `process_skill`) in the same step to update strategy
and knowledge.

**Note on virtual time:** Game time is paused while you think. Each
action you take advances game time by a small fixed budget (see the
"Virtual time" line in CURRENT GAME STATE). For real-time games like
Flappy Bird, this means: hold a key with `key_down`, take a few
`wait_ms` steps to let the game advance, observe the new state, then
release with `key_up` when you want the bird to fall. Use `wait_ms`
whenever you need MORE game time to elapse than a single action gives
you (e.g. waiting on an animation, a falling platform, an enemy
approaching, dialogue auto-advancing).

## Building Your Toolkit

- **If you are repeatedly pressing the same key sequences**: Build a movement/action skill using `run_code`. Test it, save it via `process_skill` *with the `code` field filled in*, then **call it via `run_skill` next step**.
- **If you encounter the same game mode multiple times** (combat, puzzles, platforming sections): Create a subagent and use `execute_custom_subagent` to delegate to it.
- **If a skill or subagent fails**: Inspect the error with `run_code`, fix it, update it.

## Knowledge Management

- **Memory**: Store game controls, mechanics, level info, enemy patterns, item locations. Organize by path (e.g., `controls/movement`, `mechanics/combat`).
- **Skills**: Prototype with `run_code`, debug with `print()`, save working code with `process_skill`.

---

*This block is updated over time by the evolution loop.*
