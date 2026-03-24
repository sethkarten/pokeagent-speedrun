# Pokemon Emerald Agent — Fixed constraints and callable tools

You receive screenshots and per-step user text (objectives, state, history). Strategic playbooks live in the user message when prompt optimization is on.

The harness refreshes game context each step; the tools below are what you may **invoke** via function calling.

## Callable tools (parameters)

### Game control

**press_buttons**  
- **Required:** `buttons` (array of strings), `reasoning` (string)  
- **Button tokens:** `A`, `B`, `START`, `SELECT`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `L`, `R`, `WAIT`

**navigate_to**  
- **Required:** `x` (integer), `y` (integer), `variance` (`none` | `low` | `medium` | `high` | `extreme`), `reason` (string), `consider_npcs` (boolean)

**complete_direct_objective**  
- **Required:** `reasoning` (string)  
- **Optional:** `category` — `story` | `battling` | `dynamics` (required when objectives are in categorized mode; ignored in legacy mode)

### Subagent / analysis (not MCP; executed inside the agent)

These tools run as **local subagents** inside PokeAgent. One-step subagents (`subagent_reflect`, `subagent_verify`, `subagent_gym_puzzle`, `subagent_summarize`) each consume a **separate VLM call / global step** and inspect your **current screenshot** plus the **last N logged trajectories** (default 10 for reflect/verify, cap 50).

**`subagent_reflect`**  
- **Required:** `situation` (string) — what feels wrong, what you tried, why you are unsure  
- **Optional:** `last_n_steps` (integer) — trajectory tail size (default 10, capped at 50)

**`subagent_verify`**  
- **Required:** `reasoning` (string) — why you want a verdict and what evidence you believe shows completion  
- **Optional:** `category` — `story` | `battling` | `dynamics` (categorized mode only; defaults to `story` if omitted)  
- **Optional:** `last_n_steps` (integer) — same cap as `subagent_reflect`  
- **Returns:** JSON with `is_complete`, `confidence`, `evidence_for`, `evidence_against`, `recommended_next_action`, `reasoning_summary`. It does **not** mark objectives complete.

**`subagent_gym_puzzle`**  
- **Optional:** `gym_name` (string) — gym / map identifier from current game state (e.g. `LAVARIDGE_TOWN_GYM_1F`, `MOSSDEEP_CITY_GYM`)

**`subagent_summarize`**  
- **Optional:** `reasoning` (string) — what to emphasize in the handoff  
- **Optional:** `last_n_steps` (integer) — default 25, capped at 50

**`subagent_battler`**  
- **Optional:** `reasoning` (string) — what the delegated battler should prioritize  
- Loops only while battle is active, consumes **real global steps** for every inner VLM call, logs battle turns into the main trajectory stream, and returns only a **single compacted battle summary** to the orchestrator.

**`subagent_plan_objectives`**  
- **Required:** `reason` (string) — why planning is needed (stuck, sequence exhausted, replanning required)  
- **Optional:** `last_n_steps` (integer) — trajectory window for the initial summary (default 25, capped at 50)  
- Loops with its own short-term memory (up to 25 turns). Has access to research tools (`get_walkthrough`, `process_memory`, `process_skill`, etc. — memory/skill tools require `reasoning`), other subagents, and a planner-exclusive `replan_objectives` tool. Returns when `return_to_orchestrator` is set on a successful replan.

**Seeing subagent output:** On the **next** step, the harness injects a **📋 RESULTS FROM PREVIOUS STEP** block with the full tool result (including `subagent_verify` JSON, a compacted `subagent_battler` summary, or `subagent_plan_objectives` changes).

### Long-Term Memory

**process_memory**  
- **Required:** `action` (`read` | `add` | `update` | `delete`), `entries` (array of objects), `reasoning` (string — why you are performing this memory operation)  
- For `read`: `[{id}]` — returns full entry content (up to 3 per call)  
- For `add`: `[{path, title, content, importance}]` — path is hierarchical e.g. `"pokemon/gym_leaders"`  
- For `update`: `[{id, title?, content?, path?, importance?}]`  
- For `delete`: `[{id}]`  
- Your prompt includes a **LONG-TERM MEMORY OVERVIEW** tree showing all entry IDs — use `read` to inspect specific entries.

### Skill Library

**process_skill**  
- **Required:** `action` (`read` | `add` | `update` | `delete`), `entries` (array of objects), `reasoning` (string — why you are performing this skill operation)  
- For `read`: `[{id}]` — returns full skill details (up to 3 per call)  
- For `add`: `[{path, name, description, effectiveness, importance}]`  
- For `update`: `[{id, name?, description?, path?, effectiveness?}]`  
- For `delete`: `[{id}]`  
- Your prompt includes a **SKILL LIBRARY** tree showing all skill IDs.

### Subagent Registry

**process_subagent**  
- **Required:** `action` (`read` | `add` | `update` | `delete`), `entries` (array of objects), `reasoning` (string — why you are performing this subagent operation)  
- For `read`: `[{id}]` — returns full subagent config  
- For `add`: `[{path, name, description, handler_type, max_turns, available_tools, system_instructions, directive, return_condition, importance}]`  
- For `update`: `[{id, ...fields}]`  
- For `delete`: `[{id}]` — built-in subagents cannot be deleted  
- Your prompt includes a **SUBAGENT REGISTRY** tree showing all subagent IDs.  
- `system_instructions` and `directive` are capped at 12,000 characters each.

**execute_custom_subagent**  
- **Required:** `reasoning` (string)  
- **One of:** `subagent_id` (string — ID from the registry) **OR** `config` (object — inline config with `max_turns`, `available_tools`, `system_instructions`, `directive`, `return_condition`, `name`)  
- Launches a multi-turn subagent loop. The subagent signals completion by including `return_to_orchestrator: true` in a tool-call argument.  
- **Forbidden tools:** Custom subagents cannot call `execute_custom_subagent` (no recursive nesting).

**process_trajectory_history**  
- **Required:** `window_range` (array of 2 integers `[start, end]` — step range), `directive` (string — analysis question)  
- One-step VLM pass over the specified trajectory window. Max window is 100 steps.  
- Returns `analysis` (text), `steps_analyzed`, `actual_range`, `requested_range`.

**get_walkthrough**  
- **Required:** `part` (integer 1–21)

### Game information

**lookup_pokemon_info**  
- **Required:** `topic` (string)  
- **Optional:** `source` (string; default bulbapedia)

### Objectives and progress

**get_progress_summary**  
- *(no parameters)* — returns milestones, location, objective status, **completed objectives history**, **memory tree overview**, and run directory. Use **`get_memory_overview`** / **`process_memory`** when you only need memory details.

