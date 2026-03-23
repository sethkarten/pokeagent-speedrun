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
- Loops with its own short-term memory (up to 25 turns). Has access to research tools (`get_walkthrough`, `search_knowledge`, etc.), other subagents, and a planner-exclusive `replan_objectives` tool. Returns when `return_to_orchestrator` is set on a successful replan.

**Seeing subagent output:** On the **next** step, the harness injects a **📋 RESULTS FROM PREVIOUS STEP** block with the full tool result (including `subagent_verify` JSON, a compacted `subagent_battler` summary, or `subagent_plan_objectives` changes).

### Knowledge

**add_knowledge**  
- **Required:** `category` (`location` | `npc` | `item` | `pokemon` | `strategy` | `custom`), `title` (string), `content` (string), `importance` (integer 1–5)  
- **Optional:** `location` (string), `coordinates` (string, e.g. `"x,y"`)

**search_knowledge**  
- **Optional:** `category`, `query`, `location`, `min_importance` (integer)

**get_knowledge_summary**  
- **Optional:** `min_importance` (integer; default 3)

**get_walkthrough**  
- **Required:** `part` (integer 1–21)

### Game information

**lookup_pokemon_info**  
- **Required:** `topic` (string)  
- **Optional:** `source` (string; default bulbapedia)

### Objectives and progress

**get_progress_summary**  
- *(no parameters)*

