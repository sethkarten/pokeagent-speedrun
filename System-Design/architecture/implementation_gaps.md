# Implementation gaps & known issues

Living list of **behavior that diverges from intent** or is easy to misread. **Code is authoritative**; update this file when fixing or confirming a gap.

Profiling input: system-design-profiler review of `utils/stores/*`, `agents/utils/harness_evolver.py`.

## Map rendering parity (UI vs LLM / movement preview)

- **Status:** Open — UI `MAP & ACTIONS` can diverge from the post-filtered porymap view used for movement preview and LLM context.
- **Current behavior:** Movement preview is porymap-aware; `visual_map` in `/state` still merges multiple producers (SLAM, memory reader, stitcher, porymap) with cache reuse, so stale or mismatched panels are possible.
- **Desired end state:** One canonical map artifact per `/state`; UI panel and movement preview both consume it.

---

## 1. `mutation_history` and `last_modified_step` on stores

### `mutation_history` (memory, subagents, skills)

**What works**

- New entries get `mutation_history=[]` in `BaseStore.add` (`utils/stores/base_store.py`).
- `BaseStore.update` compares incoming fields to the existing entry, appends `{"timestamp", "fields": {key: {old, new}}}` when something actually changed, and skips the key `mutation_history` so callers cannot overwrite history via `update`.
- Full objects (including `mutation_history`) are persisted in JSON via `_serialize_entry` / `save`.

**Why it can look “broken”**

- Orchestrator-facing reads use `to_display_dict`, which **strips** `mutation_history` (and `last_modified_step`, timestamps, etc.) per `_INTERNAL_FIELDS` in `base_store.py`. Inspecting only tool/API payloads will not show history even though the on-disk JSON has it.
- `updated_at` is refreshed on **every** `update` call, even when `changed` is empty (no field deltas). History only grows when at least one non-`mutation_history` field changed—so timestamps and history can disagree.
- `remove` / `clear` delete entries with **no** tombstone or final history line on that entry.

### `last_modified_step` (memory only)

- Defined on `MemoryEntry` and defaulted on load/migration in `utils/stores/memory.py`.
- **Not** present on subagent or skill entries.
- **Grep ground truth:** outside tests and store definitions, nothing in the repo sets `last_modified_step` during gameplay or evolution. If the design assumed “last touched at agent step *N*,” that wiring is **missing** (and harness evolution receives `current_step` but does not pass it into memory `add`/`update`).

### Related: HarnessEvolver vs `BaseStore.add`

`BaseStore.add` returns the new entry’s **`str` id**, but `_evolve_subagents`, `_evolve_skills`, and `_evolve_memory` treat the return value as an entry object (`entry.id`, `entry.name`, …). That raises after a successful `add` (the row is still saved). See `agents/utils/harness_evolver.py` (create branches) vs `base_store.py` `return entry_id`.

### Related: Falsy filter on evolved subagent/skill updates

Subagent and skill evolution build `fields` with `if k != "id" and v`, which drops **`0`**, **`""`**, and **`False`**. Legitimate updates can be skipped. Memory evolution uses `v is not None` instead.

---

## 2. Planner does an extra post-replan objective fetch

### Symptoms

- In the planner loop, each successful `replan_objectives` tool call is immediately followed by another MCP call to `get_full_objective_sequence`.
- This adds an extra round-trip per successful replan turn even when the planner only needs the replan result metadata to continue.

### Ground-truth files

- `agents/PokeAgent.py` (`_run_planner_loop`): after `replan_objectives` success, it calls `get_full_objective_sequence` and refreshes `cached_sequence`.
- `server/app.py` (`mcp_replan_objectives`): returns edit results and metadata, but not a full snapshot payload.

### Current stance

- Known inefficiency accepted for now to keep Objective Planning API cleanup scoped to contract consolidation and legacy-surface removal.
- If optimized later, prefer extending `replan_objectives` response contract (or introducing an optional snapshot flag) so planner turns can skip the immediate fetch.

---

## 3. Milestone cache vs checkpoint milestone files

### Symptoms

- Runs often show both `milestones_progress.json` and `checkpoint_milestones.json` with nearly identical milestone payloads, which can look redundant at first glance.

### Ground-truth files

- `pokemon_env/emulator.py`:
  - `MilestoneTracker` defaults `self.filename` to `milestones_progress.json` in cache.
  - `save_to_file()` writes live milestone progress continuously (e.g., after `mark_completed`).
  - `save_milestones_for_state(state_filename)` writes `<state_basename>_milestones.json` (checkpoint companion file).
- `server/app.py`:
  - `/checkpoint` saves `checkpoint.state` plus `save_milestones_for_state(checkpoint.state)` -> `checkpoint_milestones.json`.
  - `/load_checkpoint` loads milestones from `load_milestones_for_state(checkpoint.state)`.

### Current stance

- These files are **overlapping snapshots of the same schema**, but they are **not redundant in lifecycle**:
  - `milestones_progress.json` is the live runtime cache file.
  - `checkpoint_milestones.json` is the snapshot intentionally paired to `checkpoint.state` for resume consistency.
- Do not merge/remove one as part of Optimization Config refactors unless checkpoint save/load semantics are redesigned and revalidated.

---

## 4. Fortree City Gym: rotating gates vs 2D map representation

### Status

**Not fully addressed.** Rotating gates are edge- and state-dependent (approach direction, pillar-blocked sweeps, orientation vars). The project does not yet have a single representation that makes the agent’s 2D map, pathfinding, and on-screen truth obviously consistent for Fortree the way Mauville Gym’s live-metatile overlay does for tile-level barriers.

### Prototype on branch `tersoo-dev-2-fortree`

A **pathfinding-first** direction was explored on that branch:

- Gate interaction zones are marked on the ASCII grid with `?` (prompt to verify walkability from the game frame).
- Live gate orientations are read from script vars and passed into pathfinding; `_can_move_to` applies a Fortree-specific **edge constraint** (`fortree_movement_blocked` in `utils/mapping/dynamic_map_overlay.py`) so A* can reject moves that the ROM would block for that direction.
- The static Porymap grid still does not model gate arms as tiles; gates remain “between” tiles in the decomp.

This improves `navigate_to` for many cases but is **not a complete solution**: multi-step manipulation (rotate gate A to open a path to gate B) depends on changing orientations over time, while pathfinding sees a snapshot; the `?` hint acknowledges residual uncertainty.

### Advisor direction (future work)

Focus on the **2D map itself** rather than (or in addition to) pathfinding hacks. One concrete idea under discussion: **load a custom Fortree map** (or derived layer) where **gate arms are explicit tiles** (walkable / blocked per orientation or per “puzzle state”), so the same grid drives UI, LLM-visible ASCII, and A* without collapsing directional mechanics into ambiguous `#` / `.` cells. That would require either hand-authored state maps, procedural projection from `read_var` orientations into a tile grid, or both.

### Ground-truth files

- `utils/mapping/dynamic_map_overlay.py` — Mauville vs Fortree strategies, `fortree_movement_blocked`, gate config.
- `utils/mapping/pathfinding.py` — `fortree_gate_orientations` on map payload, Fortree branch inside `_can_move_to`.
- `server/game_tools.py` — `apply_live_metatile_overlay` before pathfinding.

---

## 5. Elevation-aware Porymap: agent sees `^`, stream MAP often does not

### What works (agent + pathfinding)

- `utils/state_formatter.py` calls `_format_porymap_info` (`utils/mapping/porymap_state.py`), which flood-fills from the player and builds a **filtered grid**: walkable tiles at another elevation become **`^`**; unreachable water from land stays **`#`**; walls stay **`#`**.
- The same pass regenerates **`json_map['ascii']`** and appends it under **“ASCII Map:”** in the LLM context, with a legend defining **`^`** (walkable at different elevation — use ramp/stairs).
- **`map_info['porymap']['grid']`** matches that grid for pathfinding; **`utils/mapping/pathfinding.py`** treats **`^`** like walkable ground; edge elevation is enforced in **`_can_move_to`**.

So the **full-map ASCII in the agent prompt includes `^`** wherever the filter applies (not limited to a 15×15 window).

### Symptoms (stream / `stream.html`)

- The MAP panel reads **`data.map.visual_map`** from **`GET /state`** (`server/app.py`). Multiple earlier branches can set **`visual_map`** (SLAM file, memory_reader, map stitcher, **`format_map_for_llm`**, etc.); a later overlay (**`_build_porymap_visual_map_15x15`**) may not win or may see no **`porymap.grid`** on that tick.
- Even when the overlay runs, the UI is only **15×15** around the player — **`^`** appears only if those cells are in-window.
- **`server/stream.html`** drops rows whose trimmed line matches **`/^[\d\-\s#N]+$/`**, which can yield **`NO_MAP_DATA`** in edge cases.

### Current stance

- Treat the **stream MAP as best-effort**; **authoritative navigation semantics** for the agent are the **prompt ASCII map** and **`porymap` grid** used by pathfinding.

### Ground-truth files

| Concern | Location |
|--------|----------|
| Flood-fill, `^` / `#`, ASCII | `utils/mapping/porymap_state.py` |
| Agent prompt + `map_info['porymap']` | `utils/state_formatter.py` |
| Pathfinder + `^` | `utils/mapping/pathfinding.py` |
| Movement preview label for `^` | `utils/state_formatter.py` |
| HTTP **`visual_map`** | `server/app.py` — `get_comprehensive_state` |
| Browser polling | `server/stream.html` — `updateMapVisualization` |

### Pokémon Red (`GAME_TYPE=red`)

- **§5 above is Emerald / Porymap–centric.** Red skips the porymap **`visual_map`** branch; map text and `map_source` come from **`RedMapReader`** / `pokemon_red_env` (see [pokemon_infrastructure.md](pokemon_infrastructure.md)). Stream MAP parity issues may differ from Emerald; treat Red map UI as best-effort against the Red reader output.

---

## 6. `save_cumulative_metrics()` rewrites entire JSON on every step

### Status

**Open — high priority for long-running experiments.**

### Symptoms

- `save_cumulative_metrics()` serializes the entire `cumulative_metrics` dict (including the full `steps` array with per-step tool_calls and actions_executed) and rewrites `cumulative_metrics.json` on disk.
- It is called from `log_interaction`, `add_step_tool_calls`, `add_step_actions`, `log_milestone_completion`, `log_objective_completion`, and the `take_action` endpoint — multiple times per agent step.
- After multi-day runs (7,500+ steps), the file reaches 80+ MB. Each rewrite involves full JSON serialization and a file write of that size.

### Ground-truth files

- `utils/data_persistence/llm_logger.py` — `save_cumulative_metrics()`
- `server/app.py` — `take_action()` (calls `llm_logger.save_cumulative_metrics()`)

### Recommended direction

- Append-only JSONL for per-step rows; keep aggregates (and milestones/objectives) in `cumulative_metrics.json`; align `reconciling_cumulative_metrics.md` and `/sync_llm_metrics`. Scope as its own refactor.

## Adding new items

Title, **symptoms**, **ground-truth files**, optional fix. Link [harness_evolver.md](harness_evolver.md), [data_persistence.md](data_persistence.md) instead of duplicating specs.
