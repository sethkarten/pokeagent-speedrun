# Pokemon Emerald distillation plan — Gemini Pro 3.1 → gemma4:e4b

> Captured 2026-04-08. Lock-in document for the multi-phase distillation
> of Gemini Pro 3.1's Pokemon Emerald play into a 4B local agent. This
> file is the source of truth — every implementation decision below
> traces back to it.

## North star

A fine-tuned **gemma4:e4b** that plays Pokemon Emerald from the
auto-evolve init state to the **third gym** at Gemini Pro 3.1's
quality, while running locally on a single 5090 in real time.

The architecture is deliberately constrained:

- **Frontier teacher = Gemini Pro 3.1 only.** No GLM, no Claude, no
  GPT. Other frontier models aren't worth collecting data with unless
  they expose token logprobs, and we verified live that none of them
  do as of 2026-04-07 (the only logprobs-supporting vision model on
  any major API is GPT-4o, which is too old to be useful).
- **Student = gemma4:e4b.** 4B dense, the smallest gemma4. Local
  inference on a 5090, easy to ship via Ollama.
- **Local fallback teacher: none.** gemma4:31b is not good enough
  per the project's stance, so the Thinking Machines on-policy
  distillation recipe (which needs teacher logprobs) is **not viable**
  for Phase 2. We substitute with on-policy DPO.

## The data path

Every Gemini Pro 3.1 trajectory we collect uses this exact command,
matching the existing autoevolve scaffold and the path through
Birch's Lab → 3rd gym:

```bash
uv run python run.py \
  --backend gemini --model-name gemini-3.1-pro-preview \
  --backup-state Emerald-GBAdvance/auto-evolve_init.zip \
  --port 8778 --agent-auto --scaffold autoevolve \
  --enable-prompt-optimization --optimization-frequency 50 \
  --direct-objectives autonomous_objective_creation \
  --direct-objectives-start 6 \
  --record \
  --run-name ae_autoevolve
```

Notes on the flags:

- `--scaffold autoevolve` selects the empty-registry / harness-evolution
  scaffold (`H_auto`) — the autoevolve loop is the only way the harness
  improves itself across the run, so we want it on for both data
  collection and the final student.
- `--enable-prompt-optimization --optimization-frequency 50` runs the
  prompt optimizer every 50 steps. It's part of the autoevolve loop and
  produces the directive signal we harvest as a quality boost.
- `--direct-objectives autonomous_objective_creation
  --direct-objectives-start 6` starts the agent at objective index 6
  in the autonomous-creation sequence — that's the path from the
  Pokedex / starting state through to the third gym.
- `--record` captures a Playwright video of the entire run (browser
  games) or per-step frames (Pokemon, after the screenshot patch).
- `--run-name ae_autoevolve` appends a stable suffix so the run dir is
  greppable.

## What gets recorded per step

After the PokeAgent screenshot patch (added 2026-04-08), every step
of a Pokemon run writes the following to `run_data/run_<id>/`:

| Path | What's in it | Used by |
|---|---|---|
| `trajectory_history.jsonl` | per-step `{step, reasoning, action, pre_state, outcome, location, player_coords, llm_prompt}` | trajectory exporter (primary index) |
| `screenshots/step_NNNNN.png` | per-step screenshot, written by PokeAgent right after `get_game_state()` returns | trajectory exporter (image input) |
| `prompt_evolution/llm_traces/llm_log.jsonl` | every LLM call: `prompt`, `response`, `model_info`, `metadata.token_usage`, `agent_step` | exporter joins on `agent_step` |
| `prompt_evolution/meta_prompts/steps_X_to_Y_metadata.json` | autoevolve directive output for each evolution window — what was added/updated/kept | exporter attaches as `directive_window` |
| `end_state/game_state/milestones.json` | final milestone state (also tracked per-step inside `pre_state.milestones`) | exporter quality scoring |
| `end_state/game_state/skills.json` | final skill registry from the autoevolve loop | reference, not directly used |
| `end_state/game_state/subagents.json` | final subagent registry | reference |
| `end_state/game_state/memory.json` | long-term memory entries the agent wrote | reference |
| `end_state/videos/` | recording of the run (mp4 for Pokemon emulator, webm for browser) | showcase, debugging |
| `agent_scratch_space/` | objectives, ephemeral state | reference |

**Critical patch shipped 2026-04-08** (`agents/PokeAgent.py`): without
this, Pokemon trajectories have prompts and tool calls but no
screenshots, which makes them useless for vision distillation. The
patch saves the per-step PIL image as `screenshots/step_<step>.png`
right after the agent receives it from the harness.

## The trajectory exporter

`data/export_trajectories.py` joins everything above into a single
SFT-ready row per agent step. Schema (one JSONL row, see the docstring
of the file for the full spec):

```jsonc
{
  "schema_version": 1,
  "run_id":           "run_20260408_032015_ae_autoevolve",
  "step":             142,
  "image_path":       "screenshots/step_00142.png",
  "image_b64":        "...",
  "prompt":           "...",  // full system+user prompt sent to teacher
  "raw_response":     "...",  // teacher's raw text response
  "completion": {
    "reasoning":      "ANALYZE: ...",
    "tool_calls":     [{"name": "press_buttons", "args": {...}}]
  },
  "pre_state": {
    "location":       "ROUTE 101",
    "player_coords":  [12, 7],
    "context":        "overworld",
    "is_in_battle":   false,
    "dialog_active":  false,
    "milestones":     ["entered_lab", "talked_to_birch"]
  },
  "post_state": {
    "milestones":      ["entered_lab", "talked_to_birch", "left_lab"],
    "milestones_added": ["left_lab"]
  },
  "directive_window": {
    "window":         [110, 160],
    "evolved":        true,
    "raw":            { ... }
  },
  "model_info": {
    "model":          "gemini-3.1-pro-preview",
    "backend":        "gemini",
    "prompt_tokens":  8123,
    "completion_tokens": 142
  },
  "weight":           3.0,
  "weight_reasons":   ["preceded_milestone", "evolved_skill_used"],
  "filter_status":    "kept"
}
```

Filters applied (see `FilterConfig` in the file for tunable knobs):

**Trajectory-level (drop the whole run):**
- Reached fewer than 3 story milestones
- More than 30% of steps emitted no tool call
- Single contiguous run of identical actions longer than 50 steps
- (optional) `--teacher-model` mismatch

**Step-level (drop individual steps from kept runs):**
- Tool call missing or `success: false`
- Identical to last 3 steps (loop detection)
- Next 5 steps undid the action and the agent did move in between (regret detection)
- Reasoning lacks any specific game-state grounding (location, coords, button) and is shorter than 80 chars
- Image is mid-transition: low std AND high pixel-mean delta from previous frame

**Quality boosts (multiply sample weight, applied in training):**
- ×3 if the step immediately preceded a milestone increment
- ×2 if the directive window contains a `skill_update` or `skill_create`
- ×1.5 if the reasoning is grounded
- ×1 default

**Verified end-to-end on existing data:** running the exporter against
`run_data/run_20260407_023149` (a gemma4 Pokemon run, no screenshots)
yields a 41% step keep rate with 12 regret-detected drops, 3 no-tool-call
drops, and 2 ungrounded-reasoning drops. The filter is doing what we
expect.

Usage:

```bash
.venv/bin/python -m data.export_trajectories \
  --runs run_data \
  --output data/sft_dataset/v1 \
  --teacher-model gemini-3.1-pro-preview \
  --shard-size 2000
```

## Phase plan

### Phase 1 — Off-policy SFT from filtered Gemini Pro 3.1 trajectories

**Goal:** student emits structured tool calls in the same format as
Gemini, with similar reasoning, on the screen states Gemini visited.
"Gemini imitator" quality.

**Why this fixes the prose-format problem:** the e4b student's
documented failure mode is producing prose (`"PLAN: Action: Press A.
Reason: ..."`) instead of structured tool calls. Distilling Gemini's
`(reasoning, tool_call)` pairs jointly forces the student to learn
both — reasoning fills the prose channel, structured tool call fills
the function-call channel, and cross-entropy on the call tokens
punishes any other format.

**Steps:**

1. **Calibration run** ($15-30) — single 1000-step Gemini Pro 3.1
   run with the command above. Verify it picks a starter, exits
   Birch's Lab, reaches Route 101, and ideally hits at least one gym.
   This is the "is the teacher actually good enough" check before
   we spend the full budget.
2. **Run the exporter against the calibration trajectory** with
   `--dry-run` to see what fraction survives. Tighten or loosen the
   filters until ~50-70% of steps are kept.
3. **Collect the dataset** — 50-100 trajectories of ~1000 steps each.
   - 4 parallel emulator instances on the local box → ~75 hours of
     wall clock for 100 trajectories
   - Cost: ~$1500-3000 in Gemini API
4. **Export filtered SFT dataset** — `data/sft_dataset/v1/`,
   ~30K-70K examples after filtering.
5. **SFT training**:
   - Phase 1a (iteration): bf16 LoRA rank 256 on 1× 5090, ~4 hours.
     Use Unsloth. Validates dataset and hyperparams.
   - Phase 1b (production): bf16 full FT on 1× H200, ~12-16 hours.
     Use Axolotl. Loss = cross-entropy on completion tokens
     (reasoning + tool_call jointly), prompt masked. Sample weights
     applied from the exporter's `weight` field.
6. **Eval** — `eval/pokemon_eval.py` (TBD) runs the trained checkpoint
   on a held-out scenario set, reports tool-call dispatch rate,
   milestone progression, and step-time vs Gemini.

**Success criterion to advance to Phase 2:**
- ≥95% tool-call dispatch rate (was ~10% in baseline e4b)
- ≥70% of Gemini's milestone count on a 5-seed Pokemon eval
- ≤2× Gemini's wall time per step

If the SFT'd model stalls below 60% of Gemini's milestone count, the
dataset has problems — tighten filters before paying for Phase 2.

**Resources:** ~$1500-3000 Gemini API + 1× H200 for ~16 hours.

---

### Phase 2 — On-policy DPO with Gemini as preference judge

**Why DPO instead of reverse-KL:** the Thinking Machines recipe needs
teacher token logprobs and Gemini doesn't expose them. DPO solves the
same problem (align student to teacher's preferences without
distribution mismatch) using sequence-level preference labels: at
each state, ask Gemini "of these N candidates, which is better?" That
needs only the teacher's response, not its logprobs.

The student still visits its **own** state distribution during this
phase, which is the on-policy benefit that matters for avoiding
compounding error.

**Goal:** push the Phase 1 student past Gemini-imitation toward
Gemini-preferred-actions on the student's own visited states. Catch
up to Gemini's milestone progression on the eval.

**Steps:**

1. **Rollout pool** — N parallel Pokemon Emerald instances running
   the Phase 1 SFT'd student (loaded once via vLLM, shared across
   workers).
2. **Per-step candidate generation** — at each step, sample **K=4
   candidate `(reasoning, tool_call)` pairs** from the student at
   temperature 0.9. Pick one to actually advance the game (e.g. the
   first one) and save all K to a candidate pool.
3. **Gemini judges** — for each step, send Gemini the screenshot +
   prompt + the K candidates, ask it to rank them or pick the best
   one. **One Gemini call per agent step** (not per token).
4. **Construct DPO pairs** — `(state, chosen, rejected)` from
   adjacent ranks. With K=4, that's up to 6 pairs per state.
5. **DPO training** — TRL's `DPOTrainer`:
   - bf16 throughout
   - Reference model: frozen Phase 1 checkpoint
   - β = 0.1 (canonical starting point)
   - lr 5e-7, cosine schedule, warmup 50 steps
   - 1 epoch over the rated dataset
6. **Cycle** — every M=2000 new examples, run a 30-min DPO update,
   then continue rollouts with the new policy. ~5-10 cycles total.

**Optionally (if DPO plateaus): switch to expert-iteration / RSFT.**
Same data shape, but instead of DPO loss, just append `(state,
best_candidate)` to the SFT dataset and re-SFT periodically. Simpler,
sometimes more stable.

**Resources:**
- Rollout pool: 2× 5090 local (student inference)
- Gemini API: ~$2000-4000 for ~10-20K rated examples (~$0.20-0.40 per
  judgment, depends on prompt length)
- Training: 1× H200 for ~10 DPO cycles, ~30-60 H200-hours total
- Wall time: ~1-2 weeks of clock time

**Success criterion to advance to Phase 3:**
- ≥90% of Gemini's milestone count on the eval
- Gemini agrees with student picks ≥70% of the time on a held-out batch
- No regression on Phase 1's tool-call dispatch rate (must stay ≥95%)

---

### Phase 3 — GRPO on environment reward + autoevolve directives

**Goal:** push past frontier teacher quality. Phase 2 is bounded by
Gemini (you can't beat the judge by majority vote). Phase 3 has no
teacher — only the environment.

This phase no longer needs a frontier model at all.

**Reward signal — combine three sources:**

| Component | Source | Weight | Why |
|---|---|---|---|
| Milestone delta | `pokemon_env/hackathon_milestones.json` | +10 per new milestone | sparse but ground truth |
| Player movement reward | `player_coords` change toward objective | +0.1 per move, +0.5 if matched | denser, encourages exploration |
| Autoevolve directive density | `prompt_evolution/meta_prompts/*` for the trajectory window | +1 per skill/subagent edit | the harness post-hoc ratifies useful steps |
| Tool-call validity | dispatched + game state changed | +0.05 | format compliance |
| Loop penalty | identical to last 3 steps | −1 | prevents the failure mode we just fixed |
| Battle outcome | win/lose flag | +5 / −2 | dense feedback in the densest part of gameplay |

**Algorithm: GRPO from TRL.**
- Group size K=4 (4 rollouts per state for relative advantage)
- KL penalty against the Phase 2 checkpoint to prevent collapse
- Reward normalization within each group
- No value head (cheaper than PPO)

**Steps:**

1. **Rollout pool** — 2× 5090, ~16 parallel emulator instances,
   queue-based, refreshable policy weights.
2. **Reward function** (`training/rl/reward.py`) computes the composite
   reward per step from already-recorded trajectory data.
3. **Train** with TRL's `GRPOTrainer` on 4× H200 for a 72-hour run.
4. **Eval** every 1000 gradient steps against the held-out scenarios.

**Resources:**
- Rollout pool: 2× 5090 local
- Actor training: **4× H200, single 72-hour job**
- No Gemini API calls

**Success criterion:** student exceeds the Phase 2 checkpoint on the
eval. Stretch: meets or beats Gemini Pro 3.1 itself, validating the
"frontier-via-RL-from-distilled-base" thesis.

---

## Cross-cutting infrastructure

### Already built (2026-04-08):

- ✅ `data/export_trajectories.py` — trajectory exporter with full
  filter pipeline. Tested against existing run data, ~41% keep rate
  on a noisy gemma4 trajectory (which is the right ballpark).
- ✅ `agents/PokeAgent.py` screenshot patch — saves
  `screenshots/step_NNNNN.png` per step. Without this Pokemon
  trajectories are unusable for vision distillation.
- ✅ Trajectory schema documented above and in
  `data/export_trajectories.py` docstring.

### Still to build:

- ⏳ `eval/pokemon_eval.py` — runs N seeds × M scenarios against any
  checkpoint, reports milestone progression + tool-call rate +
  step-time vs Gemini. Required by all 3 phases for evaluation.
- ⏳ `training/sft/configs/gemma4_e4b_lora.yaml` — Unsloth config for
  Phase 1a iteration loop.
- ⏳ `training/sft/configs/gemma4_e4b_full.yaml` — Axolotl config for
  Phase 1b production.
- ⏳ `training/dpo/configs/gemma4_e4b_dpo.yaml` — TRL DPOTrainer
  config for Phase 2.
- ⏳ `training/rl/grpo_train.py` + reward function for Phase 3.
- ⏳ `data/judge_with_gemini.py` — Phase 2 judging script. Reads
  candidate pools, calls Gemini, writes DPO pairs.
- ⏳ Per-trajectory checkpoint registry —
  `(base_model, dataset_hash, hyperparam_hash, phase, eval_score)` so
  we can always answer "which checkpoint was best on eval set X?"
  without re-running anything.

## Total budget

| Phase | Compute | $ cost | Wall time |
|---|---|---|---|
| 1 — collection | Gemini API (~100 traj × $15-30) | $1500-3000 | ~3 days collection |
| 1 — training | 1× H200 ×16h | (H200 hours) | 16 hours |
| 2 — rollouts | 2× 5090 local | $0 | ~24 hours |
| 2 — Gemini judging | Gemini API (~10-20K judgments) | $2000-4000 | parallel with rollouts |
| 2 — DPO training | 1× H200 × 30-60h | (H200 hours) | 30-60 hours |
| 3 — rollout pool | 2× 5090 local | $0 | continuous |
| 3 — RL training | 4× H200 × 72h | (H200 hours) | 72 hours |
| **Total** | **~150 H200-hours + 2× 5090 continuous** | **$3500-7000** | **~2-3 weeks of clock time** |

## What unblocks the next concrete action

The single most useful thing to do right now is **the calibration
run** — one 1000-step Gemini Pro 3.1 trajectory through the exporter.
That gives us:

1. Real evidence that Gemini Pro 3.1 actually clears Birch's Lab and
   reaches a meaningful milestone. (It might not — we have no Gemini
   Pro Pokemon data on disk yet, and that's the foundational
   assumption.)
2. The actual filter survival rate against a real teacher trajectory
   (vs the noisy gemma4 baseline we tested with).
3. A representative example of what one row of the SFT dataset will
   look like in practice.
4. Validation that the screenshot patch works end-to-end through the
   exporter into a usable example.

Estimated cost: **$15-30** for one 1000-step run.
Estimated time: **~3 hours** of agent wall clock.

After that, the next decision is whether to scale up Phase 1 to the
full $1500-3000 collection budget or to revise the plan based on what
the calibration trajectory taught us.

---

*Status as of 2026-04-08: planning complete, infrastructure partially
shipped, calibration run not yet executed. This file is the source of
truth for the project; update it as decisions change.*
