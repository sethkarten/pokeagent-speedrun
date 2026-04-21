# Training roadmap — autoevolve harness across many games

> Captured 2026-04-07. The plan for turning the autoevolve harness from a
> runtime-only system into a training pipeline that produces small,
> generalist game-playing agents.
>
> Inspired by:
> - **MolmoWeb-8B** (AI2, Molmo2 architecture: Qwen3-8B + SigLIP 2) —
>   open multimodal web agent trained as a complete browser-action
>   planner. Outperforms Fara-7B / UI-Tars-1.5-7B / Holo1-7B at similar
>   scale. We use it as a **click oracle** (Pattern A): wrap it in a
>   single-step `find_element(description)` call and parse the
>   coordinates out of its `ACTION:` line. Throws away most of its
>   capability but gets us perfect click coordinates for the price of
>   one inference call.
> - **OpenClaw-RL** (Wang et al., arxiv 2603.10165) — agentic-RL framework
>   that recovers next-state signal from every action and uses
>   "Hindsight-Guided On-Policy Distillation (OPD)" to turn evaluative +
>   directive feedback into training data
>
> The autoevolve harness already does the hard part (directive signal
> generation via skill/subagent evolution from trajectory analysis). We've
> been throwing that signal away as runtime mutations instead of using it
> as training data. This plan fixes that.

---

## North star

Produce a fine-tuned **gemma4:e4b** (4B dense) agent that can play
**arbitrary browser games and Pokemon-class emulator games** at the
quality of a frontier API model, without per-game prompt engineering.

Concretely the success criterion is: a fine-tuned gemma4:e4b checkpoint
that on a held-out game library matches or beats gemma4:26b's
end-to-end task completion rate while running 3-5× faster on a single
consumer GPU.

## Model family constraint

The student is **gemma4:e4b** (gemma4-only family commitment for the
student). The teacher is **per-game-class**: Gemini Pro 3.1 is the
default for Pokemon/RPG, gemma4 variants are the default for browser
games, and we evaluate whether gemma4 is good enough to *replace*
Gemini for Pokemon during Step 0.5.

The three gemma4 variants we care about:

| Model | Total params | Active | Disk Q4 | bf16 inference VRAM | Full FT VRAM (bf16) |
|---|---|---|---|---|---|
| **gemma4:e4b** | 4B dense | 4B | 4 GB | 8 GB | ~48 GB |
| **gemma4:26b** | 26B MoE | 4B | 17 GB | ~52 GB | ~310 GB |
| **gemma4:31b** | 31B dense | 31B | 20 GB | ~62 GB | ~370 GB |

**MoE training trap:** at inference time, 26b only activates 4B params
per token, but at training time the optimizer state has to cover all
26B params because any token can route to any expert. Active-param
count doesn't reduce training memory at all — only inference cost.

**Roles:**
- **Student (SFT target):** gemma4:e4b. Smallest, fastest, easy to ship,
  trains on a single H200. gemma4-only family commitment.
- **Primary teacher for Pokemon / RPG / complex tool-calling games:**
  **Gemini Pro 3.1** (`gemini-3-pro`). Native function-calling API,
  reliably produces structured tool calls, handles the dense 20-tool
  PokeAgent scaffold without prompt engineering. This is the default
  Pokemon teacher and the SFT dataset will be built primarily from its
  trajectories unless gemma4 proves capable of replacing it.
- **Primary teacher for browser games:** gemma4:26b paired with
  MolmoWeb-8B as a click oracle (see Step 0.1). Local, no API cost,
  shares a tokenizer with the student.
- **Cost-saving candidate to replace Gemini for Pokemon:** gemma4:26b.
  Already shows ~90% tool-call success rate on Pokemon in our smokes
  via the text-action extractor path. If Step 0.5 shows it matches
  Gemini's quality on the Pokemon eval suite, we swap it in for the
  Pokemon teacher and save the API budget. If not, Gemini stays.
- **Research alternative teacher:** gemma4:31b dense. Strongest gemma.
  Try as a teacher quality lift if 26b is close to Gemini but not
  quite there. Smoke-test for tool-calling reliability needed before
  committing.

**No quantization during training.** bf16 base, bf16 activations, fp32
optimizer state. QLoRA (4-bit base + LoRA) hurts downstream tool-call
quality by ~1-3 points on benchmarks and is worse for precision-
sensitive tasks like structured tool-call format. The one exception
worth considering: **fp8 mixed-precision training on H200**, which is
hardware-accelerated and ~free quality-wise on Hopper.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 0  ·  EXPERT ORACLE TIER                │
│                                                                  │
│   browser puzzles       Pokemon / RPG          real-time games  │
│   ─────────────────     ─────────────────      ───────────────  │
│   gemma4:26b  +         Gemini Pro 3.1         gemma4:26b  +    │
│   MolmoWeb-8B oracle    (primary teacher)      tuned key/wait   │
│                         gemma4:26b candidate                    │
│                         to replace Gemini if                    │
│                         it passes Step 0.5                      │
│         ↓                      ↓                      ↓         │
│            ┌───────────────────┴────────────────────┐           │
│            │   STEP 0.5: TEACHER QUALITY EVAL        │           │
│            │   (does gemma4:26b match Gemini on the  │           │
│            │    Pokemon suite? if yes → save API $)  │           │
│            └────────────────────┬───────────────────┘           │
│                                 ↓                                │
│            ┌────────────────────┴───────────────────┐           │
│            │   trajectory collector (slow but        │           │
│            │   correct expert demonstrations)        │           │
│            └────────────────────┬───────────────────┘           │
└─────────────────────────────────┼────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1  ·  TRAJECTORY DATASET                       │
│   (state, prompt, valid_tool_call) JSONL across all game types  │
│                + autoevolve directive signals                    │
└─────────────────────────────────┬────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2  ·  SFT DISTILLATION (gemma4:e4b)            │
│         2a: LoRA rank 256 bf16 on 1× 5090 (iteration loop)      │
│         2b: full FT bf16 on 1× H200 (production checkpoint)     │
│         No quantization. fp8 only on H200 if available.         │
└─────────────────────────────────┬────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3  ·  ONLINE RL REFINEMENT                     │
│      OpenClaw-RL style: live rollouts → next-state signal →     │
│      directive + evaluative judges → on-policy distillation     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 0 — Expert oracle tier

**Objective.** Build a multi-tier expert system that can collect *slow
but correct* trajectories across the three game classes we care about.
Quality matters more than speed at this step — these trajectories are
the SFT teacher data for everything downstream. A bad teacher caps
every later phase.

The trick is that no single model is good enough for all game classes
yet, so we use the *right tool for each game*:

### 0.1 Browser puzzle games — gemma4:26b + MolmoWeb-8B click oracle

**Games.** Folder Dungeon, Brackeys jam HTML5 games, point-and-click
adventures, anything where the action space is "click the right pixel".

**Why this combination.** Gemma4:26b is *good at planning* ("I should
click the START button to begin the game") but *bad at coordinates*
(it hallucinates pixel locations in canvases it can't pixel-grid).
MolmoWeb-8B is a complete web agent (Qwen3-8B + SigLIP 2 base) trained
to take a goal + screenshot and produce structured `THOUGHT:` /
`ACTION:` next steps with click coordinates. We use it as a **click
oracle (Pattern A)**: wrap it in a single-step `find_element`
formulation, parse the coordinates out of its `ACTION:` line, throw
away the rest. The model is doing more than we ask, but the result is
correct click coordinates and we keep gemma4:26b in charge of the
planning loop.

**Architecture.**

```
agent step:
  1. screenshot + prompt → gemma4:26b (planner)
  2. gemma decides: "click the START button"
  3. NEW TOOL: find_element(description="START button")
     → MolmoWeb-8B serves the screenshot + a synthetic single-step
       task ("Click the START button")
     → MolmoWeb returns "THOUGHT: ... ACTION: click(x=480, y=455)"
     → server parses (x, y), discards thought
     → returns (x, y) in canvas-relative coords
  4. harness dispatches mouse_click(x, y)
  5. server captures next screenshot
```

**Implementation work.**

- New MCP tool `find_element(description, screenshot=None)` that calls
  the molmoweb_server with the current screenshot + the agent's
  natural-language description and returns `{x, y, confidence}`.
- New helper `click_element(description)` that composes find_element +
  mouse_click in one tool call so the planner doesn't have to do the
  two-step dance.
- `molmoweb_server.py`: standalone process, separate port from Ollama
  (probably 11436), HTTP `/find_element` endpoint. Loads MolmoWeb-8B
  via HF transformers, applies the chat template, parses the
  `ACTION:` line for coordinates.
- Update browser_game_agent prompt to prefer `click_element` over raw
  `mouse_click` when the target has a name.

**Resources.**

| Setup | VRAM | Fits on 5090? | Notes |
|---|---|---|---|
| **fp32** (model card recommendation) | ~36 GB | ❌ | Doesn't fit on a 32 GB 5090; needs an H200 or two 5090s with model parallelism |
| **bf16** (most likely good enough) | ~18 GB | ✅ | Fits on a single 5090 with ~14 GB headroom for activations + KV cache. **Default.** |
| Q4 (bnb-4bit) | ~5 GB | ✅ | Quantization is what we're avoiding for training, but for *inference* Q4 may be acceptable if accuracy holds |

- **Layout:** MolmoWeb-8B in bf16 on **GPU 0**, gemma4:26b on **GPU 1**.
  Each card has one model. Validate bf16 vs fp32 on a held-out set of
  ~20 game screenshots before committing — if bf16 matches fp32 on
  click accuracy, we ship bf16.
- Inference ≈ **300-800 ms per find_element call** on a 5090 (one image
  + short prompt + ~30-50 token decode for the `ACTION:` line). Real
  number depends on whether the model emits a long thought before the
  action — we may want to set max_new_tokens low.
- No training in this step — MolmoWeb is used as-is.

**Libraries.**
- `transformers` (HF) — MolmoWeb released as a HF model, standard
  AutoModelForImageTextToText interface
- `torch` with cuda
- `pillow` for image handling
- `accelerate` for `device_map="auto"`
- `jinja2` for the chat template
- (optional later) `vllm` or `sglang` — for serving MolmoWeb with
  batching if we want to drive >1 game in parallel. Not needed for the
  initial smoke.
- (intentionally NOT) `bitsandbytes` — we are avoiding quantization for
  training; for inference we may revisit if VRAM gets tight

**Success criteria.** On Folder Dungeon, the agent reaches Floor 2
within 100 steps in ≥80% of runs. On a held-out itch.io game (e.g.
Bracket Boy), reaches the first interactive screen in <20 clicks.

### 0.2 Pokemon / complex tool-calling games — Gemini Pro 3.1 primary, gemma4:26b candidate

**Games.** Pokemon Emerald, Pokemon Red, any game with a dense
PokeAgent-style tool surface (20+ tools, structured argument formats,
nested kwargs).

**Primary teacher: Gemini Pro 3.1** (`gemini-3-pro`). Pokemon and other
RPG/tool-heavy games are the hardest game class for tool calling — the
20-tool PokeAgent scaffold needs structured `tool_calls` with nested
arguments, the game state is dense, and the long-horizon planning
requirement (Story Mode → Route 101 → first gym etc.) compounds any
weakness in the policy. Gemini Pro 3.1 natively uses the function-
calling API, was trained on this exact tool-call format, and we already
verified it works on Pokemon Emerald earlier on this branch.

This is the default Pokemon teacher and the **SFT dataset will be built
primarily from its trajectories** unless gemma4 proves capable of
replacing it during Step 0.5.

**Cost-saving candidate: gemma4:26b.** Our smokes showed gemma4:26b
dispatches valid tool calls at ~90% success rate on Pokemon (mix of
structured `tool_calls` and text-format `ACTION: press_buttons(...)`
lines that the text-action extractor we shipped catches). That's
encouraging but unproven on long-horizon Pokemon scenarios — the smoke
was 30 steps. If Step 0.5 shows 26b is competitive with Gemini on the
Pokemon eval suite (≥80% of Gemini's success rate on the same set of
tasks), we **swap 26b in as the Pokemon teacher and save the API
budget**. If not, Gemini stays.

The gemma4:31b dense variant is in the bench as a quality lift in case
26b is *close* but not quite — see Step 0.5 decision tree.

**Architecture.**

- Existing PokeAgent scaffold (`--scaffold autoevolve`,
  `--backup-state Emerald-GBAdvance/auto-evolve_init.zip`)
- Primary backend: gemini (`gemini-3-pro`)
- Candidate backend: ollama with `gemma4:26b` on local GPU (optionally
  `gemma4:31b` as the research alternative)
- All branches run from the autoevolve scaffold variant with prompt
  optimization enabled so the trajectories also include the directive
  signal that the harness evolver generates
- All branches feed the same trajectory exporter — the dataset has an
  `oracle_model` column so we can separate / mix them at training time

**Implementation work.** Mostly already done — both backends exist and
work. The remaining piece is the **trajectory exporter**: scrape every
successful tool call from the run trajectories and emit them as
SFT-ready pairs. Details under "Step 1: Trajectory dataset" below.

**Resources.**

*Primary (Gemini Pro 3.1):*
- API only — no local GPU needed for the oracle itself
- Cost: Gemini Pro 3.1 ≈ ~$3 per million input tokens, ~$15 per million
  output. A 1000-step Pokemon run ≈ ~3M input tokens (prompt + image
  per step) + ~150K output tokens. **~$10-15 per 1000-step run.**
- **Pokemon dataset budget:** ~50 1000-step trajectories spanning the
  game from start to first few gyms = **~$500-750 baseline budget**.
  This is the default spend assumption for the plan.
- Mitigation: start with 5 trajectories to validate Gemini's quality
  on the eval suite before committing the full budget.

*Cost-saving candidate (gemma4:26b local):*
- 1× 5090 per game instance (16 GB VRAM Q4 for the model + a few GB for
  the harness)
- ~8.5 s/step measured on a real Pokemon run, so a 1000-step trajectory
  takes ~2.5 hours of wall clock
- We can run multiple game instances in parallel — easily 4-8 on the
  2× 5090 setup if we keep one model copy per GPU and serialize requests
- Zero $ cost
- If Step 0.5 says this works, we **save the entire ~$500-750 Pokemon
  Gemini budget** and the dataset is collected fully locally

**Libraries.**
- `google-genai` (already in the repo)
- `ollama` Python client (already wired)

**Success criteria.**

*For Gemini Pro 3.1 (primary):* On Pokemon Emerald from
`auto-evolve_init.zip`, picks a starter Pokemon, exits Birch's Lab, and
reaches Route 101 in <300 steps in ≥80% of runs. Reaches the first gym
in <2000 steps in ≥50% of runs. (This is the bar the gemma4 candidates
have to match in Step 0.5 to displace Gemini.)

*For gemma4:26b (candidate):* In Step 0.5, achieves ≥80% of Gemini's
success rate on the same set of Pokemon scenarios. If yes → 26b
replaces Gemini for Pokemon and saves the budget. If close but
short → try 31b. If neither matches → Gemini stays as primary and the
$500-750 Pokemon budget is committed.

### 0.3 Real-time games — Flappy Bird and friends

**Games.** Flappy Bird (current pain point), other twitch / arcade
games where game time matters.

**Why this is hard.** Real-time games require:
1. **Precise timing** — the bird falls 200ms/frame; pressing Space too
   late means death
2. **Hold-and-release loops** — `key_down` / `wait_ms` / `key_up` patterns
   that gemma doesn't naturally generate
3. **Stable frame capture** — the game state at screenshot time must
   reflect the same frame the agent acts on, otherwise you train on
   stale data

We have the *primitives* (`key_down`, `key_up`, `wait_ms`, virtual time
shim) but the *prompting* and *step budget tuning* aren't there yet.

**Architecture.**

- Tier A — **gemma4:26b + tuned step budget + portrait viewport**: shrink
  the browser viewport so flappybird.io doesn't render letterbox bars
  (which currently confuse coordinate-based clicks). Bump
  `BROWSER_STEP_BUDGET_MS` from 200 to 500-1000ms so each action has time
  to actually render before screenshot. Update the Flappy-specific
  prompt directive to teach the press-and-wait pattern explicitly.
- Tier B — **scripted hand-tuned baseline**: a hardcoded Flappy Bird
  bot that we *know* plays well. Used as ground truth for the
  trajectory dataset and as a sanity floor — if our learned agent
  is worse than the scripted bot, we have a bug.

**Implementation work.**

- Per-game viewport override (env var: `BROWSER_VIEWPORT_WIDTH`,
  `BROWSER_VIEWPORT_HEIGHT`). For Flappy Bird, set 480x720.
- Per-game step budget override.
- Per-game prompt directives — small markdown files like
  `agents/prompts/browser-game-directives/games/flappybird.md` that get
  appended to the base prompt when the URL matches.
- Scripted Flappy bot in `agents/scripted/flappybird.py` — uses the
  same MCP API but follows a deterministic policy. Useful as both
  teacher data source AND baseline.

**Resources.**
- Local 2× 5090 (already running)
- No new training in this step

**Libraries.** None new.

**Success criteria.** gemma4:26b oracle scores ≥10 on Flappy Bird in
≥50% of runs (~10 pipes cleared). Scripted bot scores ≥30 reliably
(used as upper bound for sanity).

### 0.4 Game library + reward signals

**Objective.** Define a fixed game library spanning the three game
classes and a per-game reward signal so we can (a) evaluate consistently
and (b) provide the evaluative signal for OpenClaw-RL in step 3.

**Library (initial, ~10 games).**

| Class | Game | Reward signal |
|---|---|---|
| Browser puzzle | Folder Dungeon (itch) | Floor depth + items collected |
| Browser puzzle | Bracket Boy / TBD jam game | Levels completed |
| Browser puzzle | A Dark Room (textgame.dev) | Stages reached |
| Browser real-time | Flappy Bird | Pipes cleared |
| Browser real-time | Slither.io / agar.io (TBD) | Mass / survival time |
| Browser arcade | Pacman (HTML5) | Score |
| Pokemon RPG | Emerald | Story milestones (objective list) |
| Pokemon RPG | Red | Story milestones |
| Visual novel | Doki Doki Literature Club Web | Choice tree depth |
| Walking sim | Florence (HTML5) | Chapter progression |
We have a large number (100+) game jam games that we can leverage for this.

**Reward signal extraction.** For browser games, scrape from the page:
either the canvas-rendered score (OCR or VLM judge) or a JS
`window.__score` poll. For Pokemon, the existing milestone system in
`pokemon_env/` already provides this. For visual novels, page text
diff.

**Implementation work.**

- `eval/game_library.yaml` — declarative game list with URL, viewport,
  step budget, reward extractor function name
- `eval/reward_extractors/*.py` — one extractor per reward type
- `eval/run_eval.py` — driver that runs N seeds × M games and reports a
  scoreboard

**Resources.** Local only.

**Libraries.** Existing (Playwright, PIL, requests).

### 0.5 Teacher quality decision point

**Objective.** Establish the **Gemini Pro 3.1 baseline** as the
reference for the Pokemon teacher, then evaluate whether **gemma4:26b**
(or **31b**) can match it well enough to displace it and save the
~$500-750 Pokemon API budget. This is a *cost-saving* gate, not a
go/no-go gate — Gemini wins by default unless gemma4 proves itself.

**The evaluation matrix.** Run a fixed eval suite (~5 runs per game,
~200-500 steps each) for each candidate teacher and score on:

| Metric | Why it matters |
|---|---|
| Tool-call dispatch rate (% of steps that emit a valid tool call) | Format compliance — the SFT student needs valid format examples |
| Game-specific success rate (milestones / score / floor) | Did the agent actually accomplish the goal? |
| Mean steps to milestone | Efficiency — wasteful trajectories pollute the dataset |
| Self-consistency (% of runs that reach the same milestone) | Are the trajectories representative or just lucky? |
| Cost per successful trajectory ($ or GPU-hours) | Whether saving the API budget is actually worth the local GPU time |

**Candidates and what we test.**

| Teacher | Tested on | Role |
|---|---|---|
| **Gemini Pro 3.1** | Full game library (Pokemon + browser puzzles + real-time) | **Reference baseline.** Whatever it scores defines the bar gemma4 has to meet to displace it. |
| **gemma4:26b** | Same suite, head-to-head against Gemini | Cost-saving candidate for Pokemon. Already the primary teacher for browser puzzles + real-time games per Step 0.1 / 0.3. |
| **gemma4:31b** (smoke first) | Subset: 1 browser puzzle + Pokemon Birch's Lab + Flappy Bird | Quality lift if 26b is *close* to Gemini but not quite. Smoke test for tool-calling reliability before committing to a full eval. |

**Decision tree (Pokemon teacher selection).**

```
Step 0.5 evaluation results — does gemma4 displace Gemini for Pokemon?
            │
            ├── gemma4:26b ≥80% of Gemini's success rate AND
            │   ≥80% of Gemini's tool-call dispatch rate
            │       → 26b becomes Pokemon teacher
            │       → Save ~$500-750 Gemini budget
            │       → Dataset built fully locally
            │
            ├── gemma4:26b is close (60-80% of Gemini) but not enough
            │       → Try 31b on the Pokemon eval
            │       → If 31b clears 80%, 31b is the teacher
            │       │       (note: production student is still e4b,
            │       │        so 31b only matters as a teacher)
            │       → If 31b doesn't clear 80% either, fall through
            │
            ├── gemma4:26b <60% of Gemini AND 31b doesn't recover it
            │       → Gemini Pro 3.1 stays as Pokemon teacher
            │       → Commit ~$500-750 Pokemon API budget
            │       → 26b still teacher for browser puzzles + real-time
            │       → Mixed dataset: ~70% Gemini-Pokemon, ~30% gemma4-other
            │
            └── Browser puzzles fail across ALL gemma4 teachers
                    → MolmoWeb click oracle needs more work (revisit 0.1)
                    → Or fall back to Gemini for browser puzzles too
                      (cost rises sharply — Gemini for the whole library
                      could be $1500-2500)
                    → Escalation: consider whether the SFT student
                      needs to be larger than e4b
```

**Default assumption.** Until Step 0.5 runs, **assume Gemini is the
Pokemon teacher** in all downstream planning. The cost line is real
and committed unless gemma4 displaces it.

**Implementation work.**

- `eval/teacher_eval.py` — runs the matrix above and emits a comparison
  table to W&B + a markdown summary in `eval/results/teacher_eval_<date>.md`
- Cache trajectories from this eval — they're the first batch of SFT
  data anyway, so the eval pays for itself

**Resources.**
- Local 2× 5090 for 26b and 31b runs (~6-12 hours of wall clock for the
  full matrix, parallelized)
- **~$100-150 of Gemini budget for the baseline eval** (5 runs ×
  ~200-500 steps each = a meaningful but modest spend to establish the
  reference number Pokemon-only)

**Success criteria.** A clear written decision in
`eval/results/teacher_eval_<date>.md` that names the Pokemon teacher
(Gemini, 26b, or 31b) and projects the SFT dataset composition (what
fraction comes from which model).

---

## Step 1 — Trajectory dataset

**Objective.** Convert oracle trajectories into a clean, deduplicated,
multi-game SFT dataset.

**Format.** JSONL, one example per agent step, schema:

```jsonc
{
  "game_id": "folder_dungeon",
  "game_class": "browser_puzzle",
  "trajectory_id": "run_20260407_023149",
  "step": 14,
  "screenshot_b64": "...",          // 960x576 PNG, base64
  "screenshot_resized_b64": "...",  // 448x448 for training-time eff
  "prompt": "...",                  // full system + user prompt
  "page_text": "...",               // visible DOM text (browser)
  "game_state": { ... },            // structured state (Pokemon coords etc)
  "tool_calls": [
    {
      "name": "click_element",
      "args": {"description": "START button"},
      "result": {"x": 480, "y": 455},
      "succeeded": true
    }
  ],
  "next_screenshot_b64": "...",     // for OpenClaw-RL next-state signal
  "next_game_state": { ... },
  "reward_delta": 0,                // +1 if score increased
  "evolver_directives": [           // from autoevolve loop
    {"type": "skill_update", "id": "navigate_lab", "diff": "..."}
  ],
  "oracle_model": "gemini-3-pro",
  "oracle_temperature": 0.7,
  "timestamp_ns": 1775541878789
}
```

**Filtering rules.** A step enters the SFT set only if:
1. At least one tool call was emitted (no empty-response steps)
2. No `error` field in any tool result (no failed dispatches)
3. The trajectory it came from achieved nonzero reward (we don't
   want to train on noise)
4. Deduplicate near-identical (state, action) pairs across runs to
   prevent the model from memorizing rare loops

**Implementation work.**

- `data/export_trajectories.py` — reads `run_data/run_*/trajectory_history.jsonl`
  and `llm_logs/llm_log_*.jsonl`, joins them on step, applies filters,
  emits sharded JSONL
- `data/dataset_card.md` — provenance, model used, game distribution,
  total examples, license notes
- Storage: HuggingFace datasets format on disk (parquet shards) +
  optional push to a private HF dataset repo

**Resources.** Local CPU only. Disk: ~1-5 GB per 1000-step run × 50
runs ≈ **50-250 GB** of trajectories. Need to think about whether to
keep the raw screenshots or just the resized 448×448 versions.

**Libraries.**
- `datasets` (HuggingFace)
- `pyarrow` for parquet
- `Pillow` for image resizing

**Target.** ~50K-100K (state, tool_call) examples by the end of
Step 1, evenly distributed across game classes.

**Success criteria.** Dataset loads cleanly into HF `datasets` format,
has a basic statistics dashboard (per-game count, per-tool count, image
size distribution), and can be sampled into a SFT-ready
`(prompt, image, completion)` triple in <10 lines of code.

---

## Step 2 — SFT distillation (gemma4:e4b)

**Objective.** Fine-tune **gemma4:e4b** on the multi-game trajectory
dataset so it produces structured tool calls in the same format as the
teacher, across all game classes, without per-game prompt engineering.

**Single target.** No model bake-off — we are committed to gemma4:e4b
as the student. The reasons are: (a) shared tokenizer / vision encoder
with the teacher (gemma4:26b), so distribution shift is minimal, (b)
smallest gemma4 variant, which means fastest local inference and the
easiest model to iterate on, (c) already integrated end-to-end in our
Ollama harness and tested (the failure mode is well-understood — see
"the e4b prose-format problem" findings from the smokes earlier on this
branch).

### 2a — LoRA iteration loop

**Why LoRA first.** Fast iteration on dataset quality, hyperparams, and
eval. LoRA is genuinely cheap on a 4B model — fits a single 5090 with
room to push rank up to 512. The point of this phase is *not* to ship
a production model; it's to validate the dataset and hyperparams
before burning H200 hours on full FT.

**Hyperparameters (starting point).**
- Base: gemma4:e4b at **bf16** (no quantization, no QLoRA)
- LoRA rank 256, alpha 512, applied to all attention + MLP projections
  (targeted modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj)
- Optimizer: AdamW, lr 2e-4 with cosine schedule, warmup 100 steps
- Effective batch size 32 (grad accumulation as needed)
- Sequence length 8192 (most prompts are 4-6K tokens; 8K leaves headroom)
- Image resolution 448×448
- 1-3 epochs over the SFT dataset
- Loss: cross-entropy on completion tokens only (mask the prompt)
- Gradient checkpointing on (saves activation memory at ~20% throughput cost)

**Memory math (bf16 LoRA rank 256 on gemma4:e4b):**
- Frozen weights: ~8 GB
- Adapter weights: ~150 MB
- Adapter gradients: ~150 MB
- AdamW state on adapter only: ~600 MB
- Activations (seq 8K, batch 4, with grad ckpt): ~5-8 GB
- **Total: ~15-18 GB** → fits trivially on 1× 5090 (32 GB)

**Hardware.** **1× 5090 local.** No need for the second card or H200s.
Iteration is the goal — short feedback loops matter more than
throughput. With gradient checkpointing and bf16, a 1-epoch pass over
50K examples takes ~2-4 hours.

**Decision criterion.** If LoRA hits ≥80% of teacher performance on
the eval game library, the dataset is good and we proceed to 2b for
the final production checkpoint. If LoRA stalls below 60%, debug the
dataset (filtering rules, balance across game classes, label noise)
*before* spending H200 hours on full FT.

### 2b — Full fine-tuning production run

**Why full FT for production.** The e4b prose-format failure mode is
deeply internalized from base training, and LoRA's low-rank
perturbations may not be sufficient to fully overwrite it. Full FT is
the safer bet for the production checkpoint we ship — and at 4B
parameters, it's cheap enough to be worth it.

**Hyperparameters.**
- Same dataset, same eval, same epochs as 2a
- Base: gemma4:e4b at **bf16** (no quantization)
- Optimizer: AdamW, lr 5e-5 (lower than LoRA — full FT is more
  sensitive), cosine schedule, warmup 200 steps
- Effective batch size 16-32
- Sequence length 8192
- Gradient checkpointing on
- **fp8 mixed precision** if running on H200 (Hopper-native, ~free
  quality, ~30% throughput boost)
- 1-3 epochs

**Memory math (full FT bf16 on gemma4:e4b):**
- Weights: 8 GB
- Gradients: 8 GB
- AdamW state (fp32 m + v): 32 GB
- Activations (seq 8K, batch 8, grad ckpt): ~15-25 GB
- **Total: ~63-73 GB**

**Hardware.**

| Setup | Fits? | Wall time est. | Notes |
|---|---|---|---|
| **1× H200 (141 GB)** | ✅ trivially | 8-16 hours | The clean choice — single node, no DeepSpeed config to maintain. ~70 GB used / 141 GB. Plenty of headroom for batch size 16+. |
| 2× H200 (DDP) | ✅ | 4-8 hours | Same memory footprint per card, DDP for wall-time reduction. Use this if H200 hours are cheap. |
| 2× 5090 (ZeRO-2) | ✅ tight | 16-30 hours | Optimizer sharded across both cards (~32 GB/card). Workable but tight; need small batch + grad ckpt. Use only if H200 unavailable. |
| 2× 5090 (ZeRO-3) | ✅ comfortable | 18-32 hours | Everything sharded (~24 GB/card). More headroom than ZeRO-2 at slightly worse throughput. |

**Recommended path:** **1× H200, 1 × 72-hour job.** Trains the production
checkpoint in 8-16 hours and leaves the rest of the budget for
hyperparameter sweeps or fp8 vs bf16 ablation.

**Implementation work (shared 2a + 2b).**

- `training/sft/configs/gemma4_e4b_lora.yaml` — Unsloth config for 2a
- `training/sft/configs/gemma4_e4b_full.yaml` — Axolotl config for 2b
- `training/sft/data_loader.py` — converts our JSONL trajectory format
  to ShareGPT-style messages with images
- `training/sft/train.py` — wrapper script with W&B logging,
  checkpointing, end-of-epoch eval
- `training/sft/eval.py` — runs the trained model against
  `eval/run_eval.py` game library, reports per-game success rate vs
  the teacher and vs the base model
- `training/sft/merge_to_ollama.py` — merge LoRA adapter (2a) or pack
  full FT checkpoint (2b) into a GGUF that Ollama can load. Validates
  the trained model can serve via the same harness it was trained on.

**Libraries.**

- **`unsloth`** — fastest LoRA trainer, native gemma4 support, ~2× HF
  TRL speed. **Primary for 2a.**
- **`axolotl`** — config-driven, mature support for full FT with
  DeepSpeed integration. **Primary for 2b.**
- **`trl`** (HuggingFace) — `SFTTrainer` as a fallback if Unsloth or
  Axolotl have bugs with our specific config
- **`peft`** — LoRA implementation, used by all of the above
- **`deepspeed`** — for ZeRO sharding if we run 2b on the 2× 5090 setup
- **`transformers`** — base
- **`flash-attn`** — required for fast training at long context
- **`accelerate`** — multi-GPU launcher
- **`wandb`** — logging
- **NOT used:** `bitsandbytes` (we are intentionally not quantizing)

**Success criteria.**
- **2a (LoRA):** Fine-tuned model achieves ≥80% of teacher performance
  on the eval game library, AND ≥95% tool-call dispatch rate (vs the
  ~10% e4b baseline). Validates the dataset.
- **2b (full FT):** Achieves ≥90% of teacher performance, AND ≥98%
  tool-call dispatch rate. This is the production checkpoint.

If 2b doesn't beat 2a by a meaningful margin, that's also a useful
result: it means LoRA was sufficient and we save the H200 hours next
time.

---

## Step 3 — Online RL refinement (OpenClaw-RL style)

**Objective.** Push the SFT'd model past oracle performance by closing
the loop: agent acts, harness observes the next state, judges generate
evaluative + directive signals, model trains on its own rollouts.

**Why now and not earlier.** SFT alone caps at oracle quality. To beat
the oracle, the model needs to learn from its *own* mistakes — which
requires a closed-loop training system. OpenClaw-RL provides exactly
this architecture: asynchronous live-serving + judging + policy update.

**Architecture (mirrors OpenClaw-RL paper).**

```
       ┌─────────────────────────────────────────────────────┐
       │                   ROLLOUT POOL                       │
       │  N parallel game instances, each running the         │
       │  current policy. Each step emits:                    │
       │     (state, action, next_state, reward_delta)        │
       └───────────────────────┬─────────────────────────────┘
                               ↓
       ┌───────────────────────┴─────────────────────────────┐
       │                    JUDGE TIER                        │
       │  Two judges run async on rollout transitions:        │
       │  • Evaluative (PRM): scalar reward from              │
       │    next_state delta + reward_delta + game milestones │
       │  • Directive: textual hint for improvement,          │
       │    generated by Gemini Pro 3.1 (the same way         │
       │    autoevolve already does it)                       │
       └───────────────────────┬─────────────────────────────┘
                               ↓
       ┌───────────────────────┴─────────────────────────────┐
       │              HINDSIGHT-GUIDED OPD UPDATE             │
       │  Token-level directional advantage from the          │
       │  directive signal + scalar advantage from the        │
       │  evaluative signal. Gradient step on the actor.      │
       └───────────────────────┬─────────────────────────────┘
                               ↓
       ┌───────────────────────┴─────────────────────────────┐
       │     UPDATED POLICY WEIGHTS BACK TO ROLLOUT POOL      │
       │     (live, no rollout-pool restart)                  │
       └─────────────────────────────────────────────────────┘
```

**The autoevolve harness IS the directive signal source.** The
evolver's `_evolve_skills` / `_evolve_subagents` / `_evolve_memory`
passes already do exactly what OpenClaw's directive judge does:
analyze a window of trajectory and generate textual hints for
improvement. We just need to:

1. Capture those hints as training signal (currently we throw them
   away as runtime mutations)
2. Convert them into a format OPD can consume (token-level advantage)

**Implementation work.**

- Adopt `OpenClaw-RL` upstream as a git submodule:
  `https://github.com/Gen-Verse/OpenClaw-RL`
- Adapter layer: feed our trajectory format into their training loop
- `training/rl/reward_models/` — game-specific PRM judges (one per
  game, or one universal "did this action progress the game?" judge)
- `training/rl/directive_judge.py` — wraps the existing
  `harness_evolver._evolve_skills` so its output becomes a training
  signal instead of just a runtime mutation
- `training/rl/rollout_pool.py` — async N-game rollout driver. Probably
  reuses the existing run_browser.sh / run.py paths but parameterized
  for parallel rollouts
- W&B integration for live training curves

**Resources.**

| Workload | Recommended | Notes |
|---|---|---|
| Rollout pool (16 parallel games) | **2× 5090 local** | Inference-bound; needs the SFT'd model loaded once and shared via vLLM |
| Judge tier (PRM + directive) | **2× 5090 local** | Or call Gemini API for the directive judge |
| Actor training (gemma4:e4b + LoRA) | **2× H200** | Asynchronous w.r.t. rollouts; needs ~60GB for actor + replay buffer |
| Full RL training run | **4× H200, 72-hour job** | One pass through the game library × N rollouts |

**Libraries.**

- **`OpenClaw-RL`** upstream — primary training framework
- **`trl`** — has `GRPO` and `DPO` trainers if we want simpler baselines
  for ablation
- **`vllm`** — for the actor inference server (low-latency, batched)
- **`ray`** — for the rollout pool orchestration if OpenClaw-RL uses it
- **`deepspeed`** — for the actor gradient updates at H200 scale
- **`wandb`** — logging

**Success criteria.** Online RL'd model beats the SFT'd model on the
held-out games by ≥10% absolute success rate. Validates the closed-loop
hypothesis. Bonus: matches Gemini Pro 3.1 oracle performance on Pokemon
while running locally on a 5090.

---

## Cross-cutting infrastructure

### Eval harness

A single command that runs the current policy against the game library
and produces a scoreboard. Used for:

- SFT epoch eval (does loss correlate with game success?)
- RL training step eval (is the policy improving in real terms?)
- Final benchmark numbers for the paper / blog post

**Lives in:** `eval/run_eval.py`
**Reports to:** `eval/results/<run_id>/scoreboard.json` + W&B table
**Format:** per-game success rate, mean steps to completion, mean
tool-call validity rate, video links

### Trajectory storage

50-250 GB of trajectories is a lot. Plan:

- **Hot storage**: last 7 days of runs in `run_data/` on local disk
- **Warm storage**: SFT-ready exported parquet on `/media/milkkarten/data/`
- **Cold storage**: compressed full trajectories (incl. screenshots) on
  external drive or S3-compatible bucket
- Keep the parquet (≤50 GB) on every machine that trains; only fetch
  raw screenshots on demand

### Reproducibility

- Pin all model versions in a `MODEL_VERSIONS.md` (gemma4:26b digest,
  MolmoWeb-8B digest, Gemini API version, etc.)
- Include the model versions and full prompt in every trajectory record
- Tag every training run with the dataset hash + base model hash so
  results are reproducible

---

## Resource summary

| Phase | Where | GPU | Wall time | $ cost | Recurring? |
|---|---|---|---|---|---|
| Step 0.1 browser oracle (gemma4:26b + MolmoWeb-8B) | Local | 2× 5090 | continuous | $0 | yes |
| **Step 0.2 Pokemon oracle — primary (Gemini Pro 3.1)** | **Cloud API** | n/a | continuous | **~$500-750 baseline** | yes (default) |
| Step 0.2 Pokemon oracle — cost-saving alternative (gemma4:26b) | Local | 2× 5090 | continuous | $0 | only if Step 0.5 displaces Gemini |
| Step 0.3 real-time games (gemma4:26b + tuned config) | Local | 2× 5090 | continuous | $0 | yes |
| Step 0.5 teacher quality decision point | Local + API | 2× 5090 + Gemini baseline eval | 6-12 hours | ~$100-150 Gemini baseline | one-shot per teacher rotation |
| Step 1 trajectory export | Local | 0 | <1 hour | $0 | one-shot per dataset version |
| **Step 2a SFT LoRA (gemma4:e4b, bf16, rank 256)** | **Local** | **1× 5090** | **2-4 hours** | $0 | per dataset iteration |
| **Step 2b SFT full FT (gemma4:e4b, bf16)** | **H200** | **1× H200 (or 2× H200 for speed)** | **8-16 hours** | H200 hours | per production checkpoint |
| Step 3 RL (rollout pool) | Local | 2× 5090 | continuous | $0 | during training |
| Step 3 RL (actor training) | H200 | **2-4× H200, 72-hour job** | 72 hours | H200 hours | per experiment |
| Eval | Local | 2× 5090 | 1-2 hours | $0 | per checkpoint |

**H200 budget needed.**
- **Step 2b production runs:** 1 H200 × 8-16h × 4-6 sweeps = **32-96 H200-hours per dataset version**
- **Step 3 RL runs:** 4 H200s × 72h × 1-2 runs = **288-576 H200-hours per experiment**
- **Total per training cycle:** **~320-670 H200-hours**

If we can get **one 72-hour 4× H200 job per week**, that's 288 GPU-hours
per week, which fits one full RL training cycle. Step 2a (LoRA
iteration) runs on the local 5090 in parallel and burns zero H200
hours. Step 2b (full FT) is a single H200 job per production
checkpoint.

**$ budget — Gemini Pro 3.1 (default assumption):**
- **Pokemon dataset collection (Step 0.2):** ~$500-750 baseline. Real
  spend, committed unless Step 0.5 displaces Gemini.
- **Step 0.5 baseline eval:** ~$100-150 to establish the reference
  number that gemma4 has to beat. This is spent regardless.
- **Total $ exposure (default):** ~$600-900
- **Total $ exposure if gemma4 displaces Gemini in Step 0.5:** ~$100-150
  (just the baseline eval) + the saved $500-750 stays unspent

**Cost-saving trigger.** Whether we spend the full $500-750 Pokemon
budget depends entirely on Step 0.5. Build a Gemini API key and budget
into the plan from day 1; treat the $0 outcome as the upside, not the
expectation.

---

## Open questions / risks

1. **Does MolmoWeb-8B work on game canvases at all?** MolmoWeb is
   trained on web screenshots — DOM-rendered HTML pages with standard
   UI elements (buttons, forms, links). Pixel-art game canvases
   (Folder Dungeon, Flappy Bird) are out of training distribution
   because the canvas is opaque to the DOM and the visual style is
   pixelated. **Mitigation:** Step 0.1 includes a smoke test — manually
   evaluate MolmoWeb on 20 game screenshots before committing to the
   architecture. If it fails on pixel-art but works on more standard
   HTML5 games (Brackeys jam entries with normal UI), narrow the
   browser-puzzle category to those. Fallback: train a small detection
   head or use template matching for the truly stylized games.

2. **Will the SFT'd model generalize across game classes?** The dataset
   is multi-game but each example is single-game. The hope is that the
   *tool-call format* generalizes even if the per-game knowledge
   doesn't. **Mitigation:** Step 2 explicitly evaluates on a held-out
   game (a game whose trajectories were NOT in the SFT set).

3. **Is the autoevolve directive signal actually useful as RL signal?**
   The current evolver's outputs are loose textual hints, not
   well-grounded credit assignment. **Mitigation:** Step 3 starts with
   PRM-only RL (scalar reward) and adds the directive signal as an
   ablation, not as the primary loss.

4. **OpenClaw-RL is brand new (March 2026).** The codebase may be alpha
   quality. **Mitigation:** Have a fallback to plain GRPO via TRL if
   OpenClaw doesn't work out of the box.

5. **Gemini Pro 3.1 cost.** Default $500-750 for the Pokemon dataset is
   the baseline assumption. **Mitigation:** Step 0.5 explicitly tests
   whether gemma4:26b (or 31b) can match Gemini quality on the same
   eval suite — if yes, the Gemini Pokemon spend goes away. Start with
   5 trajectories on Gemini to validate quality on the full Pokemon
   eval before spending the full budget; bail out early if Gemini
   itself doesn't reach Route 101 reliably.

6. **gemma4:e4b base capacity ceiling.** A 4B model may have a hard
   cap on what tool-call complexity it can learn, regardless of how
   good the SFT data is. **Mitigation:** Step 0.5's teacher quality eval
   reveals the ceiling first — if 26b struggles even as a teacher, e4b
   as a student is unlikely to do better. Decision tree handles the
   "escalate to a larger student" branch implicitly: if Step 2a/2b
   stalls below 60% of teacher performance, the dataset isn't the
   problem and we'd revisit the model-family constraint.

---

## Phase ordering & dependencies

```
0.1 ─┐
0.2 ─┼─→ 0.4 game library ─→ 0.5 teacher eval ─→ 1 dataset ─→ 2a LoRA ─→ 2b full FT ─┐
0.3 ─┘                                                                                ├─→ 3 RL
                                                                                       │
                            eval harness ────────────────────────────────────────────┘
                            (used by 0.5, 2a, 2b, 3)
```

**Critical path:** 0.4 (game library) blocks 0.5 (teacher eval). 0.5
blocks 1 (dataset) — we can't start building the dataset until we know
which teacher(s) to use. 1 blocks 2a. 2a's results inform whether 2b
is needed (or if LoRA is enough). 2b blocks 3. The eval harness is
needed by 0.5, 2a, 2b, and 3 — build it during 0.4 so it's ready in
time for Step 0.5.

**Parallel work:**
- 0.1, 0.2, 0.3 are independent — pick whichever oracle is cheapest to
  set up first (probably 0.1 because MolmoWeb is local)
- The scripted Flappy bot (0.3 tier B) can be built any time
- 31b smoke can run in parallel with 26b smokes during Step 0.5
- Step 2a (LoRA on 1× 5090) can run in parallel with Step 0.5 / Step 1
  iteration once an initial dataset exists — short feedback loop is the
  whole point of the LoRA tier

---

## Stretch goals

- **Multi-modal context distillation**: train the small model to also
  consume Pokemon's structured `game_state` JSON, not just the screenshot
- **Per-game routing**: train a tiny classifier to pick which sub-policy
  (browser puzzle / Pokemon / real-time) given a screenshot, so a single
  agent process can handle the whole library
- **Public benchmark release**: turn the eval harness + game library +
  trained checkpoint into a HuggingFace leaderboard for "small VLM game
  agents"

---

*This plan is intentionally over-specified — every section here should
be challenged before implementation starts. The goal is to make the
options legible, not to commit to all of them.*
