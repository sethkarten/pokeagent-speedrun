# Gemma4 Pokemon Agent — Training Results Summary

**As of:** 2026-04-15
**Purpose:** Snapshot of all SFT + GRPO runs across Gemma4 model sizes for paper writing. Includes pointers to raw artifacts on della and locally for follow-up analysis.

---

## TL;DR

- **8 SFT runs trained** (4 model sizes × 2 games) on della H200s.
- **31B SFT was trained but never evaluated** for either game. Largest gap in the eval matrix.
- **Only the 26B SFT survived eval** — smaller models (e4b, e2b) collapsed to `tool_format ≈ 0` on real harness prompts despite low training loss.
- **GRPO is in-progress** on 26B Emerald + 26B Red. Diversity bug fixed (temp=0.3→0.5); current run shows non-zero `reward_std` and recent decline in `clipped_ratio`. Tool-match still ~0 because of stochastic non-EOS truncation, not EOS bug.
- All raw training/eval artifacts catalogued below.

---

## 1. SFT Training Matrix

All runs use Unsloth + LoRA (r=256, α=512) on Gemma4 base, bf16, gradient checkpointing, max_seq_length=8192.

Source data: Gemini-3.x trajectories collected via the auto-evolve harness, exported to JSONL with the `data/export_sft_dataset.py` pipeline.

| Model | Game | Steps | Final loss | Status | Eval'd? |
|---|---|---|---|---|---|
| **gemma4:31b** | Emerald | 2,838 | **0.017** | ✅ converged | ❌ **no eval** |
| **gemma4:31b** | Red | 3,426 | **0.020** | ✅ converged | ❌ **no eval** |
| gemma4:26b | Emerald | 1,190 | 0.617 | ✅ converged | ✅ |
| gemma4:26b | Red | 1,350 | 0.045 | ✅ converged | ✅ partial |
| gemma4:e4b | Emerald | 1,894 | 0.035 | ✅ converged | ✅ |
| gemma4:e4b | Red | 2,186 | 0.014 | ✅ converged | ❌ no eval |
| gemma4:e2b | Emerald | 502 | 0.064 | ⚠️ short run | ✅ |
| gemma4:e2b | Red | **85** | 0.579 | ❌ **incomplete** | ❌ no eval |

**Loss anomaly worth flagging in paper:** 26B Emerald final loss (0.617) is 10–30× higher than the other completed runs. This is despite using the same data and recipe. Could be undertrained at 1.19 epochs, could be a dataset quality issue (Emerald has more subagent variety than Red). Worth re-running to higher epoch count or investigating data quality.

**Dataset sizes:**
- `emerald_v3`: 8,592 examples (307 MB), 5 shards
- `red_v1`: 6,609 examples (199 MB), 4 shards

**Source data on della:**
```
/scratch/gpfs/CHIJ/milkkarten/grpo_staging/data/sft_dataset/emerald_v3/  (8,592 examples)
/scratch/gpfs/CHIJ/milkkarten/grpo_staging/data/sft_dataset/red_v1/      (6,609 examples)
```

**Local copy:**
```
/media/milkkarten/data/gen-harness/data/sft_dataset/{emerald_v3,red_v1}/
```

**Underlying Gemini trajectories (raw):**
```
/scratch/gpfs/CHIJ/milkkarten/grpo_staging/run_data/
  20260408_195739_autonomous_autonomous_objective_creation_ae_autoevolve/  (23M)
  20260409_211932_autonomous_autonomous_objective_creation_ae_autoevolve/  (35M)
  20260409_212128_autonomous_autonomous_objective_creation_ae_autoevolve/  (2.8M)
```
These contain step-level screenshots + raw Gemini-3.1-pro responses + harness state snapshots.

**SFT checkpoints on della:**
```
/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/{model_size}_{game}_{ver}/
  ├── checkpoint-NNN/   # LoRA adapter snapshots
  ├── losses.jsonl      # Step-level training loss + grad_norm + lr
  ├── README.md         # Run config
  └── trainer_state.json (in each ckpt dir)
```

Specifically:
- `26b_emerald_v3/` — used as GRPO init
- `26b_red_v1/` — used as GRPO init
- `31b_emerald_v3/` checkpoints 1750, 2000, 2250, 2500, **2750 (latest)** — never eval'd
- `31b_red_v1/` checkpoints 2250, 2500, 2750, 3000, **3250 (latest)** — never eval'd
- `e4b_emerald_v3/`, `e4b_red_v1/`, `e2b_emerald_v3/`, `e2b_red_v1/`

**Adapters used in GRPO (staged):**
```
/scratch/gpfs/CHIJ/milkkarten/grpo_staging/adapters/
  ├── 26b_emerald_v3/
  └── 26b_red_v1/
```

**Merged + GGUF-converted models (for local Ollama serving):**
- `gemma4-emerald:26b` (Q4_K_M, served via Ollama on local 5090)
- `gemma4-emerald:e4b`
- `gemma4-emerald:e2b`
- `gemma4-red:26b`
- `gemma4:26b`, `gemma4:e4b`, `gemma4:e2b` (base baselines)
- 31B merged checkpoints: pending (per task #96)

---

## 2. SFT Eval Results (existing)

### 2a. Simplified prompt eval — `data/eval_sft_quality_report.{json,md}` (2026-04-12)
- Models: `gemma4:26b` vs `gemma4-emerald:26b`
- Dataset: 20 emerald orchestrator samples
- Headline: SFT lifts `tool_format` 0.05→0.55, `action_relevance` 0.53→0.68, `reasoning_similarity` 0.40→0.61.
- Identified bug: SFT 26B hallucinates dialogue boxes in overworld (overworld grounding -0.15 vs base).

### 2b. Full base-vs-SFT comparison — `data/eval_full_model_comparison.{json,md}` (2026-04-12)
- Models: `gemma4:{26b,e4b,e2b}` × {base, SFT-emerald}
- Dataset: 20 emerald samples, simplified prompt
- Headline: 26B SFT wins everything; e4b/e2b SFT score `tool_format=0` (collapsed).
- **No 31B in this matrix.**

### 2c. Real harness prompt eval — `data/eval_real_prompts_comparison.{json,md}` (2026-04-13)
- Models: same 6 as above
- Dataset: 16 emerald samples with **full harness prompts** (40-55K chars including system prompt + state + history)
- Headline:
  - SFT 26B: **tool_format 0.94**, grounding 0.81, degenerate 0.00 — **only viable model**
  - SFT e4b: tool_format 0.00, degenerate 0.00, but action_relevance 0.19 (low)
  - SFT e2b: tool_format 0.06, degenerate 0.06 — broken
- This is the eval that defined the GRPO scoping — only 26B made the cut.
- **No 31B in this matrix.**

### 2d. Red-specific eval — `data/eval_red_comparison.json` (2026-04-13)
- Models: `gemma4:26b` vs `gemma4-red:26b`
- Note: tool_format is 0.0 for both base AND SFT in spot-checks. Per memory, the Red SFT adapter is suspected degenerate (clipped completions in GRPO further confirm). **Worth re-investigating before publication.**

### 2e. E2B-only eval — `data/eval_e2b_comparison.json` (2026-04-13)
- Models: `gemma4:e2b` vs `gemma4-emerald:e2b`
- Spot results show e2b base actually scores tool_format=1.0 on simple prompts (it can produce bracket format), but SFT version collapses on real prompts (per 2c).

### 2f. **Missing: 31B SFT eval (Emerald + Red)**
Neither model has been evaluated. Given that 31B converged to a much lower loss than 26B (0.017 vs 0.617 on Emerald), **31B is likely the strongest SFT model and the right candidate to be the paper's headline number.** Highest priority gap.

Required to fill the gap:
1. Merge 31B Emerald checkpoint-2750 LoRA + base, convert to GGUF, serve via Ollama (in flight, task #96).
2. Run the same `data/eval_real_prompts_comparison.py` script with 31B added.
3. Same for 31B Red.

---

## 3. GRPO Training Status

GRPO offline pipeline (Unsloth-patched TRL with DDP monkey-patches) — see `train/grpo_offline.py`, `train/grpo_multigpu.py`, `train/reward_functions.py`.

**Reward functions (4):**
1. `tool_match_reward` (weight 2.0): exact match of model's tool name vs Gemini teacher's tool name
2. `action_similarity_reward` (weight 1.5): Jaccard on button lists for `press_buttons`, partial credit for same-tool-different-args
3. `state_accuracy_reward` (weight 1.0): heuristic match of model's reasoning to game state (location + battle + dialog)
4. `format_reward` (weight 0.5): bracket/`call:`/`Calling` parseable + `ANALYZE:` present

**Config (current `temp05` runs):**
- 4×H200 DDP, Unsloth FastVisionModel + Liger fused linear-CE loss
- num_generations=4, batch_size=1, grad_accum=1
- max_prompt_length=8192, max_completion_length=1024
- temperature=0.5, top_p=0.9, beta=0.0, loss_type=dapo
- mask_truncated_completions=True, scale_rewards=group

**Runs to date (chronological, della):**
| Run dir | Game | Outcome |
|---|---|---|
| `grpo_2gpu_test{2..8}` | smoke | DDP debugging — Unsloth + DDP attribute access |
| `grpo_4gpu_smoke` | smoke | First 4-GPU validation, OOM at grad_accum=2 |
| `grpo_emerald_4gpu_v1` | Emerald | early failure |
| `grpo_red_4gpu_v{1,3,5,6,7,8,9}` | Red | iterations on grad_accum=1 + Liger to fix OOM |
| `grpo_emerald_4gpu_v3` | Emerald | 103 steps; loss -0.01, reward 0.47, but **reward_std=0** → no signal |
| `grpo_emerald_4gpu_postmaint` | Emerald | post-maintenance restart (cancelled) |
| `grpo_red_4gpu_postmaint` | Red | post-maintenance restart (cancelled) |
| **`grpo_emerald_4gpu_v3_temp05`** | Emerald | **CURRENTLY RUNNING** (SLURM 6957199, 5h+) |
| **`grpo_red_4gpu_v3_temp05`** | Red | **CURRENTLY RUNNING** (SLURM 6957201, 5h+) |

**Current run metrics (step ~150, ~6h elapsed of 24h budget):**

| | step | reward | reward_std | tool_match | state_acc | format | clipped |
|---|---|---|---|---|---|---|---|
| Emerald 26B | 142 | 0.69 | 0.43 | 0.0 | 0.31 | 0.75 | 0.75 |
| Red 26B | 151 | 0.94 | 1.55 | 0.25 | 0.19 | 0.13 | 1.0 |

Wins so far:
- Emerald `reward_std` non-zero (was stuck at 0 with temp=0.3 → no gradient)
- Red `tool_match` first non-zero signal (0.25)
- Recent `clipped_ratio` distribution: 45/50 at 0.75, 5/50 at 1.0 (was 100% at 1.0 early)

Open issue:
- Bimodal completion termination — same prompt, 4 samples: 1 terminates ~120 tokens, 3 hit 1024 cap. Suggests sampling rolls into a non-EOS regime stochastically. Diagnosed; fix candidates: `repetition_penalty=1.1`, lower `max_completion_length` to 384, `stop_strings` redundancy. Pending decision before next restart.

**GRPO artifacts on della:**
```
/scratch/gpfs/CHIJ/milkkarten/grpo_staging/train_runs/{run_name}/
  ├── checkpoint-NNN/   # LoRA + tokenizer + chat_template
  ├── losses.jsonl      # Per-step metrics (reward, reward_std, clipped_ratio, etc.)
  ├── config.json       # Run-time config snapshot
  └── README.md
```

**SLURM logs (per-step generations + rich tables):**
```
/scratch/gpfs/CHIJ/milkkarten/pokeagent-speedrun/logs/slurm-NNNNNNN.out
```
Note: TRL `log_completions=True` dumps prompt+completions+rewards as Rich tables — extremely narrow columns due to ~8 columns rendering, but recoverable for paper appendix examples with re-formatting.

---

## 4. Source-Data Pointers Summary

| Asset | Path | Notes |
|---|---|---|
| Raw harness traces | della: `/scratch/gpfs/CHIJ/milkkarten/grpo_staging/run_data/` | Gemini-3.1-pro full trajectories with screenshots + state |
| SFT dataset (emerald) | della: `.../grpo_staging/data/sft_dataset/emerald_v3/` (5 shards, 8,592 ex.) + local mirror at `data/sft_dataset/emerald_v3/` | |
| SFT dataset (red) | della: `.../grpo_staging/data/sft_dataset/red_v1/` (4 shards, 6,609 ex.) + local mirror | |
| SFT checkpoints | della: `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/{model}_{game}_{ver}/` | LoRA adapters |
| SFT training metrics | della: `.../{run}/losses.jsonl` | per-step loss/grad/lr/timestamp |
| GRPO checkpoints | della: `.../grpo_staging/train_runs/{run}/` | LoRA adapters + tokenizer |
| GRPO training metrics | della: `.../grpo_staging/train_runs/{run}/losses.jsonl` | per-step rewards + completion lengths |
| GRPO logged completions | della: `.../pokeagent-speedrun/logs/slurm-{ID}.out` | Rich tables, prompt+4 generations+rewards |
| Eval JSON (raw) | local: `data/eval_*.json` | per-example scores incl. full responses |
| Eval markdown summaries | local: `data/eval_*.md` | aggregate metrics + sample comparisons |
| Merged/GGUF models | local: `merged/`; Ollama tags `gemma4-emerald:{26b,e4b,e2b}`, `gemma4-red:26b` | quantized for local serving |

---

## 5. Open Tasks Blocking Publication

1. **Eval 31B SFT (Emerald + Red).** Highest priority. 31B is best-converged model; without eval we have no headline number.
2. **Re-investigate 26B Red SFT degeneracy.** Eval shows tool_format=0 even on Red SFT — but base also 0; possible eval prompt issue. Worth a closer look before concluding model is broken.
3. **Re-train 26B Emerald SFT to higher epochs?** Final loss 0.62 is anomalously high vs 31B (0.017) and e4b (0.035). Could be undertrained.
4. **Finish GRPO Emerald + Red.** Currently 6h of 24h budget; let run unless we kill for the EOS-termination fix.
5. **Convert 31B adapters to GGUF for local serving.** In flight (task #96), prerequisite for #1.
6. **Re-run eval matrix once 31B is available** — produces the final paper table.
