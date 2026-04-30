# Multi-checkpoint cohort launch spec

12 runs, each 24h on della-ailab (1× H200 141GB), branch `feature/generalized-harness`.

## Launch gate

Fire only when E_traj_complete (job `673778c8`, slurm 7503894) shows:
1. Iter 5+ completed with `reward_mean ≥ 0.30` (no erosion below E_traj_v2's iter 5)
2. `obj_idx > 3` in `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_E_traj_complete/objectives_index.txt`

If either fails by E_traj_complete iter 8, abort cohort and investigate.

## Common args (all 12 runs)

```
--adapter /scratch/gpfs/CHIJ/milkkarten/grpo_staging/adapters/26b_red_v1
--base-model-id unsloth/gemma-4-26B-A4B-it
--game red
--rom /scratch/gpfs/CHIJ/milkkarten/grpo_staging/PokemonRed-GBC/pokered.gbc
--data-root /scratch/gpfs/CHIJ/milkkarten/grpo_staging
--rollout-steps 128
--iterations 20
--num-gpus 1
--rollout-device 0
--relabel-threshold 1.0
--sft-epochs 3
--sft-lr 5e-6
--rollout-backend unsloth
--shard-mode accumulate
--shard-window 10
--reset-free
--prm-mode pairwise
--prm-stride 8
--direct-objectives categorized_full_game
--socks-login-host della-gpu.princeton.edu
```

For each run, ADD `--output <run_dir>` and `--initial-state <zip>`.

## Phase A — Multi-checkpoint generalization (8 runs)

| Short name | Output dir | Initial state | story_idx |
|---|---|---|---|
| chk01_rival_battle | `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_chk01_rival_battle` | `/scratch/gpfs/CHIJ/milkkarten/checkpoints/chk01_rival_battle.zip` | 4 |
| chk02_leave_oaks_lab | `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_chk02_leave_oaks_lab` | `/scratch/gpfs/CHIJ/milkkarten/checkpoints/chk02_leave_oaks_lab.zip` | 10 |
| chk03_get_oaks_parcel | `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_chk03_get_oaks_parcel` | `/scratch/gpfs/CHIJ/milkkarten/checkpoints/chk03_get_oaks_parcel.zip` | 7 |
| chk04_reach_pewter | `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_chk04_reach_pewter` | `/scratch/gpfs/CHIJ/milkkarten/checkpoints/chk04_reach_pewter.zip` | 12 |
| chk05_defeat_brock | `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_chk05_defeat_brock` | `/scratch/gpfs/CHIJ/milkkarten/checkpoints/chk05_defeat_brock.zip` | 14 |
| chk06_enter_mt_moon | `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_chk06_enter_mt_moon` | `/scratch/gpfs/CHIJ/milkkarten/checkpoints/chk06_enter_mt_moon.zip` | 18 |
| chk07_defeat_rival_cerulean | `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_chk07_defeat_rival_cerulean` | `/scratch/gpfs/CHIJ/milkkarten/checkpoints/chk07_defeat_rival_cerulean.zip` | 24 |
| chk08_meet_bill | `/scratch/gpfs/CHIJ/milkkarten/pokeagent_runs/dagger_chk08_meet_bill` | `/scratch/gpfs/CHIJ/milkkarten/checkpoints/chk08_meet_bill.zip` | 30 |

## Phase B — Ablations (4 runs, all from chk05_defeat_brock)

Drop a single piece of the fixed-teacher stack to isolate which matters.

| Short name | Variant | Branch / extra args |
|---|---|---|
| chk05_abl_no_completed | drop ALREADY-COMPLETED block | branch `ablation/no-completed-block` (TODO: create + apply patch) |
| chk05_abl_no_reasoning | drop reasoning snippets in traj window | branch `ablation/no-reasoning-snippets` (TODO) |
| chk05_abl_64step | 128→64 rollout-steps | same branch, change `--rollout-steps 64` |
| chk05_abl_old_teacher | revert to commit `f58d73e` (pre-fix teacher) | branch `ablation/old-teacher` (revert 436bbbd locally) |

## Submission via gpu_submit_job (one per run)

```
mcp__gpu-manager__gpu_submit_job(
  repo="git@github.com:sethkarten/pokeagent-speedrun.git",
  branch="feature/generalized-harness",
  script="train/dagger_prm.py",
  args=<COMMON_ARGS> + " --output <run_dir> --initial-state <zip>",
  gpu_count=1,
  gpu_memory_gb=141,
  prefer_resource="della-ailab",
  time_limit_hours=24,
)
```

## Expected behavior per run

- iter 0 starts from staged checkpoint.state with story_index auto-set
- 20 iters × 128 rollout steps = 2560 steps of training data per scenario
- Auto-prune keeps disk ≤ 16-20 GB per run (verified on E_traj_v2)
- 24h wall fits ~10-15 iters; 20-iter cap is a safety ceiling

## Success metric (post-cohort analysis)

Per-checkpoint: max obj_idx delta across iters, iter-5 reward, location entropy
in iter 5 vs iter 1. Cohort claim: ≥6/8 scenarios show Δobj_idx ≥ 1, ≥4/8 show
Δobj_idx ≥ 3.
