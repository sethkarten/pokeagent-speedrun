"""
PokeAgent vs BenchPress: Does Pokemon battling provide new evaluation signal?

Compares PokeAgent Battling Track (Gen 9 OU) GXE scores against
the BenchPress (model, eval) matrix to check orthogonality.

Updated March 2026 with latest GXE data from extended timer ladder.
"""

import json
import itertools
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
#  Load BenchPress data
# ---------------------------------------------------------------------------

with open("llm_benchmark_data.json") as f:
    bp = json.load(f)

bp_models = {m["id"]: m for m in bp["models"]}
bp_benchmarks = {b["id"]: b for b in bp["benchmarks"]}

# Build score matrix: model_id -> benchmark_id -> score
bp_scores = {}
for s in bp["scores"]:
    bp_scores.setdefault(s["model_id"], {})[s["benchmark_id"]] = s["score"]

# ---------------------------------------------------------------------------
#  PokeAgent GXE data (Gen 9 OU Extended Timer, updated March 2026)
#  Map BenchPress model IDs -> PokeAgent GXE
# ---------------------------------------------------------------------------

pokeagent_gxe = {
    "gemini-3.1-pro":       90.76,
    "gpt-5":                89.86,
    "gpt-5.2":              89.86,   # Same eval as GPT-5 in our ladder
    "gemini-3-flash":       82.29,
    "gemini-3-pro":         75.66,
    "claude-opus-4.6":      69.58,
    "gemini-2.5-flash":     55.52,
    "claude-sonnet-4.6":    55.11,
    "grok-3-beta":          53.91,   # Grok-3 in our ladder
    "minimax-m2":           52.43,   # MiniMax M2.5
    "deepseek-v3":          42.98,
    "qwen3-14b":            42.61,
    "kimi-k2.5":            42.37,
    "qwen3.5-397b":         40.39,   # Qwen3.5 Plus
    "qwen3-4b":             36.69,
    "qwen3-8b":             29.18,
}

# Remove models not in BenchPress
pokeagent_gxe = {k: v for k, v in pokeagent_gxe.items() if k in bp_models}

print("=" * 70)
print("PokeAgent vs BenchPress: Orthogonality Analysis")
print("=" * 70)
print(f"\nBenchPress: {len(bp_models)} models x {len(bp_benchmarks)} benchmarks")
print(f"PokeAgent models matched to BenchPress: {len(pokeagent_gxe)}")
print()

for mid, gxe in sorted(pokeagent_gxe.items(), key=lambda x: -x[1]):
    name = bp_models[mid]["name"]
    n_bp = len(bp_scores.get(mid, {}))
    print(f"  {name:30s}  GXE={gxe:5.1f}%  ({n_bp} BenchPress scores)")

# ---------------------------------------------------------------------------
#  1. Spearman rank correlation: PokeAgent GXE vs each benchmark
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("1. Spearman Rank Correlation: PokeAgent GXE vs Each Benchmark")
print("=" * 70)

model_ids = list(pokeagent_gxe.keys())
benchmark_ids = [b["id"] for b in bp["benchmarks"]]

correlations = []
for bid in benchmark_ids:
    paired = []
    for mid in model_ids:
        if mid in bp_scores and bid in bp_scores[mid]:
            paired.append((pokeagent_gxe[mid], bp_scores[mid][bid]))

    if len(paired) >= 5:
        gxes = [p[0] for p in paired]
        bscores = [p[1] for p in paired]
        rho, pval = stats.spearmanr(gxes, bscores)
        correlations.append((bid, bp_benchmarks[bid]["name"], rho, pval, len(paired)))

correlations.sort(key=lambda x: -abs(x[2]))

print(f"\n{'Benchmark':<30s} {'ρ':>8s} {'p-val':>8s} {'N':>4s}")
print("-" * 54)
for bid, name, rho, pval, n in correlations:
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"{name:<30s} {rho:>+8.3f} {pval:>8.4f} {n:>4d} {sig}")

abs_rhos = [abs(c[2]) for c in correlations]
print(f"\nMean |ρ|: {np.mean(abs_rhos):.3f}")
print(f"Max  |ρ|: {max(abs_rhos):.3f} ({correlations[0][1]})")
print(f"|ρ| > 0.8: {sum(1 for r in abs_rhos if r > 0.8)}/{len(abs_rhos)}")
print(f"|ρ| < 0.5: {sum(1 for r in abs_rhos if r < 0.5)}/{len(abs_rhos)}")

# ---------------------------------------------------------------------------
#  2. Find the largest fully-observed submatrix
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("2. SVD Analysis: How well does rank-2 structure explain PokeAgent?")
print("=" * 70)

# Strategy: greedily build the largest complete submatrix
# Start with all PokeAgent models, find benchmarks they all share

def find_best_submatrix(model_pool, min_models=8):
    """Find the largest (models x benchmarks) complete block."""
    best = ([], [])
    best_size = 0

    # For each subset of benchmarks, find how many models have all of them
    # Greedy: start with benchmarks that have highest coverage, add greedily
    bench_coverage = {}
    for bid in benchmark_ids:
        covered = [mid for mid in model_pool if bid in bp_scores.get(mid, {})]
        bench_coverage[bid] = set(covered)

    # Sort benchmarks by coverage (desc)
    sorted_benches = sorted(bench_coverage.items(), key=lambda x: -len(x[1]))

    # Greedy forward selection
    selected_benches = []
    current_models = set(model_pool)

    for bid, covered in sorted_benches:
        new_models = current_models & covered
        if len(new_models) >= min_models:
            selected_benches.append(bid)
            current_models = new_models
            size = len(current_models) * len(selected_benches)
            if size > best_size:
                best = (list(current_models), list(selected_benches))
                best_size = size

    return best

sub_models, sub_benchmarks = find_best_submatrix(model_ids, min_models=6)

# Also try: find ALL benchmarks that a minimum set of models share
# Try with progressively fewer models
for min_m in [14, 12, 10, 8, 6]:
    sm, sb = find_best_submatrix(model_ids, min_models=min_m)
    if len(sm) * len(sb) > len(sub_models) * len(sub_benchmarks):
        sub_models, sub_benchmarks = sm, sb
    # Also try: fix models first, then find shared benchmarks
    if len(model_ids) >= min_m:
        # Take top min_m models by BenchPress coverage
        by_coverage = sorted(model_ids, key=lambda m: -len(bp_scores.get(m, {})))
        top_m = by_coverage[:min_m]
        shared_b = [bid for bid in benchmark_ids
                    if all(bid in bp_scores.get(mid, {}) for mid in top_m)]
        if len(top_m) * len(shared_b) > len(sub_models) * len(sub_benchmarks):
            sub_models, sub_benchmarks = top_m, shared_b

print(f"\nBest complete submatrix: {len(sub_models)} models x {len(sub_benchmarks)} benchmarks")
print(f"\nModels:")
for mid in sub_models:
    print(f"  {bp_models[mid]['name']:30s}  PokeAgent GXE={pokeagent_gxe[mid]:.1f}%")
print(f"\nBenchmarks:")
for bid in sub_benchmarks:
    print(f"  {bp_benchmarks[bid]['name']}")

# Build the matrix
n_models = len(sub_models)
n_bench = len(sub_benchmarks)

M = np.zeros((n_models, n_bench))
for i, mid in enumerate(sub_models):
    for j, bid in enumerate(sub_benchmarks):
        M[i, j] = bp_scores[mid][bid]

poke_col = np.array([pokeagent_gxe[mid] for mid in sub_models])

# Normalize columns to [0, 1]
M_norm = (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0) + 1e-10)
poke_norm = (poke_col - poke_col.min()) / (poke_col.max() - poke_col.min() + 1e-10)

# SVD of BenchPress-only matrix
U, S, Vt = np.linalg.svd(M_norm, full_matrices=False)

print(f"\nSingular values (top 5): {S[:min(5, len(S))].round(3)}")
variance_explained = (S ** 2) / (S ** 2).sum()
print(f"Variance explained:      {(variance_explained[:min(5, len(S))] * 100).round(1)}%")
cum2 = variance_explained[:2].sum() * 100
cum3 = variance_explained[:min(3, len(S))].sum() * 100
print(f"Cumulative (rank-2):     {cum2:.1f}%")
print(f"Cumulative (rank-3):     {cum3:.1f}%")

# Project PokeAgent onto rank-k BenchPress structure
print(f"\nR² of PokeAgent prediction from rank-k BenchPress structure:")
print(f"{'Rank':>6s} {'R²':>8s} {'Unexplained':>12s}")
print("-" * 30)

for k in [1, 2, 3, 5, min(8, n_models - 1)]:
    if k > len(S):
        break
    U_k = U[:, :k]
    beta, _, _, _ = np.linalg.lstsq(U_k, poke_norm, rcond=None)
    poke_pred = U_k @ beta
    ss_res = np.sum((poke_norm - poke_pred) ** 2)
    ss_tot = np.sum((poke_norm - poke_norm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"{k:>6d} {r2:>8.3f} {(1 - r2) * 100:>10.1f}%")

# ---------------------------------------------------------------------------
#  3. Leave-one-out prediction error
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("3. Leave-One-Out Prediction Error")
print("=" * 70)

M_full = np.column_stack([M_norm, poke_norm])
bench_names = [bp_benchmarks[bid]["name"] for bid in sub_benchmarks] + ["PokeAgent GXE"]
n_cols = M_full.shape[1]
poke_idx = n_cols - 1

loo_errors = []
print(f"\n{'Model':<30s} {'True GXE':>10s} {'Pred GXE':>10s} {'Error':>8s}")
print("-" * 62)

for hold_out in range(n_models):
    train_idx = [i for i in range(n_models) if i != hold_out]
    M_train = M_full[train_idx, :]

    U_t, S_t, Vt_t = np.linalg.svd(M_train, full_matrices=False)
    V2 = Vt_t[:2, :]

    known_scores = M_full[hold_out, :poke_idx]
    V2_known = V2[:, :poke_idx]
    coords, _, _, _ = np.linalg.lstsq(V2_known.T, known_scores, rcond=None)

    pred_norm = coords @ V2[:, poke_idx]
    pred_gxe = pred_norm * (poke_col.max() - poke_col.min()) + poke_col.min()
    true_gxe = pokeagent_gxe[sub_models[hold_out]]

    error = abs(pred_gxe - true_gxe)
    loo_errors.append(error)

    name = bp_models[sub_models[hold_out]]["name"]
    print(f"{name:<30s} {true_gxe:>10.1f} {pred_gxe:>10.1f} {error:>8.1f}")

print(f"\nLOO MAE:    {np.mean(loo_errors):.1f} pp")
print(f"LOO Median: {np.median(loo_errors):.1f} pp")
print(f"LOO Max:    {np.max(loo_errors):.1f} pp")

# ---------------------------------------------------------------------------
#  4. Benchmark predictability comparison
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("4. Benchmark Predictability Ranking (LOO MAE, normalized)")
print("=" * 70)

all_bench_errors = {}
for target_j in range(n_cols):
    errors = []
    for hold_out in range(n_models):
        train_idx = [i for i in range(n_models) if i != hold_out]
        M_train = M_full[train_idx, :]

        U_t, S_t, Vt_t = np.linalg.svd(M_train, full_matrices=False)
        V2 = Vt_t[:2, :]

        known_idx = [j for j in range(n_cols) if j != target_j]
        known_scores = M_full[hold_out, known_idx]
        V2_known = V2[:, known_idx]

        coords, _, _, _ = np.linalg.lstsq(V2_known.T, known_scores, rcond=None)
        pred = coords @ V2[:, target_j]
        true = M_full[hold_out, target_j]
        errors.append(abs(pred - true))

    all_bench_errors[bench_names[target_j]] = np.mean(errors)

sorted_errors = sorted(all_bench_errors.items(), key=lambda x: -x[1])

poke_rank = next(i + 1 for i, (n, _) in enumerate(sorted_errors) if n == "PokeAgent GXE")

print(f"\n{'Rank':>4s}  {'Benchmark':<30s} {'LOO MAE':>10s}")
print("-" * 48)
for i, (name, mae) in enumerate(sorted_errors):
    marker = " <<<" if name == "PokeAgent GXE" else ""
    print(f"{i+1:>4d}  {name:<30s} {mae:>10.4f}{marker}")

print(f"\nPokeAgent is #{poke_rank} hardest to predict out of {n_cols} benchmarks")

# ---------------------------------------------------------------------------
#  5. Model anomaly analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("5. Model Anomalies: PokeAgent vs Standard Benchmark Rankings")
print("=" * 70)

poke_ranks = stats.rankdata(poke_col) / len(poke_col)
bench_mean_ranks = np.mean(stats.rankdata(M, axis=0) / n_models, axis=1)

print(f"\n{'Model':<30s} {'Bench %ile':>10s} {'Poke %ile':>10s} {'Delta':>8s}")
print("-" * 62)

anomalies = []
for i, mid in enumerate(sub_models):
    name = bp_models[mid]["name"]
    delta = (poke_ranks[i] - bench_mean_ranks[i]) * 100
    anomalies.append((name, bench_mean_ranks[i] * 100, poke_ranks[i] * 100, delta))

anomalies.sort(key=lambda x: -abs(x[3]))
for name, bp, pp, d in anomalies:
    marker = " ***" if abs(d) > 25 else " **" if abs(d) > 20 else " *" if abs(d) > 15 else ""
    print(f"{name:<30s} {bp:>9.1f}% {pp:>9.1f}% {d:>+7.1f}{marker}")

# ---------------------------------------------------------------------------
#  6. Sensitivity: with vs without Kimi/MiniMax
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("6. Sensitivity: Effect of Kimi K2.5 and MiniMax")
print("=" * 70)

outlier_ids = {"kimi-k2.5", "minimax-m2"}
core_models = [mid for mid in sub_models if mid not in outlier_ids]
core_in_sub = [i for i, mid in enumerate(sub_models) if mid not in outlier_ids]

if len(core_models) >= 4 and len(core_in_sub) >= 4:
    M_core = M_norm[core_in_sub, :]
    poke_core = poke_norm[core_in_sub]

    U_c, S_c, Vt_c = np.linalg.svd(M_core, full_matrices=False)

    for label, U_svd, poke_vec, n in [
        ("WITHOUT Kimi/MiniMax", U_c, poke_core, len(core_models)),
        ("WITH    Kimi/MiniMax", U, poke_norm, n_models),
    ]:
        for k in [2, 3]:
            if k > U_svd.shape[1]:
                break
            U_k = U_svd[:, :k]
            beta, _, _, _ = np.linalg.lstsq(U_k, poke_vec, rcond=None)
            pred = U_k @ beta
            ss_res = np.sum((poke_vec - pred) ** 2)
            ss_tot = np.sum((poke_vec - poke_vec.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"  {label} (N={n:>2d}): rank-{k} R² = {r2:.3f}  ({(1-r2)*100:.1f}% unexplained)")
        print()
else:
    print("  (Not enough core models in submatrix to test)")

# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Models matched:     {len(pokeagent_gxe)} total, {n_models} in complete submatrix
Submatrix size:     {n_models} models x {n_bench} benchmarks

Spearman correlations with PokeAgent GXE:
  Max |ρ|:          {max(abs_rhos):.3f} ({correlations[0][1]})
  Mean |ρ|:         {np.mean(abs_rhos):.3f}
  # with |ρ| > 0.8: {sum(1 for r in abs_rhos if r > 0.8)}/{len(abs_rhos)}

SVD rank-2 R²:      {1 - ss_res/ss_tot:.3f} (with k=2)
LOO prediction MAE:  {np.mean(loo_errors):.1f} percentage points
Predictability rank: #{poke_rank} hardest out of {n_cols}

Conclusion: PokeAgent GXE is {'poorly' if poke_rank <= 3 else 'moderately' if poke_rank <= n_cols//2 else 'well'} predicted
by rank-2 BenchPress structure, suggesting it {'provides' if poke_rank <= n_cols//2 else 'does not provide'}
novel evaluation signal beyond standard benchmarks.
""")
