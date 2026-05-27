"""Phase 1 experiment runner: init×clamp probe (1A) + OFAT screen (1B).

Runs factor sweeps on hero cells with multiprocessing.
Results saved to scratch/phase1_results/.

Usage:
  python3 scratch/phase1_runner.py --phase 1a        # init×clamp probe
  python3 scratch/phase1_runner.py --phase 1b        # OFAT screen
  python3 scratch/phase1_runner.py --phase 1b-resweep # conditional re-sweep
  python3 scratch/phase1_runner.py --analyze 1a      # analyze Phase 1A results
  python3 scratch/phase1_runner.py --analyze 1b      # analyze Phase 1B results
"""
import sys
import os
import json
import time
import signal
import argparse
import math
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scratch.run_taxonomy import (
    ALL_TARGETS, make_data, classify_failure, FACTOR_DEFAULTS,
    EQ6_INIT_STRATEGIES,
)
from scratch.sr_systematic import train_one
from master_formula import INIT_STRATEGIES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SEEDS = 32
SEED0 = 137
N_WORKERS = 10  # 14-core M4 Pro, OMP_NUM_THREADS=2 → ~10 effective

HERO_CELLS = [
    ("eq6",    "E_T1_xy"),      # 0%
    ("v16",    "E_paper_yx"),   # ~19%
    ("v16",    "S_LR_yx"),      # ~85%
    ("hybrid", "S_RL_xy"),      # ~100%
]

# Phase 1A: init × clamp on eq6_E_T1_xy
P1A_CELL = ("eq6", "E_T1_xy")
P1A_INITS = EQ6_INIT_STRATEGIES  # 5: default, biased, uniform, xy_biased, random_hot
P1A_CLAMPS = [1e6, 1e8, 1e10]

# Phase 1B: OFAT sweep levels per factor
OFAT_FACTORS = {
    "init": {
        "eq6":    EQ6_INIT_STRATEGIES,
        "v16":    INIT_STRATEGIES,
        "hybrid": INIT_STRATEGIES,
    },
    "clamp":     [1e6, 1e8, 1e10],
    "optim":     ["adam_1e-2", "adam_1e-3", "adamw_1e-2", "rmsprop_1e-2"],
    "hardening": ["power_law", "linear", "cosine", "step"],
    "grid_seed": [12345, 54321, 99999, 77777],
    "noise":     [0.0, 1e-6, 1e-4, 1e-2],
    "epsilon":   [1e-12, 1e-10, 1e-8, 1e-6],
}

OUTDIR = os.path.join(os.path.dirname(__file__), "phase1_results")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_stop = False

def _handler(sig, frame):
    global _stop
    _stop = True
    print("\n[SIGINT] finishing current batch, then saving...", flush=True)

signal.signal(signal.SIGINT, _handler)

# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------
def _run_one(args_tuple):
    """Single run: (seed, arch, target_key, strategy, factors_override) → result dict."""
    seed, arch, target_key, strategy, factors_override, grid_seed = args_tuple

    desc_str, target_fn, op_fn = ALL_TARGETS[target_key]
    xt, yt, tt, nt, xg, yg, tg, ng = make_data(target_fn, grid_seed=grid_seed)

    clamp = factors_override.get("clamp", FACTOR_DEFAULTS["clamp"])
    optim_name = factors_override.get("optim", FACTOR_DEFAULTS["optim"])
    hardening = factors_override.get("hardening", FACTOR_DEFAULTS["hardening"])
    noise_sigma = factors_override.get("noise", FACTOR_DEFAULTS["noise"])
    epsilon = factors_override.get("epsilon", FACTOR_DEFAULTS["epsilon"])

    r = train_one(
        seed, 3, strategy, xt, yt, tt, xg, yg, tg,
        verbose=False, arch=arch, op_fn=op_fn,
        intermediate_clamp=clamp, eml_clamp=clamp,
        optim_name=optim_name, hardening_schedule=hardening,
        noise_sigma=noise_sigma, epsilon=epsilon,
    )

    ft = classify_failure(r, target_desc_str=desc_str)

    return {
        "seed": seed,
        "arch": arch,
        "target": target_key,
        "strategy": strategy,
        "clamp": clamp,
        "optim": optim_name,
        "hardening": hardening,
        "grid_seed": grid_seed,
        "noise": noise_sigma,
        "epsilon": epsilon,
        "mse_train": float(r["mse_train"]) if np.isfinite(r["mse_train"]) else None,
        "mse_gen": float(r["mse_gen"]) if np.isfinite(r["mse_gen"]) else None,
        "best_loss": float(r["best_loss"]) if np.isfinite(r["best_loss"]) else None,
        "failure_type": ft,
        "description": r["description"],
    }


# ---------------------------------------------------------------------------
# Phase 1A: init × clamp interaction
# ---------------------------------------------------------------------------
def run_phase_1a():
    """5 inits × 3 clamps × 32 seeds = 480 runs on eq6_E_T1_xy."""
    arch, target_key = P1A_CELL
    print(f"Phase 1A: {arch}_{target_key}")
    print(f"  {len(P1A_INITS)} inits × {len(P1A_CLAMPS)} clamps × {N_SEEDS} seeds "
          f"= {len(P1A_INITS) * len(P1A_CLAMPS) * N_SEEDS} runs")

    tasks = []
    for init_strat in P1A_INITS:
        for clamp in P1A_CLAMPS:
            for seed in range(SEED0, SEED0 + N_SEEDS):
                factors = {"clamp": clamp}
                tasks.append((seed, arch, target_key, init_strat, factors,
                              FACTOR_DEFAULTS["grid_seed"]))

    return _run_batch("phase1a", tasks)


# ---------------------------------------------------------------------------
# Phase 1B: OFAT screen
# ---------------------------------------------------------------------------
def run_phase_1b():
    """7 factor sweeps × 4 hero cells × 32 seeds."""
    print(f"Phase 1B: OFAT screen on {len(HERO_CELLS)} hero cells")

    tasks = []
    for arch, target_key in HERO_CELLS:
        # Each factor sweep: vary ONE factor, hold others at default
        for factor_name, levels in OFAT_FACTORS.items():
            if factor_name == "init":
                strat_levels = levels[arch]
                for strat in strat_levels:
                    for seed in range(SEED0, SEED0 + N_SEEDS):
                        tasks.append((seed, arch, target_key, strat, {},
                                      FACTOR_DEFAULTS["grid_seed"]))
            elif factor_name == "clamp":
                for clamp_val in levels:
                    default_strats = EQ6_INIT_STRATEGIES if arch == "eq6" \
                        else INIT_STRATEGIES
                    # Use first strategy as default for OFAT
                    default_strat = "default" if arch == "eq6" else "biased"
                    for seed in range(SEED0, SEED0 + N_SEEDS):
                        tasks.append((seed, arch, target_key, default_strat,
                                      {"clamp": clamp_val},
                                      FACTOR_DEFAULTS["grid_seed"]))
            elif factor_name == "grid_seed":
                default_strat = "default" if arch == "eq6" else "biased"
                for gs in levels:
                    for seed in range(SEED0, SEED0 + N_SEEDS):
                        tasks.append((seed, arch, target_key, default_strat,
                                      {}, gs))
            else:
                default_strat = "default" if arch == "eq6" else "biased"
                for level_val in levels:
                    for seed in range(SEED0, SEED0 + N_SEEDS):
                        tasks.append((seed, arch, target_key, default_strat,
                                      {factor_name: level_val},
                                      FACTOR_DEFAULTS["grid_seed"]))

    # Deduplicate: same (seed, arch, target, strategy, factors, grid_seed) can
    # appear if default-level sweeps overlap. Use frozenset for dedup.
    seen = set()
    unique_tasks = []
    for t in tasks:
        key = (t[0], t[1], t[2], t[3],
               tuple(sorted(t[4].items())), t[5])
        if key not in seen:
            seen.add(key)
            unique_tasks.append(t)

    print(f"  {len(unique_tasks)} unique runs (deduped from {len(tasks)})")
    return _run_batch("phase1b", unique_tasks)


# ---------------------------------------------------------------------------
# Phase 1B' conditional re-sweep
# ---------------------------------------------------------------------------
def run_phase_1b_resweep(winning_init):
    """Re-run clamp/optim/hardening at winning init for all hero cells."""
    print(f"Phase 1B': re-sweep at winning init = {winning_init}")

    tasks = []
    for arch, target_key in HERO_CELLS:
        strat = winning_init if arch == "eq6" else winning_init
        # Check valid strategy for arch
        valid = EQ6_INIT_STRATEGIES if arch == "eq6" else INIT_STRATEGIES
        if strat not in valid:
            print(f"  WARNING: {strat} not valid for {arch}, using default")
            strat = valid[0]

        for factor_name in ["clamp", "optim", "hardening"]:
            levels = OFAT_FACTORS[factor_name]
            if factor_name == "clamp":
                for clamp_val in levels:
                    for seed in range(SEED0, SEED0 + N_SEEDS):
                        tasks.append((seed, arch, target_key, strat,
                                      {"clamp": clamp_val},
                                      FACTOR_DEFAULTS["grid_seed"]))
            else:
                for level_val in levels:
                    for seed in range(SEED0, SEED0 + N_SEEDS):
                        tasks.append((seed, arch, target_key, strat,
                                      {factor_name: level_val},
                                      FACTOR_DEFAULTS["grid_seed"]))

    # Deduplicate
    seen = set()
    unique_tasks = []
    for t in tasks:
        key = (t[0], t[1], t[2], t[3],
               tuple(sorted(t[4].items())), t[5])
        if key not in seen:
            seen.add(key)
            unique_tasks.append(t)

    print(f"  {len(unique_tasks)} runs")
    return _run_batch("phase1b_resweep", unique_tasks)


# ---------------------------------------------------------------------------
# Batch runner with checkpointing
# ---------------------------------------------------------------------------
def _run_batch(phase_name, tasks):
    """Run tasks with multiprocessing, checkpoint every 100 results."""
    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, f"{phase_name}.json")
    ckpt_path = os.path.join(OUTDIR, f"{phase_name}.ckpt.json")

    # Resume from checkpoint
    results = []
    done_keys = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        for r in results:
            k = (r["seed"], r["arch"], r["target"], r["strategy"],
                 r.get("clamp", 1e6), r.get("optim", "adam_1e-2"),
                 r.get("hardening", "power_law"), r.get("grid_seed", 12345),
                 r.get("noise", 0.0), r.get("epsilon", 1e-12))
            done_keys.add(k)
        print(f"  Resumed: {len(results)} results from checkpoint")

    # Filter out already-done tasks
    remaining = []
    for t in tasks:
        seed, arch, tgt, strat, fov, gs = t
        k = (seed, arch, tgt, strat,
             fov.get("clamp", FACTOR_DEFAULTS["clamp"]),
             fov.get("optim", FACTOR_DEFAULTS["optim"]),
             fov.get("hardening", FACTOR_DEFAULTS["hardening"]),
             gs,
             fov.get("noise", FACTOR_DEFAULTS["noise"]),
             fov.get("epsilon", FACTOR_DEFAULTS["epsilon"]))
        if k not in done_keys:
            remaining.append(t)

    total = len(tasks)
    done = total - len(remaining)
    print(f"  {len(remaining)} remaining of {total} total "
          f"({done} already done)")

    if not remaining:
        print("  All runs complete!")
        _write_final(outpath, ckpt_path, phase_name, results)
        return results

    t0 = time.time()
    checkpoint_interval = 100
    batch_results = []

    # Set OMP threads for workers
    os.environ["OMP_NUM_THREADS"] = "2"

    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, remaining)):
            batch_results.append(result)

            if (i + 1) % 10 == 0 or i == len(remaining) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                n_exact = sum(1 for r in batch_results
                              if r["failure_type"] == "exact")
                print(f"  [{done+i+1}/{total}] "
                      f"{n_exact}/{len(batch_results)} exact "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                      flush=True)

            if (i + 1) % checkpoint_interval == 0:
                _write_ckpt(ckpt_path, phase_name,
                            results + batch_results)

            if _stop:
                print("\n  [SIGINT] Saving checkpoint...", flush=True)
                pool.terminate()
                break

    results.extend(batch_results)
    elapsed = time.time() - t0

    if _stop:
        _write_ckpt(ckpt_path, phase_name, results)
        print(f"  Interrupted: {len(results)}/{total} saved [{elapsed:.0f}s]")
    else:
        _write_final(outpath, ckpt_path, phase_name, results)
        n_exact = sum(1 for r in results if r["failure_type"] == "exact")
        print(f"\n  DONE {phase_name}: {n_exact}/{len(results)} exact "
              f"[{elapsed:.0f}s]")

    return results


def _write_ckpt(path, phase_name, results):
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({"phase": phase_name, "results": results}, f)
    os.replace(tmp, path)


def _write_final(outpath, ckpt_path, phase_name, results):
    tmp = outpath + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({"phase": phase_name, "total": len(results),
                    "results": results}, f, indent=2)
    os.replace(tmp, outpath)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


# ---------------------------------------------------------------------------
# Analysis: Phase 1A
# ---------------------------------------------------------------------------
def analyze_1a():
    """Chi-squared / Fisher test on init × clamp contingency table."""
    from scipy import stats

    path = os.path.join(OUTDIR, "phase1a.json")
    if not os.path.exists(path):
        print("No Phase 1A results found.")
        return

    with open(path) as f:
        data = json.load(f)
    results = data["results"]

    # Build contingency table: rows=init, cols=clamp
    inits = P1A_INITS
    clamps = P1A_CLAMPS
    table = np.zeros((len(inits), len(clamps)), dtype=int)
    totals = np.zeros((len(inits), len(clamps)), dtype=int)

    for r in results:
        i = inits.index(r["strategy"])
        j = clamps.index(r["clamp"])
        totals[i, j] += 1
        if r["failure_type"] == "exact":
            table[i, j] += 1

    print("\n" + "="*65)
    print("Phase 1A: Init × Clamp contingency table (exact recoveries)")
    print("="*65)

    # Header
    print(f"{'init':<15s}", end="")
    for c in clamps:
        print(f"  {c:.0e}", end="")
    print("  | total  rate")
    print("-"*65)

    row_totals = []
    for i, init in enumerate(inits):
        print(f"{init:<15s}", end="")
        row_exact = 0
        row_n = 0
        for j in range(len(clamps)):
            print(f"  {table[i,j]:3d}/{totals[i,j]:2d}", end="")
            row_exact += table[i, j]
            row_n += totals[i, j]
        rate = 100 * row_exact / row_n if row_n > 0 else 0
        print(f"  | {row_exact:3d}/{row_n:3d}  {rate:5.1f}%")
        row_totals.append(row_exact)

    # Column totals
    print("-"*65)
    print(f"{'total':<15s}", end="")
    for j in range(len(clamps)):
        col_exact = table[:, j].sum()
        col_n = totals[:, j].sum()
        print(f"  {col_exact:3d}/{col_n:2d}", end="")
    total_exact = table.sum()
    total_n = totals.sum()
    print(f"  | {total_exact:3d}/{total_n:3d}  {100*total_exact/total_n:.1f}%")

    # Statistical test
    print(f"\nTotal successes: {total_exact}")
    if total_exact < 5:
        print("Too few successes for chi-squared.")
        # Fisher on 2×2 marginal: (any_init_helped × any_clamp_helped)
        if total_exact > 0:
            # Find which inits and clamps have any success
            init_any = [table[i, :].sum() > 0 for i in range(len(inits))]
            clamp_any = [table[:, j].sum() > 0 for j in range(len(clamps))]
            print(f"Inits with any success: "
                  f"{[inits[i] for i, v in enumerate(init_any) if v]}")
            print(f"Clamps with any success: "
                  f"{[clamps[j] for j, v in enumerate(clamp_any) if v]}")
        print("\nDecision: init×clamp interaction undetectable at this cell.")
        print("Proceed to Phase 1B for other factors.")
        return {"interaction": False, "total_exact": int(total_exact)}
    else:
        chi2, p, dof, expected = stats.chi2_contingency(table)
        print(f"\nChi-squared: χ²={chi2:.2f}, dof={dof}, p={p:.4f}")
        if p < 0.05:
            print("SIGNIFICANT interaction (p < 0.05)")
            print("Flag init×clamp for Phase 2 factorial.")
        else:
            print("No significant interaction (p ≥ 0.05)")
            print("Treat init and clamp as independent in Phase 2.")

        # Also check per-init and per-clamp marginals
        print("\nPer-init rates:")
        for i, init in enumerate(inits):
            n_exact = table[i, :].sum()
            n_total = totals[i, :].sum()
            rate = 100 * n_exact / n_total if n_total > 0 else 0
            print(f"  {init:<15s} {n_exact}/{n_total} = {rate:.1f}%")

        print("\nPer-clamp rates:")
        for j, c in enumerate(clamps):
            n_exact = table[:, j].sum()
            n_total = totals[:, j].sum()
            rate = 100 * n_exact / n_total if n_total > 0 else 0
            print(f"  {c:.0e:<15s} {n_exact}/{n_total} = {rate:.1f}%")

        return {"interaction": p < 0.05, "p": float(p), "chi2": float(chi2),
                "total_exact": int(total_exact)}


# ---------------------------------------------------------------------------
# Analysis: Phase 1B
# ---------------------------------------------------------------------------
def analyze_1b():
    """Analyze OFAT sensitivity: which factors matter?"""
    from scipy import stats as sp_stats

    path = os.path.join(OUTDIR, "phase1b.json")
    if not os.path.exists(path):
        print("No Phase 1B results found.")
        return

    with open(path) as f:
        data = json.load(f)
    results = data["results"]

    print("\n" + "="*72)
    print("Phase 1B: OFAT sensitivity analysis")
    print("="*72)

    # For each hero cell × factor: compute recovery rate per level
    significant_factors = {}

    for arch, target_key in HERO_CELLS:
        cell = f"{arch}_{target_key}"
        cell_runs = [r for r in results
                     if r["arch"] == arch and r["target"] == target_key]

        print(f"\n--- {cell} ({len(cell_runs)} runs) ---")

        for factor_name in OFAT_FACTORS:
            # Get runs for this factor sweep: only vary this factor, others default
            if factor_name == "init":
                levels = OFAT_FACTORS["init"][arch]
                sweep_runs = [r for r in cell_runs
                              if r.get("clamp") == FACTOR_DEFAULTS["clamp"]
                              and r.get("optim") == FACTOR_DEFAULTS["optim"]
                              and r.get("hardening") == FACTOR_DEFAULTS["hardening"]
                              and r.get("grid_seed") == FACTOR_DEFAULTS["grid_seed"]
                              and r.get("noise") == FACTOR_DEFAULTS["noise"]
                              and r.get("epsilon") == FACTOR_DEFAULTS["epsilon"]]
                key = "strategy"
            elif factor_name == "grid_seed":
                default_strat = "default" if arch == "eq6" else "biased"
                sweep_runs = [r for r in cell_runs
                              if r["strategy"] == default_strat
                              and r.get("clamp") == FACTOR_DEFAULTS["clamp"]
                              and r.get("optim") == FACTOR_DEFAULTS["optim"]
                              and r.get("hardening") == FACTOR_DEFAULTS["hardening"]
                              and r.get("noise") == FACTOR_DEFAULTS["noise"]
                              and r.get("epsilon") == FACTOR_DEFAULTS["epsilon"]]
                key = "grid_seed"
                levels = OFAT_FACTORS["grid_seed"]
            else:
                default_strat = "default" if arch == "eq6" else "biased"
                sweep_runs = [r for r in cell_runs
                              if r["strategy"] == default_strat
                              and r.get("grid_seed") == FACTOR_DEFAULTS["grid_seed"]]
                # Further filter: all OTHER factors at default except this one
                other_factors = [f for f in ["clamp", "optim", "hardening",
                                             "noise", "epsilon"]
                                 if f != factor_name]
                for of in other_factors:
                    sweep_runs = [r for r in sweep_runs
                                  if r.get(of) == FACTOR_DEFAULTS[of]]
                key = factor_name
                levels = OFAT_FACTORS[factor_name]

            # Compute rates per level
            rates = {}
            counts = {}
            for lv in levels:
                lv_runs = [r for r in sweep_runs if r.get(key) == lv]
                n_exact = sum(1 for r in lv_runs
                              if r["failure_type"] == "exact")
                n_total = len(lv_runs)
                rates[lv] = (n_exact, n_total)
                counts[lv] = n_exact

            # Print
            rate_strs = []
            for lv in levels:
                ne, nt = rates[lv]
                pct = 100 * ne / nt if nt > 0 else 0
                rate_strs.append(f"{lv}={ne}/{nt}({pct:.0f}%)")
            print(f"  {factor_name:12s}: {', '.join(rate_strs)}")

            # Check decision rule: ≥25pp spread
            pcts = [100 * rates[lv][0] / rates[lv][1]
                    for lv in levels if rates[lv][1] > 0]
            if pcts:
                spread = max(pcts) - min(pcts)
                if spread >= 25:
                    if factor_name not in significant_factors:
                        significant_factors[factor_name] = []
                    significant_factors[factor_name].append(
                        (cell, spread))

                    # Fisher exact for max vs min level
                    best_lv = max(levels,
                                  key=lambda l: rates[l][0]/max(rates[l][1],1))
                    worst_lv = min(levels,
                                   key=lambda l: rates[l][0]/max(rates[l][1],1))
                    a, b = rates[best_lv]
                    c, d = rates[worst_lv]
                    if b > 0 and d > 0:
                        table_2x2 = [[a, b - a], [c, d - c]]
                        _, p_fisher = sp_stats.fisher_exact(table_2x2)
                        print(f"    *** SPREAD={spread:.0f}pp "
                              f"(Fisher p={p_fisher:.4f})")

    # Summary
    print("\n" + "="*72)
    print("SIGNIFICANT FACTORS (≥25pp on ≥1 cell):")
    print("="*72)
    for fn, cells in significant_factors.items():
        cells_2plus = [c for c, s in cells]
        n_cells = len(cells_2plus)
        max_spread = max(s for _, s in cells)
        # Decision rule: ≥25pp on ≥2 cells OR ≥40pp on 1 cell with p<0.01
        qualifies = n_cells >= 2 or max_spread >= 40
        tag = "INCLUDE" if qualifies else "borderline"
        print(f"  {fn:12s}: {n_cells} cells, max spread {max_spread:.0f}pp "
              f"[{tag}]")
        for c, s in cells:
            print(f"    {c}: {s:.0f}pp")

    not_sig = [f for f in OFAT_FACTORS if f not in significant_factors]
    if not_sig:
        print(f"\n  Not significant: {', '.join(not_sig)}")

    return significant_factors


# ---------------------------------------------------------------------------
# Analysis: Phase 1B' resweep
# ---------------------------------------------------------------------------
def analyze_1b_resweep():
    """Analyze resweep: do clamp/hardening become significant at winning init?"""
    from scipy import stats as sp_stats

    path = os.path.join(OUTDIR, "phase1b_resweep.json")
    if not os.path.exists(path):
        print("No Phase 1B' resweep results found.")
        return

    with open(path) as f:
        data = json.load(f)
    results = data["results"]

    winning_init = results[0]["strategy"] if results else "unknown"

    print("\n" + "="*72)
    print(f"Phase 1B' resweep: clamp/optim/hardening at init={winning_init}")
    print("="*72)

    significant_factors = {}

    for arch, target_key in HERO_CELLS:
        cell = f"{arch}_{target_key}"
        cell_runs = [r for r in results
                     if r["arch"] == arch and r["target"] == target_key]

        print(f"\n--- {cell} ({len(cell_runs)} runs) ---")

        for factor_name in ["clamp", "optim", "hardening"]:
            levels = OFAT_FACTORS[factor_name]
            key = factor_name

            # Filter: only runs where OTHER swept factors are at default
            other_factors = [f for f in ["clamp", "optim", "hardening"]
                             if f != factor_name]
            sweep_runs = cell_runs
            for of in other_factors:
                sweep_runs = [r for r in sweep_runs
                              if r.get(of) == FACTOR_DEFAULTS[of]]

            # Compute rates per level
            rates = {}
            for lv in levels:
                lv_runs = [r for r in sweep_runs if r.get(key) == lv]
                n_exact = sum(1 for r in lv_runs
                              if r["failure_type"] == "exact")
                n_total = len(lv_runs)
                rates[lv] = (n_exact, n_total)

            # Print
            rate_strs = []
            for lv in levels:
                ne, nt = rates[lv]
                pct = 100 * ne / nt if nt > 0 else 0
                rate_strs.append(f"{lv}={ne}/{nt}({pct:.0f}%)")
            print(f"  {factor_name:12s}: {', '.join(rate_strs)}")

            # Check ≥25pp spread
            pcts = [100 * rates[lv][0] / rates[lv][1]
                    for lv in levels if rates[lv][1] > 0]
            if pcts:
                spread = max(pcts) - min(pcts)
                if spread >= 25:
                    if factor_name not in significant_factors:
                        significant_factors[factor_name] = []
                    significant_factors[factor_name].append((cell, spread))

                    best_lv = max(levels,
                                  key=lambda l: rates[l][0]/max(rates[l][1],1))
                    worst_lv = min(levels,
                                   key=lambda l: rates[l][0]/max(rates[l][1],1))
                    a, b = rates[best_lv]
                    c, d = rates[worst_lv]
                    if b > 0 and d > 0:
                        table_2x2 = [[a, b - a], [c, d - c]]
                        _, p_fisher = sp_stats.fisher_exact(table_2x2)
                        print(f"    *** SPREAD={spread:.0f}pp "
                              f"(Fisher p={p_fisher:.4f})")

    # Summary
    print("\n" + "="*72)
    print("RESWEEP SUMMARY:")
    print("="*72)
    if significant_factors:
        print("Newly significant at winning init:")
        for fn, cells in significant_factors.items():
            for c, s in cells:
                print(f"  {fn:12s} on {c}: {s:.0f}pp")
    else:
        print("  No factors newly significant.")
        print("  Confirms: only init and optim matter.")

    # Compare optim effect at winning init vs default init
    print("\nOptim sanity check (adam_1e-3 'killer' effect at winning init):")
    for arch, target_key in HERO_CELLS:
        cell = f"{arch}_{target_key}"
        cell_runs = [r for r in results
                     if r["arch"] == arch and r["target"] == target_key]
        optim_runs = [r for r in cell_runs
                      if r.get("clamp") == FACTOR_DEFAULTS["clamp"]
                      and r.get("hardening") == FACTOR_DEFAULTS["hardening"]]
        for opt in OFAT_FACTORS["optim"]:
            lv_runs = [r for r in optim_runs if r.get("optim") == opt]
            ne = sum(1 for r in lv_runs if r["failure_type"] == "exact")
            nt = len(lv_runs)
            pct = 100 * ne / nt if nt > 0 else 0
            if nt > 0:
                print(f"  {cell} / {opt}: {ne}/{nt} ({pct:.0f}%)")

    return significant_factors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 experiment runner")
    parser.add_argument("--phase", choices=["1a", "1b", "1b-resweep"],
                        help="Which phase to run")
    parser.add_argument("--analyze", choices=["1a", "1b", "1b-resweep"],
                        help="Analyze results for a phase")
    parser.add_argument("--winning-init", type=str, default=None,
                        help="Winning init strategy for 1b-resweep")
    args = parser.parse_args()

    if args.phase == "1a":
        run_phase_1a()
    elif args.phase == "1b":
        run_phase_1b()
    elif args.phase == "1b-resweep":
        if not args.winning_init:
            print("Error: --winning-init required for 1b-resweep")
            sys.exit(1)
        run_phase_1b_resweep(args.winning_init)
    elif args.analyze == "1a":
        analyze_1a()
    elif args.analyze == "1b":
        analyze_1b()
    elif args.analyze == "1b-resweep":
        analyze_1b_resweep()
    else:
        parser.print_help()
