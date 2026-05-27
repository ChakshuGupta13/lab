"""Phase 2 experiment runner: init factorial across 18 focused cells.

Design: 18 cells × 4 init strategies × 64 seeds = 4,608 runs
(eq6 cells use all 5 strategies including 'default' scale-control → 5,056 total)
All other factors fixed at safe defaults: optim=adam_1e-2, hardening=power_law,
clamp=1e6, grid_seed=12345, noise=0, epsilon=1e-12.

Usage:
  python3 scratch/phase2_runner.py --run          # run Phase 2
  python3 scratch/phase2_runner.py --analyze      # analyze results
"""
import sys
import os
import json
import time
import signal
import argparse
import math
from multiprocessing import Pool
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scratch.run_taxonomy import (
    ALL_TARGETS, make_data, classify_failure, FACTOR_DEFAULTS,
    FOCUSED_CELLS, EQ6_INIT_STRATEGIES,
)
from scratch.sr_systematic import train_one
from master_formula import INIT_STRATEGIES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SEEDS = 64
SEED0 = 137
N_WORKERS = 6  # 14-core M4 Pro, OMP_NUM_THREADS=2, cap at 12 threads

# 4 common strategies for all archs; eq6 additionally gets 'default'
COMMON_INITS = INIT_STRATEGIES  # ["biased", "uniform", "xy_biased", "random_hot"]

OUTDIR = os.path.join(os.path.dirname(__file__), "phase2_results")

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
# Worker function (same as Phase 1)
# ---------------------------------------------------------------------------
def _run_one(args_tuple):
    """Single run: (seed, arch, target_key, strategy, factors_override, grid_seed) → result dict."""
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
# Phase 2: init factorial
# ---------------------------------------------------------------------------
def run_phase2():
    """18 cells × init strategies × 64 seeds."""
    tasks = []
    for arch, target_key in FOCUSED_CELLS:
        inits = EQ6_INIT_STRATEGIES if arch == "eq6" else COMMON_INITS
        for strat in inits:
            for seed in range(SEED0, SEED0 + N_SEEDS):
                tasks.append((seed, arch, target_key, strat, {},
                              FACTOR_DEFAULTS["grid_seed"]))

    print(f"Phase 2: init factorial on {len(FOCUSED_CELLS)} cells")
    print(f"  {len(tasks)} total runs")
    return _run_batch("phase2", tasks)


# ---------------------------------------------------------------------------
# Batch runner with checkpointing (same pattern as Phase 1)
# ---------------------------------------------------------------------------
def _run_batch(phase_name, tasks):
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
# Analysis
# ---------------------------------------------------------------------------
def analyze_phase2():
    """Analyze init effect across all 18 cells."""
    from scipy import stats as sp_stats

    path = os.path.join(OUTDIR, "phase2.json")
    if not os.path.exists(path):
        print("No Phase 2 results found.")
        return

    with open(path) as f:
        data = json.load(f)
    results = data["results"]

    all_inits = list(dict.fromkeys(
        EQ6_INIT_STRATEGIES + list(COMMON_INITS)))

    print("\n" + "="*80)
    print("Phase 2: Init factorial — recovery rates by cell × init strategy")
    print("="*80)

    # Header
    print(f"\n{'Cell':<25s}", end="")
    for init in all_inits:
        print(f" {init:>12s}", end="")
    print(f" {'spread':>8s} {'best_init':>12s} {'p(Fisher)':>10s}")
    print("-" * (25 + 13 * len(all_inits) + 35))

    summary = []
    for arch, target_key in FOCUSED_CELLS:
        cell = f"{arch}_{target_key}"
        cell_runs = [r for r in results
                     if r["arch"] == arch and r["target"] == target_key]

        rates = {}
        for init in all_inits:
            init_runs = [r for r in cell_runs if r["strategy"] == init]
            if not init_runs:
                rates[init] = (0, 0, None)
                continue
            n_exact = sum(1 for r in init_runs
                          if r["failure_type"] == "exact")
            n_total = len(init_runs)
            pct = 100 * n_exact / n_total
            rates[init] = (n_exact, n_total, pct)

        # Print row
        print(f"{cell:<25s}", end="")
        for init in all_inits:
            ne, nt, pct = rates[init]
            if nt == 0:
                print(f" {'—':>12s}", end="")
            else:
                print(f" {ne:>3d}/{nt:<3d}{pct:>5.1f}%", end="")
        
        # Spread and best init
        valid = [(init, pct) for init, (ne, nt, pct) in rates.items()
                 if pct is not None]
        if valid:
            pcts = [p for _, p in valid]
            spread = max(pcts) - min(pcts)
            best_init = max(valid, key=lambda x: x[1])[0]
            worst_init = min(valid, key=lambda x: x[1])[0]

            # Fisher exact: best vs worst
            p_val = None
            if spread >= 1:
                a, b, _ = rates[best_init]
                c, d, _ = rates[worst_init]
                if b > 0 and d > 0:
                    table_2x2 = [[a, b - a], [c, d - c]]
                    _, p_val = sp_stats.fisher_exact(table_2x2)

            print(f" {spread:>7.1f}pp", end="")
            print(f" {best_init:>12s}", end="")
            if p_val is not None:
                print(f" {p_val:>10.4f}", end="")
            else:
                print(f" {'':>10s}", end="")
            print()

            summary.append({
                "cell": cell, "arch": arch, "target": target_key,
                "spread": spread, "best_init": best_init,
                "worst_init": worst_init, "p_val": p_val,
                "rates": {init: rates[init] for init in all_inits},
            })
        else:
            print(f" {'—':>7s} {'—':>12s} {'—':>10s}")
            print()

    # Classification summary
    print("\n" + "="*80)
    print("SUMMARY BY ARCHITECTURE")
    print("="*80)

    for arch_name in ["eq6", "v16", "hybrid"]:
        arch_cells = [s for s in summary if s["arch"] == arch_name]
        if not arch_cells:
            continue
        print(f"\n  {arch_name.upper()} ({len(arch_cells)} cells):")

        # Group by behavior
        dead = [s for s in arch_cells
                if all(rates[2] is not None and rates[2] == 0
                       for rates in s["rates"].values()
                       if rates[2] is not None)]
        ceiling = [s for s in arch_cells
                   if all(rates[2] is not None and rates[2] >= 95
                          for rates in s["rates"].values()
                          if rates[2] is not None)]
        init_sensitive = [s for s in arch_cells
                          if s["spread"] >= 25
                          and s not in dead and s not in ceiling]
        stable = [s for s in arch_cells
                  if s not in dead and s not in ceiling
                  and s not in init_sensitive]

        if dead:
            print(f"    Dead (0% all inits): {', '.join(s['cell'] for s in dead)}")
        if ceiling:
            print(f"    Ceiling (≥95% all inits): {', '.join(s['cell'] for s in ceiling)}")
        if init_sensitive:
            for s in init_sensitive:
                print(f"    Init-sensitive: {s['cell']} "
                      f"(spread={s['spread']:.0f}pp, "
                      f"best={s['best_init']}, worst={s['worst_init']})")
        if stable:
            for s in stable:
                best_pct = max(p for _, _, p in s["rates"].values() if p is not None)
                print(f"    Stable: {s['cell']} "
                      f"(spread={s['spread']:.0f}pp, peak={best_pct:.0f}%)")

    # Init strategy ranking (across cells with spread ≥ 10pp)
    print("\n" + "="*80)
    print("INIT STRATEGY RANKING (cells with spread ≥ 10pp)")
    print("="*80)

    sensitive = [s for s in summary if s["spread"] >= 10]
    if sensitive:
        win_counts = defaultdict(int)
        for s in sensitive:
            win_counts[s["best_init"]] += 1

        print(f"\n  Best-init frequency ({len(sensitive)} cells with ≥10pp spread):")
        for init, count in sorted(win_counts.items(), key=lambda x: -x[1]):
            print(f"    {init:<15s}: best on {count}/{len(sensitive)} cells")

        # Direction analysis
        print(f"\n  Direction of init effect (≥25pp cells):")
        big_spread = [s for s in summary if s["spread"] >= 25]
        for s in big_spread:
            valid_rates = {init: pct for init, (ne, nt, pct) in s["rates"].items()
                          if pct is not None}
            parts = [f"{init}={pct:.0f}%" for init, pct in
                     sorted(valid_rates.items(), key=lambda x: -x[1])]
            print(f"    {s['cell']:<25s}: {' > '.join(parts)}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 experiment runner")
    parser.add_argument("--run", action="store_true", help="Run Phase 2")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze Phase 2 results")
    args = parser.parse_args()

    if args.run:
        run_phase2()
    elif args.analyze:
        analyze_phase2()
    else:
        parser.print_help()
