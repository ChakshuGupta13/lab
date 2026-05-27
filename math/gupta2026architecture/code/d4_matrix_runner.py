"""Depth-4 matrix: Eq.6 + Hybrid on 3 EML targets (LR/RL/RR).

Captures full per-run data (JSON + CSV) so nothing needs re-running.

Usage — run one cell at a time (for parallelism):
    OMP_NUM_THREADS=2 python3 scratch/d4_matrix_runner.py --cell eq6_d4_paper
    OMP_NUM_THREADS=2 python3 scratch/d4_matrix_runner.py --cell hybrid_d4_T1

Valid cell names:
    eq6_d4_paper, eq6_d4_T1, eq6_d4_T4
    hybrid_d4_paper, hybrid_d4_T1, hybrid_d4_T4

Or run all cells sequentially:
    OMP_NUM_THREADS=2 python3 scratch/d4_matrix_runner.py --all
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from master_formula import (
    EMLTreeV16, EmlMasterFormula, EMLTreeHybrid,
    INIT_STRATEGIES,
    eml_torch,
)
# Eq.6 uses single strategy "uniform" to match existing d3/d4 experiments
EQ6_STRATEGIES = ["uniform"]
from scratch.sr_systematic import (
    filter_real_domain, train_one,
    TRAIN_LO, TRAIN_HI, TRAIN_STEP, GEN_LO, GEN_HI, GEN_N,
)

# ── Constants ────────────────────────────────────────────────────────────
SEED0 = 137
N_SEEDS = 64
DEPTH = 4
RESULTS_DIR = "scratch/d4_matrix_results"
CSV_PATH = os.path.join(RESULTS_DIR, "d4_runs.csv")

CSV_FIELDS = [
    "seed", "initialization_strategy", "architecture", "depth",
    "target", "topology", "operator_family",
    "training_rmse", "heldout_rmse", "best_loss",
    "hardened_expression", "tree_signature",
    "structural_match", "failure_type",
    "nan_restarts", "wall_seconds",
]

# ── Depth-4 EML targets ─────────────────────────────────────────────────
e = np.exp(1.0)

D4_TARGETS = {
    "d4_paper": {
        "desc": "eml(1,eml(eml(1,eml(x,y)),1))  d4-LR",
        "topology": "LR",
        "fn": lambda x, y: np.log(np.exp(x) - np.log(y)),
    },
    "d4_T1": {
        "desc": "eml(eml(1,eml(eml(x,y),1)),1)  d4-RL",
        "topology": "RL",
        "fn": lambda x, y: np.exp(e - np.exp(x) + np.log(y)),
    },
    "d4_T4": {
        "desc": "eml(1,eml(1,eml(1,eml(x,y))))  d4-RR",
        "topology": "RR",
        "fn": lambda x, y: e - np.log(e - np.log(e - np.log(np.exp(x) - np.log(y)))),
    },
}

ARCHS = ["eq6", "v16", "hybrid"]

# ── Helpers ──────────────────────────────────────────────────────────────

def make_data(target_fn):
    """Prepare train + held-out data for a target function."""
    xs = np.arange(TRAIN_LO, TRAIN_HI + TRAIN_STEP * 0.5, TRAIN_STEP)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    xx, yy = xx.ravel(), yy.ravel()
    xx, yy, tt, n = filter_real_domain(xx, yy, target_fn)
    x_train = torch.tensor(xx, dtype=torch.float64)
    y_train = torch.tensor(yy, dtype=torch.float64)
    from master_formula import DTYPE
    t_train = torch.tensor(tt, dtype=DTYPE)

    rng = np.random.default_rng(12345)
    oversample = max(GEN_N * 4, GEN_N + 5000)
    xg = rng.uniform(GEN_LO, GEN_HI, size=oversample)
    yg = rng.uniform(GEN_LO, GEN_HI, size=oversample)
    xg, yg, tg, _ = filter_real_domain(xg, yg, target_fn)
    if len(xg) > GEN_N:
        xg, yg, tg = xg[:GEN_N], yg[:GEN_N], tg[:GEN_N]
    x_gen = torch.tensor(xg, dtype=torch.float64)
    y_gen = torch.tensor(yg, dtype=torch.float64)
    t_gen = torch.tensor(tg, dtype=DTYPE)
    return x_train, y_train, t_train, len(xx), x_gen, y_gen, t_gen, len(xg)


def _tree_signature(desc):
    """Extract internal/leaf routing from model.describe() output."""
    import re
    if not desc:
        return ""
    internal = leaves = ""
    m = re.search(r"Internal routing:\s*([^\n|]+)", desc)
    if m:
        internal = m.group(1).strip()
    m = re.search(r"Leaf routing:\s*([^\n|]+)", desc)
    if m:
        leaves = m.group(1).strip()
    return f"internal=[{internal}] leaves=[{leaves}]"


def _failure_type(r):
    """Classify run outcome."""
    mse_t = r.get("mse_train", float("inf"))
    if r.get("success", False):
        return "exact"
    if np.isfinite(mse_t) and mse_t < 0.01:
        return "hardening_failure"
    return "training_failure"


def _safe_rmse(mse):
    if mse is None:
        return ""
    if isinstance(mse, float) and (math.isnan(mse) or math.isinf(mse)):
        return repr(mse)
    return f"{math.sqrt(max(mse, 0.0)):.6e}"


# ── Run one cell ─────────────────────────────────────────────────────────

def run_cell(arch: str, target_name: str):
    """Run one d4 cell, save per-run JSON and append to CSV."""
    tgt = D4_TARGETS[target_name]
    target_fn = tgt["fn"]
    topology = tgt["topology"]

    print(f"=== {arch} × {target_name} (d4, {topology}) ===")
    xt, yt, tt, nt, xg, yg, tg, ng = make_data(target_fn)
    print(f"  Training points: {nt}, held-out points: {ng}")

    strategies = INIT_STRATEGIES if arch != "eq6" else EQ6_STRATEGIES
    runs = []
    n_ok = 0
    total = 0
    t0 = time.time()

    for seed in range(SEED0, SEED0 + N_SEEDS):
        for strat in strategies:
            t_run = time.time()
            r = train_one(
                seed, DEPTH, strat, xt, yt, tt, xg, yg, tg,
                verbose=False, arch=arch, op_fn=eml_torch,
            )
            wall_s = time.time() - t_run
            ft = _failure_type(r)
            if ft == "exact":
                n_ok += 1
            total += 1

            desc = r.get("description", "") or ""
            run_record = {
                "seed": r["seed"],
                "initialization_strategy": strat,
                "architecture": arch,
                "depth": DEPTH,
                "target": target_name,
                "topology": topology,
                "operator_family": "EML",
                "mse_train": r.get("mse_train"),
                "mse_gen": r.get("mse_gen"),
                "best_loss": r.get("best_loss"),
                "hardened_expression": desc,
                "tree_signature": _tree_signature(desc),
                "failure_type": ft,
                "nan_restarts": r.get("nan_restarts", 0),
                "wall_seconds": round(wall_s, 2),
            }
            runs.append(run_record)

        # Progress every 8 seeds
        if (seed - SEED0 + 1) % 8 == 0:
            elapsed = time.time() - t0
            rate_pct = 100.0 * n_ok / total if total else 0
            print(f"  seed {seed}: {n_ok}/{total} = {rate_pct:.1f}%  "
                  f"({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    rate_pct = 100.0 * n_ok / total if total else 0

    # ── Save per-run JSON ────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cell_key = f"{arch}_d4_{target_name.replace('d4_', '')}"
    json_path = os.path.join(RESULTS_DIR, f"{cell_key}.json")
    payload = {
        "cell": cell_key,
        "arch": arch,
        "target": target_name,
        "topology": topology,
        "depth": DEPTH,
        "operator": "EML",
        "n_seeds": N_SEEDS,
        "seed0": SEED0,
        "strategies": strategies,
        "n_ok": n_ok,
        "total": total,
        "rate": n_ok / total if total else 0,
        "elapsed_seconds": round(elapsed, 1),
        "n_train": nt,
        "n_gen": ng,
        "runs": runs,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"  Saved: {json_path} ({len(runs)} runs)")

    # ── Append to CSV ────────────────────────────────────────────────
    write_header = (not os.path.exists(CSV_PATH)
                    or os.path.getsize(CSV_PATH) == 0)
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for run in runs:
            writer.writerow({
                "seed": run["seed"],
                "initialization_strategy": run["initialization_strategy"],
                "architecture": arch,
                "depth": DEPTH,
                "target": target_name,
                "topology": topology,
                "operator_family": "EML",
                "training_rmse": _safe_rmse(run["mse_train"]),
                "heldout_rmse": _safe_rmse(run["mse_gen"]),
                "best_loss": run["best_loss"],
                "hardened_expression": run["hardened_expression"].replace("\n", " | "),
                "tree_signature": run["tree_signature"],
                "structural_match": "yes" if run["failure_type"] == "exact" else "no",
                "failure_type": run["failure_type"],
                "nan_restarts": run["nan_restarts"],
                "wall_seconds": run["wall_seconds"],
            })
    print(f"  Appended {len(runs)} rows to {CSV_PATH}")

    # ── Strategy breakdown ───────────────────────────────────────────
    from collections import Counter
    strat_ok = Counter()
    strat_tot = Counter()
    for run in runs:
        s = run["initialization_strategy"]
        strat_tot[s] += 1
        if run["failure_type"] == "exact":
            strat_ok[s] += 1
    print(f"\n  RESULT: {arch} × {target_name} ({topology}): "
          f"{n_ok}/{total} = {rate_pct:.1f}%  ({elapsed:.0f}s)")
    for s in strategies:
        print(f"    {s}: {strat_ok[s]}/{strat_tot[s]}")
    print()

    return n_ok, total


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Depth-4 matrix: Eq.6 + Hybrid × LR/RL/RR")
    parser.add_argument("--cell", type=str, default=None,
        help="Run one cell: eq6_d4_paper, hybrid_d4_T1, etc.")
    parser.add_argument("--all", action="store_true",
        help="Run all 6 cells sequentially")
    parser.add_argument("--dry-run", action="store_true",
        help="Show what would be run")
    args = parser.parse_args()

    all_cells = []
    for arch in ARCHS:
        for tgt_name in D4_TARGETS:
            cell_key = f"{arch}_{tgt_name}"
            all_cells.append((cell_key, arch, tgt_name))

    if args.dry_run:
        print("Depth-4 matrix cells:")
        for key, arch, tgt in all_cells:
            info = D4_TARGETS[tgt]
            n = N_SEEDS * (len(INIT_STRATEGIES) if arch != "eq6"
                           else len(EQ6_STRATEGIES))
            est_min = n * 10 / 60
            print(f"  {key:25s}  {info['topology']:3s}  "
                  f"{n:4d} runs  ~{est_min:.0f} min")
        return

    if args.cell:
        # Parse cell name: "eq6_d4_paper" → arch="eq6", target="d4_paper"
        parts = args.cell.split("_", 1)
        arch = parts[0]
        tgt_name = parts[1] if len(parts) > 1 else ""
        if arch not in ARCHS:
            print(f"Unknown arch '{arch}'. Valid: {ARCHS}")
            sys.exit(1)
        if tgt_name not in D4_TARGETS:
            print(f"Unknown target '{tgt_name}'. Valid: {list(D4_TARGETS.keys())}")
            sys.exit(1)
        run_cell(arch, tgt_name)
    elif args.all:
        for key, arch, tgt in all_cells:
            run_cell(arch, tgt)
    else:
        parser.print_help()
        print("\nUse --dry-run to see cells, --cell X to run one, --all to run all.")


if __name__ == "__main__":
    main()
