#!/usr/bin/env python3
"""Reproduce experiments from:

    "Why Architecture Choice Matters in Symbolic Regression"
    Chakshu Gupta, arXiv:2604.23256, April 2026.

Self-contained script. Requires: PyTorch >= 2.0, NumPy, matplotlib (optional).
All dependencies are in master_formula.py and eml.py (same directory).

Usage
-----
    # Run a single experimental cell (e.g., V16 on the paper target, depth 3):
    python reproduce.py --arch v16 --target paper --depth 3 --seeds 64

    # Run the full d3 heatmap matrix (Table 1 in the paper):
    python reproduce.py --mode heatmap --seeds 64

    # Reproduce the gradient trajectory (Figure 2):
    python reproduce.py --mode gradient --seeds 10

    # Verify against pre-computed results in results.json:
    python reproduce.py --mode verify
"""

import sys
import os
import math
import json
import time
import argparse
import numpy as np
import torch

# Ensure this script finds the modules in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from master_formula import (
    EmlMasterFormula, EMLTreeV16, EMLTreeHybrid,
    eml_torch, sml_torch, rml_torch,
    INIT_STRATEGIES, DTYPE,
)


# =====================================================================
# Hyperparameters (matching the author's v16_final defaults)
# =====================================================================
SEARCH_ITERS = 6000
HARDENING_ITERS = 2000
LR = 0.01
GRAD_CLIP = 1.0
TAU_SEARCH = 2.5
TAU_HARD = 0.01
HARDENING_TAU_POWER = 2.0
HARDENING_LR_FLOOR = 0.01
LAM_ENT_HARD = 0.02
LAM_BIN_HARD = 0.02
PATIENCE = 4200
PATIENCE_THR = 1e-2
PLATEAU_RTOL = 1e-3
NAN_RESTART_PATIENCE = 50
MAX_NAN_RESTARTS = 100

SYMBOL_THR = 1e-20

TRAIN_LO, TRAIN_HI, TRAIN_STEP = 1.0, 3.0, 0.1
GEN_LO, GEN_HI = 0.5, 5.0
GEN_N = 500

SEED0 = 137
EVAL_EVERY = 200

e_val = np.exp(1.0)


# =====================================================================
# EML helper (numpy, for target definitions)
# =====================================================================
def _eml(a, b):
    return np.exp(a) - np.log(b)


def _sml(a, b):
    return np.sinh(a) - np.arctan(b)


def _rml(a, b):
    return np.arctan(a) - np.sinh(b)


# =====================================================================
# Targets — every cell from Table 1 and Table 5 in the paper
# =====================================================================
TARGETS = {
    # --- EML chain targets (Table 1, Figure 1) ---
    "paper":   ("EML LR(y,x)", "eml",
                lambda x, y: _eml(_eml(1.0, _eml(y, x)), 1.0)),
    "T1":      ("EML RL(x,y)", "eml",
                lambda x, y: e_val - np.exp(x) + np.log(y)),
    "T4":      ("EML RR(x,y)", "eml",
                lambda x, y: e_val - np.log(e_val - np.log(np.exp(x) - np.log(y)))),
    "T8":      ("EML LR(x,y)", "eml",
                lambda x, y: np.exp(e_val - np.log(np.exp(x) - np.log(y)))),
    "T1_yx":   ("EML RL(y,x)", "eml",
                lambda x, y: e_val - np.exp(y) + np.log(x)),
    "T4_yx":   ("EML RR(y,x)", "eml",
                lambda x, y: e_val - np.log(e_val - np.log(np.exp(y) - np.log(x)))),

    # --- EML balanced targets (Table 5) ---
    "E_Bal":    ("EML Bal(x,y) both-ln", "eml",
                 lambda x, y: _eml(_eml(1.0, x), _eml(1.0, y))),
    "E_Bal_yx": ("EML Bal(y,x) both-ln", "eml",
                 lambda x, y: _eml(_eml(1.0, y), _eml(1.0, x))),
    "E_Bal_mx": ("EML Bal x-ln y-exp", "eml",
                 lambda x, y: _eml(_eml(1.0, x), _eml(y, 1.0))),
    "E_Bal_ms": ("EML Bal y-ln x-exp", "eml",
                 lambda x, y: _eml(_eml(1.0, y), _eml(x, 1.0))),

    # --- SML chain targets ---
    "S_LR":    ("SML LR(y,x)", "sml",
                lambda x, y: _sml(_sml(1.0, _sml(y, x)), 1.0)),
    "S_LR_xy": ("SML LR(x,y)", "sml",
                lambda x, y: _sml(_sml(1.0, _sml(x, y)), 1.0)),
    "S_RL":    ("SML RL(x,y)", "sml",
                lambda x, y: _sml(1.0, _sml(_sml(x, y), 1.0))),
    "S_RL_yx": ("SML RL(y,x)", "sml",
                lambda x, y: _sml(1.0, _sml(_sml(y, x), 1.0))),
    "S_RR":    ("SML RR(x,y)", "sml",
                lambda x, y: _sml(1.0, _sml(1.0, _sml(x, y)))),
    "S_RR_yx": ("SML RR(y,x)", "sml",
                lambda x, y: _sml(1.0, _sml(1.0, _sml(y, x)))),

    # --- SML balanced targets (Table 5) ---
    "S_Bal":    ("SML Bal(x,y) both-atan", "sml",
                 lambda x, y: _sml(_sml(1.0, x), _sml(1.0, y))),
    "S_Bal_yx": ("SML Bal(y,x) both-atan", "sml",
                 lambda x, y: _sml(_sml(1.0, y), _sml(1.0, x))),
    "S_Bal_mx": ("SML Bal x-atan y-sinh", "sml",
                 lambda x, y: _sml(_sml(1.0, x), _sml(y, 1.0))),
    "S_Bal_ms": ("SML Bal y-atan x-sinh", "sml",
                 lambda x, y: _sml(_sml(1.0, y), _sml(x, 1.0))),

    # --- RML chain targets ---
    "R_LR":    ("RML LR(y,x)", "rml",
                lambda x, y: _rml(_rml(1.0, _rml(y, x)), 1.0)),
    "R_LR_xy": ("RML LR(x,y)", "rml",
                lambda x, y: _rml(_rml(1.0, _rml(x, y)), 1.0)),
    "R_RL":    ("RML RL(x,y)", "rml",
                lambda x, y: _rml(1.0, _rml(_rml(x, y), 1.0))),
    "R_RL_yx": ("RML RL(y,x)", "rml",
                lambda x, y: _rml(1.0, _rml(_rml(y, x), 1.0))),
}

OP_FNS = {"eml": eml_torch, "sml": sml_torch, "rml": rml_torch}


# =====================================================================
# Data generation
# =====================================================================
def filter_real_domain(x, y, target_fn, imag_tol=1e-12):
    """Evaluate target in complex128, keep only finite real-valued points."""
    xc = x.astype(np.complex128)
    yc = y.astype(np.complex128)
    with np.errstate(all="ignore"):
        tc = target_fn(xc, yc)
    mask = (np.abs(tc.imag) < imag_tol) & np.isfinite(tc.real)
    return x[mask], y[mask], tc[mask].real, int(mask.sum())


def make_train_data(target_fn):
    xs = np.arange(TRAIN_LO, TRAIN_HI + TRAIN_STEP * 0.5, TRAIN_STEP)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    xx, yy = xx.ravel(), yy.ravel()
    xx, yy, tt, n = filter_real_domain(xx, yy, target_fn)
    return (torch.tensor(xx, dtype=torch.float64),
            torch.tensor(yy, dtype=torch.float64),
            torch.tensor(tt, dtype=DTYPE), n)


def make_gen_data(target_fn, seed=12345):
    rng = np.random.default_rng(seed)
    oversample = max(GEN_N * 4, GEN_N + 5000)
    x = rng.uniform(GEN_LO, GEN_HI, size=oversample)
    y = rng.uniform(GEN_LO, GEN_HI, size=oversample)
    x, y, t, _ = filter_real_domain(x, y, target_fn)
    if len(x) > GEN_N:
        x, y, t = x[:GEN_N], y[:GEN_N], t[:GEN_N]
    return (torch.tensor(x, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64),
            torch.tensor(t, dtype=DTYPE), len(x))


# =====================================================================
# Loss computation
# =====================================================================
def compute_losses(pred, target, leaf_probs, gate_probs, lam_ent, lam_bin):
    data_loss = torch.mean((pred - target).abs() ** 2).real
    eps = 1e-12

    leaf_max = leaf_probs.max(dim=1).values
    leaf_unc = torch.clamp((1.0 - leaf_max) / (2.0 / 3.0), 0.0, 1.0)
    leaf_ent = -(leaf_probs * (leaf_probs + eps).log()).sum(dim=1)
    entropy = (leaf_ent * leaf_unc).mean()

    gate_unc = torch.clamp(1.0 - (2.0 * gate_probs - 1.0).abs(), 0.0, 1.0)
    gate_bin = gate_probs * (1.0 - gate_probs)
    binarity = (gate_bin * gate_unc).mean()

    total = data_loss + lam_ent * entropy + lam_bin * binarity
    return total, data_loss


# =====================================================================
# Training one seed
# =====================================================================
def train_one(seed, depth, strategy, x_train, y_train, t_train,
              x_gen, y_gen, t_gen, arch="v16", op_fn=None, verbose=False):
    """Train one seed. Returns dict with success, mse_train, description."""
    torch.manual_seed(seed)
    if arch == "v16":
        model = EMLTreeV16(depth=depth, init_strategy=strategy, op_fn=op_fn)
    elif arch == "hybrid":
        model = EMLTreeHybrid(depth=depth, init_strategy=strategy, op_fn=op_fn)
    else:  # eq6
        model = EmlMasterFormula(depth=depth, n_vars=2, op_fn=op_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")
    best_state = None
    nan_streak = 0
    nan_restarts = 0
    plateau_counter = 0
    total_iters = SEARCH_ITERS + HARDENING_ITERS
    hard_step = 0
    phase = "search"

    for it in range(1, total_iters + 1):
        if nan_restarts >= MAX_NAN_RESTARTS:
            break

        if phase == "search":
            if it > SEARCH_ITERS or (
                plateau_counter >= PATIENCE and best_loss < PATIENCE_THR
            ):
                phase = "harden"
                hard_step = 0
                if best_state is not None:
                    model.load_state_dict(best_state)
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        if phase == "search":
            tau = TAU_SEARCH
            lam_ent = lam_bin = 0.0
            lr_mult = 1.0
        else:
            if hard_step >= HARDENING_ITERS:
                break
            t_frac = hard_step / max(1, HARDENING_ITERS)
            t_pow = t_frac ** HARDENING_TAU_POWER
            tau = TAU_SEARCH * (TAU_HARD / TAU_SEARCH) ** t_pow
            lam_ent = t_frac * LAM_ENT_HARD
            lam_bin = t_frac * LAM_BIN_HARD
            lr_mult = max(HARDENING_LR_FLOOR, (1.0 - t_frac) ** 2)
            hard_step += 1

        optimizer.param_groups[0]["lr"] = LR * lr_mult
        optimizer.zero_grad()

        pred, leaf_probs, gate_probs = model(
            x_train, y_train, tau_leaf=tau, tau_gate=tau)

        # Architecture-specific regularization
        if arch == "eq6" and gate_probs.numel() > 0:
            eps = 1e-12
            int_ent = -(gate_probs * (gate_probs + eps).log()).sum(dim=1)
            int_max = gate_probs.max(dim=1).values
            int_unc = torch.clamp((1.0 - int_max) / (2.0 / 3.0), 0.0, 1.0)
            extra_ent = (int_ent * int_unc).mean()
            dummy = torch.sigmoid(torch.zeros(1))
            total, data_loss = compute_losses(
                pred, t_train, leaf_probs, dummy, lam_ent, 0.0)
            total = total + lam_bin * extra_ent
        elif arch == "hybrid":
            total, data_loss = compute_losses(
                pred, t_train, leaf_probs, gate_probs, lam_ent, lam_bin)
            root_probs = torch.softmax(model.root_logits / tau, dim=1)
            eps = 1e-12
            root_ent = -(root_probs * (root_probs + eps).log()).sum(dim=1)
            total = total + lam_bin * root_ent.mean()
        else:
            total, data_loss = compute_losses(
                pred, t_train, leaf_probs, gate_probs, lam_ent, lam_bin)

        if not torch.isfinite(total):
            nan_streak += 1
            plateau_counter += 1
            if nan_streak >= NAN_RESTART_PATIENCE:
                if best_state is not None:
                    model.load_state_dict(best_state)
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=LR * lr_mult)
                nan_streak = 0
                nan_restarts += 1
            continue

        nan_streak = 0
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        dl = float(data_loss.item())
        if np.isfinite(dl) and dl < best_loss:
            rel_imp = (best_loss - dl) / max(best_loss, 1e-15)
            best_loss = dl
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            plateau_counter = 0 if rel_imp > PLATEAU_RTOL else plateau_counter + 1
        else:
            plateau_counter += 1

        if verbose and it % EVAL_EVERY == 0:
            rmse = math.sqrt(max(dl, 0.0))
            tag = " [H]" if phase == "harden" else ""
            print(f"  seed={seed:3d} it={it:5d} rmse={rmse:.3e} "
                  f"tau={tau:.4f}{tag}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.snap_weights()

    with torch.no_grad():
        pred_train, _, _ = model(x_train, y_train, tau_leaf=0.01, tau_gate=0.01)
        mse_train = torch.mean(
            (pred_train - t_train).abs() ** 2).real.item()
        pred_gen, _, _ = model(x_gen, y_gen, tau_leaf=0.01, tau_gate=0.01)
        mse_gen = torch.mean(
            (pred_gen - t_gen).abs() ** 2).real.item()

    success = np.isfinite(mse_train) and mse_train < SYMBOL_THR

    if verbose:
        rmse_t = math.sqrt(max(mse_train, 0.0))
        tag = "EXACT" if success else "fail"
        print(f"  seed={seed:3d} SNAP: train={rmse_t:.3e} [{tag}]")

    return {
        "seed": seed,
        "strategy": strategy,
        "mse_train": mse_train,
        "mse_gen": mse_gen,
        "success": success,
        "description": model.describe(),
    }


# =====================================================================
# Run a single experimental cell
# =====================================================================
def run_cell(arch, target_name, depth=3, n_seeds=64, verbose=False):
    """Run one cell of the matrix: arch x target."""
    desc, op_key, fn = TARGETS[target_name]
    op_fn = OP_FNS[op_key]

    x_train, y_train, t_train, n_train = make_train_data(fn)
    x_gen, y_gen, t_gen, n_gen = make_gen_data(fn)

    strategies = ["uniform"] if arch == "eq6" else INIT_STRATEGIES
    n_runs = n_seeds * len(strategies)

    print(f"\n{arch} x {target_name} ({desc}), depth={depth}, "
          f"{n_runs} runs ({n_seeds} seeds x {len(strategies)} strategies)")

    successes = 0
    t0 = time.time()

    for seed in range(SEED0, SEED0 + n_seeds):
        for strategy in strategies:
            result = train_one(
                seed, depth, strategy,
                x_train, y_train, t_train,
                x_gen, y_gen, t_gen,
                arch=arch, op_fn=op_fn, verbose=verbose)
            if result["success"]:
                successes += 1

    elapsed = time.time() - t0
    rate = 100.0 * successes / n_runs
    print(f"  Result: {successes}/{n_runs} ({rate:.1f}%) "
          f"in {elapsed:.0f}s")
    return {"arch": arch, "target": target_name, "depth": depth,
            "successes": successes, "total": n_runs, "rate": rate}


# =====================================================================
# Full heatmap matrix (Table 1 / Figure 1)
# =====================================================================
HEATMAP_CELLS = {
    # EML chain targets
    "eml": [
        ("paper", "LR(y,x)"), ("T8", "LR(x,y)"),
        ("T1", "RL(x,y)"), ("T1_yx", "RL(y,x)"),
        ("T4", "RR(x,y)"), ("T4_yx", "RR(y,x)"),
    ],
    # EML balanced targets (Table 5; Hybrid not tested)
    "eml_bal": [
        ("E_Bal", "Bal(x,y)"), ("E_Bal_yx", "Bal(y,x)"),
        ("E_Bal_mx", "Bal_mx"), ("E_Bal_ms", "Bal_ms"),
    ],
    # SML chain targets
    "sml": [
        ("S_LR", "LR(y,x)"), ("S_LR_xy", "LR(x,y)"),
        ("S_RL", "RL(x,y)"), ("S_RL_yx", "RL(y,x)"),
        ("S_RR", "RR(x,y)"), ("S_RR_yx", "RR(y,x)"),
    ],
    # SML balanced targets (Table 5)
    "sml_bal": [
        ("S_Bal", "Bal(x,y)"), ("S_Bal_yx", "Bal(y,x)"),
        ("S_Bal_mx", "Bal_mx"), ("S_Bal_ms", "Bal_ms"),
    ],
    # RML chain targets
    "rml": [
        ("R_LR", "LR(y,x)"), ("R_LR_xy", "LR(x,y)"),
        ("R_RL", "RL(x,y)"), ("R_RL_yx", "RL(y,x)"),
    ],
}

ARCHS = ["eq6", "v16", "hybrid"]


def run_heatmap(n_seeds=64):
    """Run the full heatmap matrix (Tables 1 and 5)."""
    results = {}
    for op_key, cells in HEATMAP_CELLS.items():
        # EML balanced: Hybrid not tested (paper Table 5)
        archs = ["eq6", "v16"] if op_key == "eml_bal" else ARCHS
        for target_name, label in cells:
            for arch in archs:
                cell = run_cell(arch, target_name, depth=3, n_seeds=n_seeds)
                results[f"{arch}/{target_name}"] = cell
    return results


# =====================================================================
# Gradient trajectory (Figure 2)
# =====================================================================
def measure_gradient_trajectory(n_seeds=10,
                                checkpoints=(100, 200, 300, 500, 750,
                                             1000, 1500, 2000)):
    """Measure leaf gradient x/y ratio at checkpoints for Eq.6 on paper + T1."""
    targets = {
        "paper": TARGETS["paper"][2],  # lambda
        "T1":    TARGETS["T1"][2],
    }

    results = {}
    for tkey, fn in targets.items():
        x_t, y_t, t_t, _ = make_train_data(fn)
        cp_set = set(checkpoints)
        ratios = {cp: [] for cp in checkpoints}

        for seed_idx in range(n_seeds):
            seed = SEED0 + seed_idx
            torch.manual_seed(seed)
            model = EmlMasterFormula(depth=3, n_vars=2, op_fn=eml_torch)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            for it in range(1, max(checkpoints) + 1):
                optimizer.zero_grad()
                pred, lp, gp = model(
                    x_t, y_t, tau_leaf=TAU_SEARCH, tau_gate=TAU_SEARCH)
                data_loss = torch.mean((pred - t_t).abs() ** 2).real

                if not torch.isfinite(data_loss):
                    continue

                data_loss.backward()

                if it in cp_set:
                    for n, p in model.named_parameters():
                        if n == "leaf_logits" and p.grad is not None:
                            leaf_x = p.grad.data[:, 1].norm(2).item()
                            leaf_y = p.grad.data[:, 2].norm(2).item()
                            ratio = (leaf_x / leaf_y
                                     if leaf_y > 1e-15 else float('inf'))
                            ratios[it].append(ratio)

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        means = [np.mean(ratios[cp]) for cp in checkpoints]
        sems = [np.std(ratios[cp]) / np.sqrt(len(ratios[cp]))
                for cp in checkpoints]
        results[tkey] = {
            "checkpoints": list(checkpoints),
            "means": means,
            "sems": sems,
        }
        print(f"  {tkey}: measured {n_seeds} seeds")

    return results


def plot_gradient(results, outdir="."):
    """Plot gradient trajectory (requires matplotlib with pgf backend)."""
    import matplotlib
    matplotlib.use("pgf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "figure.dpi": 200,
    })

    BLUE = "#0072B2"
    ORANGE = "#E69F00"
    GRAY = "#BBBBBB"

    labels = {
        "paper": r"LR $(y,x)$, 100\%",
        "T1": r"RL $(x,y)$, 0\%",
    }
    colors = {"paper": BLUE, "T1": ORANGE}
    markers = {"paper": "o", "T1": "s"}

    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)

    for tkey in ["paper", "T1"]:
        d = results[tkey]
        ax.errorbar(d["checkpoints"], d["means"], yerr=d["sems"],
                    color=colors[tkey], marker=markers[tkey],
                    markersize=4, linewidth=1.2, capsize=2,
                    label=labels[tkey])

    ax.axhline(y=1.0, color=GRAY, linewidth=0.8, linestyle="--",
               label="$x/y = 1$ (balanced)")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"Leaf gradient $\nabla_x / \nabla_y$")
    ax.set_xlim(0, 2100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22),
              ncol=3, framealpha=0.9, columnspacing=1.0)

    path = os.path.join(outdir, "fig_gradient.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved gradient plot to {path}")


# =====================================================================
# Verify against pre-computed results
# =====================================================================
def verify_results():
    """Load results.json and print comparison table."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    with open(path) as f:
        data = json.load(f)

    print(f"\n{'Arch':<8} {'Target':<12} {'Operator':<5} "
          f"{'Published':>10} {'Topology':<10}")
    print("-" * 55)
    for entry in data["heatmap"]:
        print(f"{entry['arch']:<8} {entry['target']:<12} {entry['operator']:<5} "
              f"{entry['rate']:>9.1f}% {entry['topology']:<10}")


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from arXiv:2604.23256")
    parser.add_argument("--mode", choices=["cell", "heatmap", "gradient", "verify"],
                        default="cell",
                        help="Execution mode (default: cell)")
    parser.add_argument("--arch", choices=["eq6", "v16", "hybrid"],
                        default="v16",
                        help="Architecture (default: v16)")
    parser.add_argument("--target", type=str, default="paper",
                        help=f"Target name (default: paper). "
                        f"Options: {', '.join(TARGETS.keys())}")
    parser.add_argument("--depth", type=int, default=3,
                        help="Tree depth (default: 3)")
    parser.add_argument("--seeds", type=int, default=64,
                        help="Seeds per strategy (default: 64)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-iteration training progress")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory for figures (default: .)")

    args = parser.parse_args()

    if args.mode == "verify":
        verify_results()
    elif args.mode == "gradient":
        print("Measuring gradient trajectories...")
        results = measure_gradient_trajectory(n_seeds=args.seeds)
        try:
            plot_gradient(results, outdir=args.outdir)
        except Exception as e:
            print(f"Plotting skipped ({e}). Data printed below:")
            for tkey, d in results.items():
                print(f"  {tkey}: {list(zip(d['checkpoints'], d['means']))}")
    elif args.mode == "heatmap":
        run_heatmap(n_seeds=args.seeds)
    else:
        if args.target not in TARGETS:
            print(f"ERROR: unknown target '{args.target}'.")
            print(f"Options: {', '.join(TARGETS.keys())}")
            sys.exit(1)
        run_cell(args.arch, args.target, depth=args.depth,
                 n_seeds=args.seeds, verbose=args.verbose)


if __name__ == "__main__":
    main()
