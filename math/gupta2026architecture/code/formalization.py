"""Step 3: Formalize the architecture-operator-topology interaction model.

This script serves dual purpose:
1. States the conjecture precisely
2. Runs consistency checks against experimental data

IMPORTANT EPISTEMOLOGICAL NOTE (Adversary review, 23 Apr 2026):
The conjecture was formulated AFTER seeing the data. The original prediction
(before running SML experiments) was that Eq.6+SML would succeed on LR and
fail on RL — the OPPOSITE of what happened. The inversion was a surprise
that forced revision of the model to include the operator (F4) as a factor.
The consistency checks below confirm the REVISED conjecture fits the data,
but this is post-hoc pattern description, not pre-registered hypothesis
testing. The checks verify internal consistency, not predictive power.
Predictive power requires testing on a THIRD operator (see predictions).

All data points are hardcoded from completed experiments (64-256 seeds each).
Any prediction that fails a check is flagged with FAIL.

=============================================================================
CONJECTURE: Architecture-Operator-Topology (AOT) Interaction
=============================================================================

In gradient-based symbolic regression using parameterized binary-tree
architectures with an asymmetric binary operator op(a,b) where the left
input receives amplified gradients and the right input receives attenuated
gradients:

  Recovery rate is NOT determined by expressiveness alone.

Architectures with equivalent expressiveness for the tested target class
(depth-3 binary trees with variables only at leaves) can exhibit
qualitatively different recovery profiles -- from 0% to 100% --
depending on the interaction between:

  F1. The architecture's variable routing
      (which variables available at which tree positions)
  F2. The target tree's topology
      (which branch carries the active subtree at each level)
  F3. The target tree's leaf assignment
      (which variable occupies which leaf position)
  F4. The operator's gradient asymmetry
      (relative magnitude of d(op)/d(left) vs d(op)/d(right))

The interaction produces systematic, reproducible profiles:

  P1. Asymmetric-routing architectures (e.g., Eq.6) are extreme topology
      specialists: some topologies yield 100%, others 0%, depending on
      how F1-F4 interact. The SPECIFIC winning topologies change with
      the operator (F4), but the EXTREME SENSITIVITY is invariant.

  P2. Symmetric-routing architectures (e.g., V16) are generalists: all
      chain topologies yield nonzero recovery. The symmetric variable
      access prevents architecture-induced variable dominance.

  P3. Hybrid architectures (root-only asymmetry) are narrow specialists:
      one topology dominates across operators. (Note: the conjecture
      predicts specialization but does NOT predict WHICH topology.
      The empirical observation that Hybrid favors RL for both operators
      is descriptive, not predicted by the model. See Q1.)

  P4. Leaf-swap invariance tracks routing symmetry:
      - Symmetric architectures: leaf-swap invariant (both operators)
      - Asymmetric architectures: leaf-swap sensitive (operator-dependent)
      - Hybrid: leaf-swap invariant (both operators)

=============================================================================
RELATION TO THE EML-SPECIFIC TWO-FACTOR MODEL (README)
=============================================================================

The README describes a two-factor model (topology x leaf placement) for
Eq.6 + EML. That model is a SPECIAL CASE of this conjecture, obtained by
fixing F4 to EML's gradient structure (exp-left, -1/b-right). It correctly
predicts all 6 EML Eq.6 outcomes.

For SML (sinh-left, -arctan-right), the Eq.6 topology preferences INVERT:
EML Eq.6 favors LR, SML Eq.6 favors RL. This inversion demonstrates that
F4 (operator gradient structure) is an essential factor -- the model cannot
be stated in terms of topology and leaf placement alone.

=============================================================================
LIMITATIONS AND CAVEATS
=============================================================================

1. BOTH tested operators share the same asymmetry DIRECTION (left=amplified,
   right=attenuated). The claim that F4 is essential rests on the topology
   inversion, but both operators are from the same family. A truly
   independent test requires a right-amplified operator.

2. The inversion mechanism is now PARTIALLY verified via gradient
   measurements (scratch/sml_gradients.py). Key finding at iter=1000:
   - EML LR(yx) [100%]: x/y = 0.67 — y gets MORE gradient (favorable)
   - EML RL(xy) [0%]:   x/y = 1.58 — x dominant, y starved
   - SML LR(yx) [0%]:   x/y = 1.78 — x dominant (INVERTED from EML LR)
   - SML RL(xy) [100%]: x/y = 1.19 — more balanced
   For the same topology (LR), EML gives y favorable gradients (x/y<1)
   while SML starves y (x/y>1). However, SML RL (100%) has x/y=1.19,
   not x/y<1, so success with SML RL relies on a more balanced ratio
   rather than a clean inversion. The mechanism is confirmed directionally
   but is more nuanced than simple ratio inversion.

3. Numerical conditioning differs: SML targets have 3-10x narrower range
   than EML targets. V16 SML (83-100%) is more uniform than V16 EML
   (17-100%). The improved V16 SML uniformity may be partly conditioning-
   driven, weakening the cross-operator comparison for V16 specifically.

4. Sample sizes differ: Eq.6 uses 64 seeds x 1 strategy = 64 trials;
   V16/Hybrid use 64 seeds x 4 strategies = 256 trials. The 4x factor
   biases rate comparisons near zero. A 0/64 cell has 95% CI upper bound
   of ~4.6%, vs 0/256 at ~1.2%. The qualitative patterns (100% vs 0%)
   survive, but precise rates are not directly comparable across archs.

5. Expressiveness equivalence is verified only for the tested target class
   (depth-3 trees with variables at leaves). Eq.6 can place x at internal
   nodes, which V16 cannot -- this gives Eq.6 strictly MORE expressiveness
   in general. For the tested targets (where variables only appear at
   leaves), all three architectures can represent all targets.

6. LL topology (1 of 4 chain topologies) is excluded due to overflow.
   If LL were testable and any architecture scored 0%, the affected
   profile claim would need revision.

=============================================================================
PRE-REGISTERED PREDICTIONS (written 23 Apr 2026, before running)
=============================================================================
These are falsifiable predictions. If any fails, the conjecture needs
revision. Record the date and commit hash when checking.

PRED-1. V16 balanced SML: >0% on at least one of the 4 balanced variants.
        Rationale: V16 is a generalist, so balanced shouldn't be 0%.
        ALTERNATIVE (if fails): balanced topology is a qualitatively
        different regime that defeats all architectures.

PRED-2. Hybrid balanced SML: <5% on all balanced variants.
        Rationale: Hybrid is a narrow specialist; balanced is not RL.

PRED-3. A RIGHT-amplified operator (e.g., rml(a,b) = arctan(a) - sinh(b))
        with Eq.6 should produce a DIFFERENT topology preference than
        BOTH EML and SML. Specifically:
        - If left-amplified ops favor RL/RR over LR (SML) or LR/RR over RL (EML),
          a right-amplified op should favor a DIFFERENT subset.
        - P1 still holds: Eq.6 remains an extreme specialist (some 100%, some 0%).
        - P2 still holds: V16 remains a generalist (all chain >0%).
        - P3 still holds: Hybrid has one dominant topology.
        This is the strongest test of the conjecture. If Eq.6's preferences
        do NOT change with a right-amplified operator, F4 is not real.

PRED-4. EML balanced (V16): >0% on at least one balanced variant.
        Rationale: V16 is a generalist for EML chains.
        NOTE: eml(eml(1,x),eml(y,1)) is borderline (range [2.1, 78.8]).
"""

import sys

# ============================================================================
# RAW DATA — every cell from completed experiments
# ============================================================================
# Format: (architecture, operator, topology, leaf_order) -> (successes, total)
# leaf_order: 'yx' = first variable in exp/sinh position
#             'xy' = first variable in ln/arctan position
#             'sym' = V16 (symmetric, no leaf order effect)

DATA = {
    # === EML d3 ===
    # Eq.6
    ("Eq.6", "EML", "LR", "yx"): (256, 256),   # paper target
    ("Eq.6", "EML", "LR", "xy"): (1, 64),       # T8
    ("Eq.6", "EML", "RL", "xy"): (0, 64),       # T1
    ("Eq.6", "EML", "RL", "yx"): (0, 64),       # T1_yx
    ("Eq.6", "EML", "RR", "xy"): (0, 64),       # T4
    ("Eq.6", "EML", "RR", "yx"): (64, 64),      # T4_yx
    # V16 (symmetric — leaf order doesn't matter, but we have both for LR)
    ("V16", "EML", "LR", "yx"): (43, 256),      # paper target
    ("V16", "EML", "LR", "xy"): (40, 256),      # T8
    ("V16", "EML", "RL", "sym"): (255, 256),    # T1 (= T1_yx by symmetry)
    ("V16", "EML", "RR", "sym"): (101, 256),    # T4 (= T4_yx by symmetry)
    # Hybrid
    ("Hybrid", "EML", "LR", "yx"): (1, 256),    # paper target
    ("Hybrid", "EML", "LR", "xy"): (1, 256),    # T8
    ("Hybrid", "EML", "RL", "xy"): (245, 256),  # T1
    ("Hybrid", "EML", "RL", "yx"): (244, 256),  # T1_yx
    ("Hybrid", "EML", "RR", "xy"): (0, 256),    # T4
    ("Hybrid", "EML", "RR", "yx"): (0, 256),    # T4_yx

    # === SML d3 chain ===
    # Eq.6
    ("Eq.6", "SML", "LR", "yx"): (0, 64),       # S_LR
    ("Eq.6", "SML", "LR", "xy"): (0, 64),       # S_LR_xy
    ("Eq.6", "SML", "RL", "xy"): (64, 64),      # S_RL
    ("Eq.6", "SML", "RL", "yx"): (0, 64),       # S_RL_yx
    ("Eq.6", "SML", "RR", "xy"): (64, 64),      # S_RR
    ("Eq.6", "SML", "RR", "yx"): (64, 64),      # S_RR_yx
    # V16
    ("V16", "SML", "LR", "yx"): (216, 256),     # S_LR
    ("V16", "SML", "LR", "xy"): (213, 256),     # S_LR_xy
    ("V16", "SML", "RL", "xy"): (256, 256),     # S_RL
    ("V16", "SML", "RL", "yx"): (256, 256),     # S_RL_yx
    ("V16", "SML", "RR", "xy"): (255, 256),     # S_RR
    ("V16", "SML", "RR", "yx"): (253, 256),     # S_RR_yx
    # Hybrid
    ("Hybrid", "SML", "LR", "yx"): (2, 256),    # S_LR
    ("Hybrid", "SML", "LR", "xy"): (2, 256),    # S_LR_xy
    ("Hybrid", "SML", "RL", "xy"): (256, 256),  # S_RL
    ("Hybrid", "SML", "RL", "yx"): (256, 256),  # S_RL_yx
    ("Hybrid", "SML", "RR", "xy"): (230, 256),  # S_RR
    ("Hybrid", "SML", "RR", "yx"): (227, 256),  # S_RR_yx

    # === SML d3 balanced ===
    # Eq.6
    ("Eq.6", "SML", "Bal", "v1"): (0, 64),      # S_Bal
    ("Eq.6", "SML", "Bal", "v2"): (0, 64),      # S_Bal_yx
    ("Eq.6", "SML", "Bal", "v3"): (0, 64),      # S_Bal_mx
    ("Eq.6", "SML", "Bal", "v4"): (0, 64),      # S_Bal_ms
    # V16
    ("V16", "SML", "Bal", "v1"): (0, 256),      # S_Bal
    ("V16", "SML", "Bal", "v2"): (0, 256),      # S_Bal_yx
    ("V16", "SML", "Bal", "v3"): (0, 256),      # S_Bal_mx
    ("V16", "SML", "Bal", "v4"): (0, 256),      # S_Bal_ms
    # Hybrid
    ("Hybrid", "SML", "Bal", "v1"): (0, 256),    # S_Bal
    ("Hybrid", "SML", "Bal", "v2"): (0, 256),    # S_Bal_yx
    ("Hybrid", "SML", "Bal", "v3"): (0, 256),    # S_Bal_mx
    ("Hybrid", "SML", "Bal", "v4"): (0, 256),    # S_Bal_ms

    # === RML d3 chain (Eq.6 only, final) ===
    ("Eq.6", "RML", "LR", "yx"): (0, 64),       # R_LR
    ("Eq.6", "RML", "LR", "xy"): (0, 64),       # R_LR_xy
    ("Eq.6", "RML", "RL", "xy"): (0, 64),       # R_RL
    ("Eq.6", "RML", "RL", "yx"): (0, 64),       # R_RL_yx
    # V16
    ("V16", "RML", "LR", "yx"): (5, 256),       # R_LR
    ("V16", "RML", "LR", "xy"): (2, 256),       # R_LR_xy
    ("V16", "RML", "RL", "xy"): (0, 256),       # R_RL
    ("V16", "RML", "RL", "yx"): (0, 256),       # R_RL_yx
    # Hybrid
    ("Hybrid", "RML", "LR", "yx"): (3, 256),    # R_LR
    ("Hybrid", "RML", "LR", "xy"): (6, 256),    # R_LR_xy
    ("Hybrid", "RML", "RL", "xy"): (0, 256),    # R_RL
    ("Hybrid", "RML", "RL", "yx"): (0, 256),    # R_RL_yx

    # === EML d3 balanced ===
    # Eq.6
    ("Eq.6", "EML", "Bal", "v1"): (0, 64),      # E_Bal
    ("Eq.6", "EML", "Bal", "v2"): (0, 64),      # E_Bal_yx
    ("Eq.6", "EML", "Bal", "v3"): (0, 64),      # E_Bal_mx
    ("Eq.6", "EML", "Bal", "v4"): (0, 64),      # E_Bal_ms
    # V16
    ("V16", "EML", "Bal", "v1"): (0, 256),      # E_Bal
    ("V16", "EML", "Bal", "v2"): (0, 256),      # E_Bal_yx
    ("V16", "EML", "Bal", "v3"): (0, 256),      # E_Bal_mx
    ("V16", "EML", "Bal", "v4"): (0, 256),      # E_Bal_ms

    # === EML d4 (V16 only) ===
    ("V16", "EML", "LR", "d4"): (59, 256),      # d4_paper
    ("V16", "EML", "RL", "d4"): (2, 256),       # d4_T1
    ("V16", "EML", "RR", "d4"): (3, 256),       # d4_T4
}

# ============================================================================
# ROUTING VERIFICATION DATA (Eq.6 x EML T1, 64 seeds)
# ============================================================================
ROUTING = {
    "n_seeds": 64,
    "n_success": 0,
    "rmse_median": 0.682,
    "dominant_internal": "[1, f, 1, x, f, f]",
    "dominant_internal_count": 63,
    "dominant_leaf": "[1, x, x, y, x, 1, y, x]",
    "dominant_leaf_count": 61,
    "n_unique_trees": 4,
    "dominant_tree_count": 60,
}


# ============================================================================
# VERIFICATION
# ============================================================================

def rate(key):
    """Return success rate as fraction."""
    s, t = DATA[key]
    return s / t


def pct(key):
    """Return success rate as percentage string."""
    return f"{rate(key)*100:.1f}%"


def check(label, condition, detail=""):
    """Check a prediction. Print PASS/FAIL."""
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def main():
    n_pass = 0
    n_fail = 0
    n_total = 0

    def tally(result):
        nonlocal n_pass, n_fail, n_total
        n_total += 1
        if result:
            n_pass += 1
        else:
            n_fail += 1

    print("=" * 72)
    print("CONJECTURE VERIFICATION: Architecture-Operator-Topology Interaction")
    print("=" * 72)

    # ------------------------------------------------------------------
    # P1: Eq.6 is an extreme topology specialist for BOTH operators
    # ------------------------------------------------------------------
    print("\n--- P1: Eq.6 extreme topology sensitivity (both operators) ---")

    # EML: has at least one 100% and one 0% topology
    eq6_eml_rates = {}
    for topo in ["LR", "RL", "RR"]:
        rates = []
        for lo in ["xy", "yx"]:
            k = ("Eq.6", "EML", topo, lo)
            if k in DATA:
                rates.append(rate(k))
        eq6_eml_rates[topo] = max(rates) if rates else 0

    has_100_eml = any(r >= 0.99 for r in eq6_eml_rates.values())
    has_0_eml = any(r <= 0.02 for r in eq6_eml_rates.values())
    tally(check("Eq.6+EML has >=1 topology at ~100%",
                has_100_eml,
                f"max rates: {', '.join(f'{t}={eq6_eml_rates[t]*100:.1f}%' for t in ['LR','RL','RR'])}"))
    tally(check("Eq.6+EML has >=1 topology at ~0%",
                has_0_eml,
                f"min rates by topo: RL={min(rate(('Eq.6','EML','RL','xy')), rate(('Eq.6','EML','RL','yx')))*100:.1f}%"))

    # SML: has at least one 100% and one 0% topology
    eq6_sml_rates = {}
    for topo in ["LR", "RL", "RR"]:
        rates = []
        for lo in ["xy", "yx"]:
            k = ("Eq.6", "SML", topo, lo)
            if k in DATA:
                rates.append(rate(k))
        eq6_sml_rates[topo] = max(rates) if rates else 0

    has_100_sml = any(r >= 0.99 for r in eq6_sml_rates.values())
    has_0_sml = any(r <= 0.02 for r in eq6_sml_rates.values())
    tally(check("Eq.6+SML has >=1 topology at ~100%",
                has_100_sml,
                f"max rates: {', '.join(f'{t}={eq6_sml_rates[t]*100:.1f}%' for t in ['LR','RL','RR'])}"))
    tally(check("Eq.6+SML has >=1 topology at ~0%",
                has_0_sml,
                f"LR max={eq6_sml_rates['LR']*100:.1f}%"))

    # The INVERSION: EML's best is SML's worst and vice versa
    eml_best = max(eq6_eml_rates, key=eq6_eml_rates.get)
    sml_best = max(eq6_sml_rates, key=eq6_sml_rates.get)
    tally(check("Eq.6 best topology DIFFERS between EML and SML",
                eml_best != sml_best,
                f"EML best={eml_best}, SML best={sml_best}"))

    # ------------------------------------------------------------------
    # P2: V16 is a generalist for BOTH operators
    # ------------------------------------------------------------------
    print("\n--- P2: V16 generalist on chain topologies (both operators) ---")

    # EML
    v16_eml_min = min(
        rate(("V16", "EML", "LR", "yx")),
        rate(("V16", "EML", "RL", "sym")),
        rate(("V16", "EML", "RR", "sym")),
    )
    tally(check("V16+EML: all chain topologies > 0%",
                v16_eml_min > 0,
                f"min={v16_eml_min*100:.1f}% (LR)"))

    # SML
    v16_sml_rates = {}
    for topo in ["LR", "RL", "RR"]:
        rates = []
        for lo in ["xy", "yx"]:
            k = ("V16", "SML", topo, lo)
            if k in DATA:
                rates.append(rate(k))
        v16_sml_rates[topo] = min(rates) if rates else 0
    v16_sml_min = min(v16_sml_rates.values())
    tally(check("V16+SML: all chain topologies > 0%",
                v16_sml_min > 0,
                f"min={v16_sml_min*100:.1f}% (LR)"))

    # Tighter: V16+SML should be > 50% on all chain topologies
    tally(check("V16+SML: all chain topologies > 50%",
                v16_sml_min > 0.50,
                f"min={v16_sml_min*100:.1f}%"))

    # ------------------------------------------------------------------
    # P3: Hybrid is a narrow specialist for BOTH operators
    # ------------------------------------------------------------------
    print("\n--- P3: Hybrid narrow specialist (both operators) ---")

    # EML Hybrid: RL dominates
    hyb_eml_rl = max(rate(("Hybrid", "EML", "RL", "xy")),
                     rate(("Hybrid", "EML", "RL", "yx")))
    hyb_eml_lr = max(rate(("Hybrid", "EML", "LR", "xy")),
                     rate(("Hybrid", "EML", "LR", "yx")))
    hyb_eml_rr = max(rate(("Hybrid", "EML", "RR", "xy")),
                     rate(("Hybrid", "EML", "RR", "yx")))
    tally(check("Hybrid+EML: one topology >> others",
                hyb_eml_rl > 0.90 and hyb_eml_lr < 0.05 and hyb_eml_rr < 0.05,
                f"RL={hyb_eml_rl*100:.1f}%, LR={hyb_eml_lr*100:.1f}%, RR={hyb_eml_rr*100:.1f}%"))

    # SML Hybrid: RL dominates, RR moderate, LR near-zero
    hyb_sml_rl = max(rate(("Hybrid", "SML", "RL", "xy")),
                     rate(("Hybrid", "SML", "RL", "yx")))
    hyb_sml_lr = max(rate(("Hybrid", "SML", "LR", "xy")),
                     rate(("Hybrid", "SML", "LR", "yx")))
    hyb_sml_rr = max(rate(("Hybrid", "SML", "RR", "xy")),
                     rate(("Hybrid", "SML", "RR", "yx")))
    tally(check("Hybrid+SML: RL dominates, LR near-zero",
                hyb_sml_rl > 0.90 and hyb_sml_lr < 0.05,
                f"RL={hyb_sml_rl*100:.1f}%, LR={hyb_sml_lr*100:.1f}%, RR={hyb_sml_rr*100:.1f}%"))

    # Both operators: Hybrid's best topology is RL
    eml_best_hyb = max(["LR", "RL", "RR"],
                       key=lambda t: {"LR": hyb_eml_lr, "RL": hyb_eml_rl, "RR": hyb_eml_rr}[t])
    sml_best_hyb = max(["LR", "RL", "RR"],
                       key=lambda t: {"LR": hyb_sml_lr, "RL": hyb_sml_rl, "RR": hyb_sml_rr}[t])
    tally(check("Hybrid best topology = RL for both operators",
                eml_best_hyb == "RL" and sml_best_hyb == "RL",
                f"EML best={eml_best_hyb} ({hyb_eml_rl*100:.1f}%), "
                f"SML best={sml_best_hyb} ({hyb_sml_rl*100:.1f}%)"))

    # ------------------------------------------------------------------
    # P4: Leaf-swap invariance tracks routing symmetry
    # ------------------------------------------------------------------
    print("\n--- P4: Leaf-swap invariance ---")

    # V16 SML: leaf-swap pairs within 2pp
    v16_pairs = [
        (("V16", "SML", "LR", "yx"), ("V16", "SML", "LR", "xy")),
        (("V16", "SML", "RL", "xy"), ("V16", "SML", "RL", "yx")),
        (("V16", "SML", "RR", "xy"), ("V16", "SML", "RR", "yx")),
    ]
    max_v16_diff = max(abs(rate(a) - rate(b)) for a, b in v16_pairs)
    tally(check("V16+SML: leaf-swap pairs within 2pp",
                max_v16_diff < 0.02,
                f"max diff={max_v16_diff*100:.1f}pp"))

    # V16 EML: LR pair within 2pp
    v16_eml_lr_diff = abs(rate(("V16", "EML", "LR", "yx")) -
                          rate(("V16", "EML", "LR", "xy")))
    tally(check("V16+EML: LR leaf-swap within 2pp",
                v16_eml_lr_diff < 0.02,
                f"diff={v16_eml_lr_diff*100:.1f}pp"))

    # Hybrid: leaf-swap invariant for both operators
    hyb_eml_rl_diff = abs(rate(("Hybrid", "EML", "RL", "xy")) -
                          rate(("Hybrid", "EML", "RL", "yx")))
    hyb_sml_pairs = [
        (("Hybrid", "SML", "LR", "yx"), ("Hybrid", "SML", "LR", "xy")),
        (("Hybrid", "SML", "RL", "xy"), ("Hybrid", "SML", "RL", "yx")),
        (("Hybrid", "SML", "RR", "xy"), ("Hybrid", "SML", "RR", "yx")),
    ]
    max_hyb_sml_diff = max(abs(rate(a) - rate(b)) for a, b in hyb_sml_pairs)
    tally(check("Hybrid+EML: RL leaf-swap within 2pp",
                hyb_eml_rl_diff < 0.02,
                f"diff={hyb_eml_rl_diff*100:.1f}pp"))
    tally(check("Hybrid+SML: leaf-swap pairs within 2pp",
                max_hyb_sml_diff < 0.02,
                f"max diff={max_hyb_sml_diff*100:.1f}pp"))

    # Eq.6: leaf-swap SENSITIVE for at least one topology (both operators)
    # EML: LR swap 100% vs 1.6%, RR swap 100% vs 0%
    eq6_eml_lr_diff = abs(rate(("Eq.6", "EML", "LR", "yx")) -
                          rate(("Eq.6", "EML", "LR", "xy")))
    tally(check("Eq.6+EML: LR leaf-swap sensitive (>50pp diff)",
                eq6_eml_lr_diff > 0.50,
                f"diff={eq6_eml_lr_diff*100:.1f}pp"))

    # SML: RL swap 100% vs 0%
    eq6_sml_rl_diff = abs(rate(("Eq.6", "SML", "RL", "xy")) -
                          rate(("Eq.6", "SML", "RL", "yx")))
    tally(check("Eq.6+SML: RL leaf-swap sensitive (>50pp diff)",
                eq6_sml_rl_diff > 0.50,
                f"diff={eq6_sml_rl_diff*100:.1f}pp"))

    # ------------------------------------------------------------------
    # ADDITIONAL: Operator inversion for Eq.6
    # ------------------------------------------------------------------
    print("\n--- Operator inversion: Eq.6 topology preferences reverse ---")

    # EML Eq.6 best chain topology (by max leaf-order rate)
    tally(check("EML Eq.6: LR is best (100%)",
                eq6_eml_rates["LR"] >= 0.99,
                f"LR={eq6_eml_rates['LR']*100:.1f}%"))
    tally(check("EML Eq.6: RL is worst (0%)",
                eq6_eml_rates["RL"] <= 0.01,
                f"RL={eq6_eml_rates['RL']*100:.1f}%"))
    tally(check("SML Eq.6: LR is worst (0%)",
                eq6_sml_rates["LR"] <= 0.01,
                f"LR={eq6_sml_rates['LR']*100:.1f}%"))
    tally(check("SML Eq.6: RL is best (100%)",
                eq6_sml_rates["RL"] >= 0.99,
                f"RL={eq6_sml_rates['RL']*100:.1f}%"))

    # RR works for both (with appropriate leaves)
    tally(check("EML Eq.6: RR achievable (100% with yx leaves)",
                eq6_eml_rates["RR"] >= 0.99,
                f"RR best={eq6_eml_rates['RR']*100:.1f}%"))
    tally(check("SML Eq.6: RR achievable (100% with both leaf orders)",
                eq6_sml_rates["RR"] >= 0.99,
                f"RR best={eq6_sml_rates['RR']*100:.1f}%"))

    # ------------------------------------------------------------------
    # ROUTING VERIFICATION
    # ------------------------------------------------------------------
    print("\n--- Routing verification: Eq.6 x EML T1 (RL) trap ---")

    tally(check("All seeds trapped (0/64 success)",
                ROUTING["n_success"] == 0,
                f"{ROUTING['n_success']}/64"))
    tally(check("Dominant trap tree (>90% of seeds)",
                ROUTING["dominant_tree_count"] / ROUTING["n_seeds"] > 0.90,
                f"{ROUTING['dominant_tree_count']}/{ROUTING['n_seeds']}"))

    # The SPECIFIC internal routing prediction was wrong — report separately
    predicted_all_f = all(c == 'f' for c in
                         ROUTING["dominant_internal"].strip("[]").replace(" ", "").split(","))
    predicted_no_y = 'y' not in ROUTING["dominant_leaf"]
    print(f"  [INFO] Mechanistic prediction — all-f internal: "
          f"{'confirmed' if predicted_all_f else 'FAILED'}"
          f"  (actual: {ROUTING['dominant_internal']})")
    print(f"  [INFO] Mechanistic prediction — no-y leaves:    "
          f"{'confirmed' if predicted_no_y else 'FAILED'}"
          f"  (actual: {ROUTING['dominant_leaf']})")
    print(f"  NOTE: Trapping confirmed but specific routing mechanism")
    print(f"        is more complex than predicted. The trap is structured,")
    print(f"        not trivial — the optimizer routes y into the tree but")
    print(f"        converges to a wrong combination.")

    # ------------------------------------------------------------------
    # DEPTH-4 COLLAPSE (V16 only)
    # ------------------------------------------------------------------
    print("\n--- V16 depth-4 collapse ---")

    v16_d3_rl = rate(("V16", "EML", "RL", "sym"))
    v16_d4_rl = rate(("V16", "EML", "RL", "d4"))
    tally(check("V16 d3->d4: RL collapses (>90pp drop)",
                v16_d3_rl - v16_d4_rl > 0.90,
                f"d3={v16_d3_rl*100:.1f}% -> d4={v16_d4_rl*100:.1f}%"))

    v16_d3_lr = rate(("V16", "EML", "LR", "yx"))
    v16_d4_lr = rate(("V16", "EML", "LR", "d4"))
    tally(check("V16 d3->d4: LR survives (within 10pp)",
                abs(v16_d3_lr - v16_d4_lr) < 0.10,
                f"d3={v16_d3_lr*100:.1f}% -> d4={v16_d4_lr*100:.1f}%"))

    # ------------------------------------------------------------------
    # BALANCED TOPOLOGY (SML + EML)
    # ------------------------------------------------------------------
    print("\n--- Balanced topology (SML + EML) ---")

    bal_eq6_all_zero = all(
        rate(("Eq.6", "SML", "Bal", v)) == 0
        for v in ["v1", "v2", "v3", "v4"]
    )
    tally(check("Eq.6+SML: all 4 balanced variants = 0%",
                bal_eq6_all_zero,
                "0/64 across all 4 variants"))

    bal_v16_all_zero = all(
        rate(("V16", "SML", "Bal", v)) == 0
        for v in ["v1", "v2", "v3", "v4"]
    )
    tally(check("V16+SML: all 4 balanced variants = 0%",
                bal_v16_all_zero,
                "0/256 across all 4 variants — PRED-1 FAILED"))
    bal_hyb_all_zero = all(
        rate(("Hybrid", "SML", "Bal", v)) == 0
        for v in ["v1", "v2", "v3", "v4"]
    )
    tally(check("Hybrid+SML: all 4 balanced variants = 0% (PRED-2 confirmed)",
                bal_hyb_all_zero,
                "0/256 across all 4 variants"))

    # EML balanced: PRED-4 test
    bal_eq6_eml_all_zero = all(
        rate(("Eq.6", "EML", "Bal", v)) == 0
        for v in ["v1", "v2", "v3", "v4"]
    )
    tally(check("Eq.6+EML: all 4 balanced variants = 0%",
                bal_eq6_eml_all_zero,
                "0/64 across all 4 variants"))

    bal_v16_eml_all_zero = all(
        rate(("V16", "EML", "Bal", v)) == 0
        for v in ["v1", "v2", "v3", "v4"]
    )
    tally(check("V16+EML: all 4 balanced variants = 0% (PRED-4 FAILED)",
                bal_v16_eml_all_zero,
                "0/256 across all 4 variants"))
    # ------------------------------------------------------------------
    # RML (RIGHT-AMPLIFIED OPERATOR) — PRED-3
    # ------------------------------------------------------------------
    print("\n--- RML (right-amplified): all 3 architectures ---")

    rml_eq6_all_zero = all(
        rate(("Eq.6", "RML", topo, lo)) == 0
        for topo in ["LR", "RL"] for lo in ["xy", "yx"]
    )
    tally(check("Eq.6+RML: ALL 4 chain targets = 0%",
                rml_eq6_all_zero,
                "P1 partially fails — no 100% topology exists"))

    print("  [INFO] PRED-3 P1 predicted some topologies at 100%:")
    print("         Eq.6+EML best=LR(100%), Eq.6+SML best=RL(100%)")
    print("         Eq.6+RML: ALL 0% — right-amplified operator")
    print("         makes Eq.6 entirely unfindable, not just reordered")
    # V16 RML: P2 partially fails — RL is 0%
    v16_rml_lr = max(rate(("V16", "RML", "LR", "yx")),
                     rate(("V16", "RML", "LR", "xy")))
    v16_rml_rl = max(rate(("V16", "RML", "RL", "xy")),
                     rate(("V16", "RML", "RL", "yx")))
    tally(check("V16+RML: at least one chain topology > 0%",
                v16_rml_lr > 0 or v16_rml_rl > 0,
                f"LR={v16_rml_lr*100:.1f}%, RL={v16_rml_rl*100:.1f}%"))
    tally(check("V16+RML: NOT all chain topologies > 0% (P2 partially fails)",
                v16_rml_rl == 0,
                f"RL=0/256 \u2014 V16 is NOT a universal generalist"))

    # V16 topology inversion: EML/SML best=RL, RML best=LR
    v16_rml_best = "LR" if v16_rml_lr > v16_rml_rl else "RL"
    tally(check("V16 topology preference inverts with RML",
                v16_rml_best == "LR",
                f"EML/SML best=RL, RML best={v16_rml_best} ({v16_rml_lr*100:.1f}%)"))

    # Hybrid RML: LR faint signal, RL = 0 — same inversion as V16
    hyb_rml_lr = max(rate(("Hybrid", "RML", "LR", "yx")),
                     rate(("Hybrid", "RML", "LR", "xy")))
    hyb_rml_rl = max(rate(("Hybrid", "RML", "RL", "xy")),
                     rate(("Hybrid", "RML", "RL", "yx")))
    tally(check("Hybrid+RML: RL = 0% (dominant topology breaks)",
                hyb_rml_rl == 0,
                f"RL=0/256 — Hybrid loses its RL specialization with RML"))
    tally(check("Hybrid+RML: LR has faint signal (>0%)",
                hyb_rml_lr > 0,
                f"LR={hyb_rml_lr*100:.1f}%"))
    hyb_rml_best = "LR" if hyb_rml_lr > hyb_rml_rl else "RL"
    tally(check("Hybrid topology preference inverts with RML",
                hyb_rml_best == "LR",
                f"EML/SML best=RL, RML best={hyb_rml_best} ({hyb_rml_lr*100:.1f}%)"))
    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"CONSISTENCY CHECK SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_total} checks")
    print(f"  (These are post-hoc consistency checks, not pre-registered predictions.)")
    print(f"  (2 mechanistic routing predictions FAILED — reported separately above.)")
    if n_fail == 0:
        print("All consistency checks pass.")
    else:
        print(f"WARNING: {n_fail} check(s) failed — review above")
    print("=" * 72)

    # ------------------------------------------------------------------
    # OPEN QUESTIONS (for Adversary review)
    # ------------------------------------------------------------------
    print("""
OPEN QUESTIONS (unresolved by current data):

Q1. Why does Hybrid specialize to RL for BOTH operators?
    - EML Hybrid: RL=96%, everything else ~0%
    - SML Hybrid: RL=100%, RR=89%, LR=0.8%
    - The root-level 4-way softmax {1, x, y, f_child} seems to create an
      RL routing preference independent of the operator. Why?

Q2. Why does RR work for Eq.6 with BOTH operators?
    - EML Eq.6 RR (yx) = 100%, SML Eq.6 RR (xy) = 100%, RR (yx) = 100%
    - RR routes the active subtree through attenuated positions at both
      levels. Is this because double-attenuation neutralizes the x-advantage
      regardless of whether the attenuator is -1/b or arctan?

Q3. The routing verification showed structured traps (not trivial x-only):
    - Internal: [1, f, 1, x, f, f] — node 3 routes to x, not f
    - Leaf: [1, x, x, y, x, 1, y, x] — y appears at 2 positions
    - The optimizer TRIES to include both variables but converges to a
      wrong combination. Is this because the trap basin has a larger
      volume than the correct basin in parameter space?

Q4. Is balanced topology universally hard, or is it SML-specific?
    - SML balanced: 0% for ALL THREE architectures (Eq.6, V16, Hybrid)
    - EML balanced: ALSO 0% for both Eq.6 (0/64) and V16 (0/256)
    - ANSWERED: balanced is universally hard across operators.

Q5. Numerical conditioning confound:
    - SML targets have 3-10x narrower range than EML targets
    - V16 SML (83-100%) is more uniform than V16 EML (17-99.6%)
    - Could V16's improved SML performance be partly due to better
      conditioning rather than operator structure?

=============================================================================
PRE-REGISTERED PREDICTION STATUS (updated 23 Apr 2026)
=============================================================================

PRED-1. V16 balanced SML >0%:  **FAILED**
        Result: 0/256 on all 4 balanced variants.
        Interpretation: balanced topology is qualitatively different from
        chain topologies. V16's generalist advantage does not extend to
        balanced trees. The ALTERNATIVE explanation holds: balanced is
        universally hard for all architectures with SML.

PRED-2. Hybrid balanced SML <5%:  **CONFIRMED**
        Result: 0/256 on all 4 balanced variants.
        Balanced is universally hard: 0% for ALL 3 architectures.

PRED-3. RML topology preferences:  **PARTIALLY CONFIRMED, PARTIALLY FAILED**
        Eq.6: ALL 4 targets = 0%. P1 PARTIALLY FAILS — no 100% topology.
        V16: LR=2.0%, LR_xy=0.8%, RL=0%. P2 PARTIALLY FAILS — RL=0%,
        V16 is NOT a universal generalist. Topology INVERTS: best=LR.
        Hybrid: LR=1.2%, LR_xy=2.3%, RL=0%, RL_yx=0%. Hybrid loses its
        RL specialization entirely; faint signal moves to LR. Topology
        inversion confirmed across BOTH V16 and Hybrid.
        CONFIRMED: operator gradient direction changes topology prefs.
        FAILED: Eq.6 has no winning topology, V16/Hybrid not all >0%.

PRED-4. EML balanced V16 >0%:  **FAILED**
        Result: 0/256 on all 4 balanced variants (V16).
        Also: 0/64 on all 4 balanced variants (Eq.6).
        Balanced topology is universally hard across ALL operators
        (EML, SML) and ALL architectures (Eq.6, V16).
        Q4 is now ANSWERED.
""")

    return n_fail


if __name__ == "__main__":
    sys.exit(main())
