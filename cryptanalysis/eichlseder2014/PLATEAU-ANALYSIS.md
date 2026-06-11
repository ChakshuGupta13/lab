# Eichlseder–Mendel–Schläffer 2014 — GnD Search Plateau Analysis

Diagnostic findings on the guess-and-determine search engine of:

> Eichlseder, M., Mendel, F. & Schläffer, M. (2014).
> *Branching Heuristics in Differential Collision Search with Applications to SHA-512.*
> FSE 2014 (IACR final version 2014-04-30).
> [ePrint 2014/302](https://eprint.iacr.org/2014/302)

## Context

The paper's Algorithm 2 (look-ahead branching heuristic) achieves a
38-step SHA-512 semi-free-start collision at 2^{40.5} compression
evaluations. We implemented Algorithm 2 and ran a 60-seed production
campaign. The SAT-based reproduction path succeeds (Table 3 → CaDiCaL in
37.9s / 640K conflicts), but the GnD search **did not converge** in any of
the 60 seeds.

This note documents **why** the GnD stalls: the Phase 2/3 plateau is
structural, not an implementation deficiency.

## Key finding: Stage 3 propagation multiplier = 1.0

The search plateau occurs at ~5280–5700 dash bits remaining (of 7464
initial), corresponding to ~17–25% of free bits resolved. Diagnostic
analysis (Tasks 10–25 in the research repo) establishes:

| Diagnostic | Result | Implication |
|------------|--------|-------------|
| Phase 2/3 plateau depth | 5280–5700 dashes (mean 6196, σ=63, 50 restarts) | Robust across seeds; 10^6 restarts would not approach 0 |
| Best-of-50 plateau | 5985 dashes (19.8% resolved) | Diminishing returns from restarts |
| Phase transition | Sharp at ~65% bits fixed | Below 65%: unsolvable in 100K decisions |
| Propagation multiplier (Σ/σ only) | **1.0** — zero cascade from XOR equations | GF(2) linear system has rank 0 over unknown variables |
| Propagation multiplier (with carry) | **1.0** — carries are nonlinear, contribute nothing to GF(2) | Adding 64-bit carry variables to GF(2) system does not increase rank |
| Twobit domain | 100% diff-domain at plateau | All two-bit conditions are differential, not value-determining |
| Look-ahead effectiveness at plateau | Score distribution collapses: all candidates score ≈ 0 | No variable provides cascade; branching degenerates to random |

## Interpretation

The plateau is the point where all **linear** deduction mechanisms
(bit-sliced propagation, XOR chains through Σ/σ, two-bit consistency) are
exhausted. What remains is **nonlinear** modular-addition structure that the
GnD's propagation engine cannot exploit. The paper's 2^{40.5} figure
represents the cost of the plateau itself: the look-ahead heuristic
narrows the tree above the plateau efficiently, but the plateau-to-solution
gap is brute force.

This is consistent with the paper's own 2^{40.5} measurement (1.5 hours
on a 40-CPU cluster) and explains why simply re-running the search with
more seeds or better random decisions does not help: the bottleneck is
**structural**, not stochastic.

## Reproduction

The diagnostic chain (Tasks 10–25) is in the private research repository.
The SAT-based collision reproduction (which bypasses the GnD plateau
entirely) is in this lab repository at
[`cryptanalysis/eichlseder2014/`](.).
