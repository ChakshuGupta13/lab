# Lean formalization — TxGraffiti Conjecture 1 over Mathlib `SimpleGraph`

A single Lake project that mechanizes the main result of *"An annihilation-number
Caro–Wei bound"* (Gupta, [arXiv:2606.29553](https://arxiv.org/abs/2606.29553))
directly over Mathlib's `SimpleGraph`, together with the paper's Corollaries 1–3.

Toolchain: `leanprover/lean4:v4.32.0-rc1`; Mathlib pinned by commit in
`lake-manifest.json` (a CI'd `master` snapshot with a published olean cache).

```bash
lake exe cache get      # fetch the pinned Mathlib olean cache
lake build              # kernel-check every module
```

## What is proved (over real `SimpleGraph`, not hypotheses)

An earlier revision of this formalization stated the graph theory as *hypotheses*
over `ℚ`. This version **proves** it, on Mathlib's real invariants
(`annihilationNumber`, `residue`, `indepNum`, `maxDegree`, `degree`):

| Module | Result |
|---|---|
| `Invariants.lean` | `annihilationNumber`, `residue` (Havel–Hakimi via **well-founded** recursion), `degreeMultiset` |
| `Vehicle.lean` | **Theorem 1** — the vehicle `a ≤ (Δ+1)/2 · W` over the real `annihilationNumber` |
| `CaroWei.lean` | **Caro–Wei** `W ≤ α` (Wei's deletion induction, subtype-free) |
| `Delta2.lean` | **Δ ≤ 2 branch** `a ≤ α` for connected graphs (bipartite *and* odd-cycle cases) |
| `Favaron.lean` | **Favaron** `R ≤ α` (strong induction on `|V|` + edgeless base + α-bridge) |
| `Conjecture.lean` | **Theorem 2** — `txgraffiti_conjecture_1 : a + R ≤ Δ·α` for connected `Δ ≥ 2` |
| `Corollaries.lean` | **Corollaries 1–3** — sharpness, domination, bracketing (algebraic layer, over `ℚ`) |

## The one cited axiom

Everything above is proved **except** one classical input, isolated as a single
named `axiom` in `Favaron.lean`:

```
axiom residue_le_residue_induce_compl_of_maxDegree :
  (∀ w, G.degree w ≤ G.degree v) → 0 < G.degree v →
  G.residue ≤ (G.induce {v}ᶜ).residue
```

This residue-monotonicity step is the graphical-majorization content underlying
the classical bound `R ≤ α` first established by **Favaron, Mahéo & Saclé (1991)**;
the paper *uses* that classical result rather than re-proving it, so it is assumed
here as a named, cited axiom — **not** a `sorry`. The dependence is therefore
explicit and machine-auditable:

```
#print axioms SimpleGraph.txgraffiti_conjecture_1
-- [propext, Classical.choice, Quot.sound, residue_le_residue_induce_compl_of_maxDegree]
```

The axiom's *truth* is verified computationally in `axiom-verification/`, against
the exact `residueAux` definition used here:

- `favaron_verify.py` — `R(G) ≤ R(G−v)` for a max-degree `v` (and `R ≤ α`): **0 failures / 40k random graphs** (n ≤ 8), with the edgeless case correctly excluded by `0 < deg v`.
- `favaron_majorize.py` — exhaustive over **all** graphical sequences `n ≤ 7`: *residue is Schur-convex on graphical sequences*, the precise lemma the axiom instantiates.
- `favaron_seq_lemma.py`, `favaron_refined.py`, `favaron_ptwise.py` — rule out simpler majorization-free routes, documenting why a native Lean proof needs a graphical-majorization + degree-sequence library not yet in Mathlib.

Replacing the axiom with a native proof is a self-contained (multi-week) project;
the rest of the development is unconditional and `sorry`-free.

## Corollaries (algebraic layer)

`Corollaries.lean` keeps the paper's Corollaries 1–3 as self-contained algebraic
lemmas over `ℚ` (their graph-theoretic inputs are exactly the bounds proved
concretely above): `sharpness_iff`, `K4_attains_sharpness`, `dominated_by_max`,
`bracketing`, `K_DeltaPlus1_attains_bracket`. These depend only on
`[propext, Classical.choice, Quot.sound]`.
