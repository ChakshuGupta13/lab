# NttFaultRank — Lean 4 formalisation

Machine-verified proof of the rank ladder and kernel equality for
twiddle-zeroing faults on the ML-KEM forward NTT. Companion to the
paper *Rank Ceiling for Twiddle-Zeroing Faults on the ML-KEM Forward
NTT*.

## Headline

`NttFaultRank.theorem2_kyber_final` in
[NttFaultRank/Theorem2Final.lean](NttFaultRank/Theorem2Final.lean)
states and proves

  `rank D_K = n - 2`  and  `ker D_K = span {e_0, e_1}`

for the Kyber-shape parameters: one fault per NTT layer, every
twiddle nonzero, and `2 ≠ 0` in the field. Closes Theorem 2 of the
paper.

## Build

```bash
lake exe cache get
lake build
```

Built against `leanprover/lean4:v4.27.0` and `mathlib v4.27.0`
(pinned in [lakefile.toml](lakefile.toml) and
[lean-toolchain](lean-toolchain)).

## Module map

| File | Role |
|---|---|
| `Basic.lean` | Field, dimension and helper definitions |
| `PairButterfly.lean` | Single-butterfly map and its inverse |
| `GroupLayer.lean`, `GroupLayerInv.lean` | One transform layer over a fixed group |
| `FullLayer.lean`, `LayerFlat.lean`, `LayerOn.lean` | Layer-wide composition |
| `Ntt.lean`, `Invertibility.lean` | The full NTT and its invertibility |
| `Reshape.lean`, `FlatLayer.lean` | Indexing helpers |
| `PerturbationRank.lean` | Theorem 1 (single-fault rank) |
| `Telescope.lean` | Telescope decomposition of `D_K` |
| `Vsubspaces.lean` | `V_ℓ` subspaces (Lemma 1) |
| `IntImageInV.lean` | Bit-subspace localization (Lemma 2) |
| `PiInjective.lean` | Quotient injection (Lemma 3) |
| `Fellsupport.lean` | Identity submatrix (Lemma 4) |
| `TeleTermRank.lean` | Per-term rank `rank T_ℓ = len_ℓ` |
| `RankCeiling.lean` | Rank upper bound `rank D_K ≤ n - 2` |
| `Assembly.lean` | Image disjointness + block-triangular LI |
| `Theorem2Subset.lean` | Kernel inclusion `span {e_0, e_1} ⊆ ker D_K` |
| `Theorem2KerReduction.lean` | Rank–nullity bridge |
| `Theorem2.lean` | Headline statement (documentation) |
| `Theorem2Final.lean` | Closed headline theorem |

## Verifying no `sorry`

```bash
grep -rn '\bsorry\b' NttFaultRank/
```

Returns matches only in comments referring to a historically-resolved
`sorry`; no live `sorry` tactic.

## Axiom check

```bash
echo 'import NttFaultRank
#print axioms NttFaultRank.theorem2_kyber_final' > /tmp/check.lean
lake env lean /tmp/check.lean
```

Expected:

```
'NttFaultRank.theorem2_kyber_final' depends on axioms:
  [propext, Classical.choice, Quot.sound]
```
