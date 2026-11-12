# Lean Coverage For The Annihilation-Number Bound

This Lake project accompanies [arXiv:2606.29553](https://arxiv.org/abs/2606.29553).
It checks selected graph-level results and algebraic consequences, not the
whole revised manuscript.

The original environment is pinned to `leanprover/lean4:v4.32.0-rc1` and
Mathlib commit `2e09891a520feeea2194c3f33d57477eb814d7a8` in
`lake-manifest.json`. Do not replace these pins merely to run the original
proof. On a machine with this toolchain and the dependency cache available:

```sh
lake exe cache get
lake build
```

If the pinned cache is unavailable, building dependencies from source can be
substantial work. A cache-hit build is not evidence that a different toolchain
or Mathlib version will compile the project.

## Graph-Level Results

The following statements use the project's actual finite `SimpleGraph`
invariants. The first three depend only on the standard Lean/Mathlib axioms
`propext`, `Classical.choice` and `Quot.sound`.

| Declaration | Scope |
|---|---|
| `SimpleGraph.annih_vehicle` | `a <= (Delta+1)/2 * W` for positive maximum degree |
| `SimpleGraph.caroWei_le_indepNum` | Caro-Wei `W <= alpha` |
| `SimpleGraph.annih_le_predMaxDegree_indepNum` | `a <= (Delta-1) * alpha` for connected graphs with maximum degree at least two |
| `SimpleGraph.residue_le_indepNum` | `R <= alpha`, conditional on the custom axiom below |
| `SimpleGraph.txgraffiti_conjecture_1` | `a + R <= Delta * alpha` for connected maximum degree at least two, conditional on that same axiom |

The annihilation number is defined as the maximum cardinality of a degree
submultiset whose sum is at most the edge count. Mathematically this equals
the ordered-prefix definition in the paper. The residue sorts degrees in
descending order and iterates the zero-retaining Havel-Hakimi reduction.

## Custom Axiom

The active TxGraffiti theorem depends on
`SimpleGraph.residue_le_residue_induce_compl_of_maxDegree`, which assumes
that deleting a positive-maximum-degree vertex does not decrease the residue.
The positivity hypothesis is essential; the statement fails on an edgeless
graph. This input is not proved in the exported project.

An import probe can inspect the dependencies:

```lean
import TxGraffitiC1FC
#print axioms SimpleGraph.annih_vehicle
#print axioms SimpleGraph.txgraffiti_conjecture_1
```

The first lists only the three standard axioms. The second additionally lists
`SimpleGraph.residue_le_residue_induce_compl_of_maxDegree`.
The scripts in `axiom-verification/` are finite computational probes, not a
proof of that axiom. Absence of `sorry` does not remove a custom assumption.

## Algebraic Lemmas And Coverage Gaps

`Corollaries.lean` proves rational-arithmetic consequences from explicitly
assumed invariant inequalities. Its numerical witness lemmas evaluate numbers,
not constructed graphs. These lemmas are not graph-level formalizations of
the revised paper's full results.

In particular, this project does not formalize the classical graph bounds
`W <= R` or `alpha <= a`, the revised five-term interval chain and its connected
sharpness family for every maximum degree, the explicit eight-graph equality
classification, or the stronger domination statement with `R` rather than
`max R W`. The corresponding mathematical proofs are in the paper.
