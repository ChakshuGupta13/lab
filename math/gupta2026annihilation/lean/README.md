# Lean Coverage For The Annihilation-Number Bound

This Lake project accompanies [arXiv:2606.29553](https://arxiv.org/abs/2606.29553).
It checks selected graph-level results and algebraic consequences, not the
whole revised manuscript.

The original environment is pinned to `leanprover/lean4:v4.32.0-rc1` and
Mathlib commit `2e09891a520feeea2194c3f33d57477eb814d7a8` in
`lake-manifest.json`. Do not replace these pins merely to run the original
proof. On a machine with this toolchain and the pinned dependencies available,
run from this directory:

```sh
lake build
lake env lean ProofAudit.lean
```

The library build compiles the graph proofs. The second command checks the
exact types of the TxGraffiti, classification and strictness theorems and
prints the axioms of the results listed below. Each axiom list is
`[propext, Classical.choice, Quot.sound]`.

If dependencies are missing, `lake exe cache get` downloads the pinned
Mathlib cache. Building dependencies from source can be substantial work.
The checked version does not establish compatibility with a different
toolchain or Mathlib version.

## Graph-Level Results

The following statements use the project's actual finite `SimpleGraph`
invariants. They depend only on the standard Lean/Mathlib axioms
`propext`, `Classical.choice` and `Quot.sound`.

| Declaration | Scope |
|---|---|
| `SimpleGraph.annih_vehicle` | `a <= (Delta+1)/2 * W` for positive maximum degree |
| `SimpleGraph.caroWei_le_indepNum` | Caro-Wei `W <= alpha` |
| `SimpleGraph.indepNum_le_annihilationNumber` | `alpha <= a` |
| `SimpleGraph.annih_le_predMaxDegree_indepNum` | `a <= (Delta-1) * alpha` for connected graphs with maximum degree at least two |
| `SimpleGraph.residue_le_residue_induce_compl_of_maxDegree` | Deleting a positive-maximum-degree vertex does not decrease `R` |
| `SimpleGraph.residue_le_indepNum` | `R <= alpha` |
| `SimpleGraph.txgraffiti_conjecture_1` | `a + R <= Delta * alpha` for connected maximum degree at least two |
| `SimpleGraph.txgraffiti_equality_classification` | Equality iff isomorphic to `P3`, `P4`, `P6`, `C3`, `C4`, `C5`, `C7` or `K4`, for connected maximum degree at least two |
| `SimpleGraph.strict_of_maxDegree_ge_four` | Strict TxGraffiti inequality for connected maximum degree at least four |

The annihilation number is defined as the maximum cardinality of a degree
submultiset whose sum is at most the edge count. Mathematically this equals
the ordered-prefix definition in the paper. The residue sorts degrees in
descending order and iterates the zero-retaining Havel-Hakimi reduction.

## Residue And Equality Proofs

`ResidueMonotonicity.lean` proves the deletion step using Havel-Hakimi
commutation identities and degree-preserving graph switches.
`Favaron.lean` uses this theorem to prove `R <= alpha` by induction.
The deletion theorem requires a vertex of positive maximum degree; it does
not assert the corresponding statement for an edgeless graph.

`Classification.lean` proves the full equality characterization by graph
isomorphism. Its dependencies classify connected maximum-degree-two graphs
as paths or cycles, compute their residues, prove the eight graph witnesses,
and reduce equality at maximum degree at least three to `K4`.

The Python scripts in `axiom-verification/` are finite computational probes
of residue-deletion monotonicity, not dependencies of the Lean proof.

## Algebraic Lemmas And Coverage Gaps

`Corollaries.lean` proves rational-arithmetic consequences from explicitly
assumed invariant inequalities. Its numerical witness lemmas evaluate numbers,
not constructed graphs. These lemmas are not graph-level formalizations of
the revised paper's full results.

In particular, this project does not formalize the classical graph bound
`W <= R`, the revised five-term interval chain and its connected sharpness
family for every maximum degree, or domination of the TxGraffiti estimate
by `R`. The corresponding mathematical proofs are in the paper.
