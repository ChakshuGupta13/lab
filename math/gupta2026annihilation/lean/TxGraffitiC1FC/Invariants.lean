import Mathlib.Combinatorics.SimpleGraph.Finite
import Mathlib.Data.Finset.Sort
import Mathlib.Data.Multiset.Sort
import Mathlib.Data.Multiset.Interval

/-!
# Local copies of the two invariants not in Mathlib

`annihilationNumber` and `residue` live in `FormalConjecturesForMathlib`, which
we do not depend on during development (see repo memory: version-decision).
These are faithful copies from that source (module-system boilerplate stripped),
to be reconciled with upstream at transplant time.

`residueAux` was FC's `partial def`; refactored here (item 4) to well-founded
recursion (`termination_by l.length` via `havelHakimiStep_length_cons`) so
Favaron's `R ≤ α` (item 6) can reason about it inductively. The computed function
is unchanged; this refactored definition is the version to contribute upstream at
transplant (item 8).
-/

namespace SimpleGraph
open Classical

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The multiset of degrees of a graph. (FC `Degrees.lean`.) -/
def degreeMultiset (G : SimpleGraph α) [DecidableRel G.Adj] : Multiset ℕ :=
  Finset.univ.val.map fun v => G.degree v

/-- The annihilation number: the largest cardinality of a sub-multiset of the
degree multiset whose sum is at most the number of edges. (FC
`AnnihilationNumber.lean`.) -/
def annihilationNumber (G : SimpleGraph α) [DecidableRel G.Adj] : ℕ :=
  letI limit := G.edgeFinset.card
  Finset.Iic (degreeMultiset G)
    |>.filter (fun S ↦ Multiset.sum S ≤ limit)
    |>.sup Multiset.card

/-- One Havel–Hakimi reduction step on a descending-sorted degree list. (FC
`Residue.lean`.) -/
def havelHakimiStep (s : List ℕ) : List ℕ :=
  match s with
  | [] => []
  | d :: rest =>
    let (to_decrement, remaining) := rest.splitAt d
    let decremented := to_decrement.map (· - 1)
    (decremented ++ remaining).mergeSort (· ≥ ·)

/-- `havelHakimiStep` drops the list length by exactly one on a nonempty list:
`|havelHakimiStep (d :: rest)| = |rest|` (`splitAt` partitions `rest`; `map`, `++`,
and `mergeSort` all preserve the total length). This is the termination measure for
the well-founded `residueAux` below. -/
theorem havelHakimiStep_length_cons (d : ℕ) (rest : List ℕ) :
    (havelHakimiStep (d :: rest)).length = rest.length := by
  simp only [havelHakimiStep, List.splitAt_eq, List.length_mergeSort,
    List.length_append, List.length_map, List.length_take, List.length_drop]
  omega

/-- Residue via iterated Havel–Hakimi until all-zero. (FC `Residue.lean`, refactored
from a `partial def` to well-founded recursion — item 4 — via
`havelHakimiStep_length_cons`, so it admits equational/inductive reasoning; the
computed function is unchanged.) -/
def residueAux : List ℕ → ℕ
  | [] => 0
  | 0 :: s => 1 + s.length
  | d :: rest => residueAux (havelHakimiStep (d :: rest))
termination_by l => l.length
decreasing_by
  simp_wf
  rw [havelHakimiStep_length_cons]
  omega

/-- The residue of a graph. (FC `Residue.lean`.) -/
noncomputable def residue (G : SimpleGraph α) [DecidableRel G.Adj] : ℕ :=
  residueAux ((Finset.univ.val.map fun v => G.degree v).sort (· ≥ ·))

end SimpleGraph
