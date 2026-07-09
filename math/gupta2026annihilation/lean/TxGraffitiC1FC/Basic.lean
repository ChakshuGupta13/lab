import Mathlib.Combinatorics.SimpleGraph.Clique
import Mathlib.Combinatorics.SimpleGraph.Finite
import Mathlib.Combinatorics.SimpleGraph.DegreeSum
import Mathlib.Data.Rat.Defs

/-!
# 0.3 smoke test — invariant availability on the pinned Mathlib

Confirms (compiler as oracle) that the Mathlib symbols the C1 proof depends on
exist on this pin (`master-2026-07-07` / `v4.32.0-rc1`):

* `SimpleGraph.indepNum`  — the independence number α
* `SimpleGraph.maxDegree` — Δ
* `SimpleGraph.IsIndepSet.card_le_indepNum` — API used by the Δ=2 branch
* `SimpleGraph.sum_degrees_eq_twice_card_edges` — handshake, for the vehicle

The other two invariants (annihilation number, residue) are copied locally from
FormalConjecturesForMathlib in a later step.
-/

open SimpleGraph

#check @SimpleGraph.indepNum
#check @SimpleGraph.maxDegree
#check @SimpleGraph.IsIndepSet.card_le_indepNum
#check @SimpleGraph.sum_degrees_eq_twice_card_edges
