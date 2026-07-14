import Mathlib

/-!
# Sequel meta-theorem: the enumerative kernel of `lem:binom`

Standalone formalisation of the counting kernel behind `Lemma lem:binom` of the
sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`lem:binom` counts the one-dimensional `±1` walks on the path `P_n` with a given
number of strict local minima and maxima: with `d = a + b` it gives
`f_{(a,b)}(n) = (1 + [a=b]) · C(n-2, d-2)`. The proof encodes a walk by its step
string and reads off the count from the **runs** (maximal monotone stretches): a
walk with `r` runs has `r - 1` interior **turns** (sign changes) and `2`
endpoints, all extrema, so `a + b = r + 1`, i.e. `r = d - 1` and the number of
turns is `d - 2`. A run pattern is then a choice of which `d-2` of the `n-2`
interior step-junctions are turns, of which there are `C(n-2, d-2)`; the leading
factor `1 + [a=b]` is the number of admissible first-step directions (one for an
unbalanced split, two for a balanced one).

This module proves the **combinatorial kernel** — the turn-placement count — and
its partition companion:

* `card_turnPatterns` : the number of turn patterns `Fin N → Bool` with exactly
  `c` turns is `C(N, c)` (here `N = n-2` junctions, `c = d-2` turns);
* `runs_count` : restated in run-language, the patterns with exactly `r` runs
  number `C(N, r-1)`;
* `sum_card_turnPatterns` : these partition all `2^N` turn patterns
  (`∑_c C(N,c) = 2^N`), confirming the count is a genuine partition by turn
  number.

The kernel is the `C(n-2, d-2)` factor of `f_{(a,b)}`. The remaining wrapper —
the bijection between walks and `(first-step direction, turn set)` and the
`1 + [a=b]` first-direction multiplicity — is the alternation structure of the
companion module `SequelWalk` (whose telescoping invariant
`numMin - numMax = w(head) - w(last)` reads the split off the endpoints) and is
documented there; it is not reproved here.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.card_turnPatterns`.
-/

namespace OrigamiCone.Sequel

open Finset

/-- The turn patterns on `N` interior junctions with exactly `c` turns: boolean
functions `Fin N → Bool` (`true` = a turn) with exactly `c` `true` values. -/
def turnPatterns (N c : ℕ) : Finset (Fin N → Bool) :=
  univ.filter (fun f => (univ.filter (fun i => f i = true)).card = c)

/-- **Turn-placement count** (enumerative kernel of `lem:binom`). The number of
turn patterns on `N` junctions with exactly `c` turns is `C(N, c)`. With
`N = n-2` and `c = d-2` this is the `C(n-2, d-2)` factor of `f_{(a,b)}(n)`. -/
theorem card_turnPatterns (N c : ℕ) : (turnPatterns N c).card = N.choose c := by
  have key : (powersetCard c (univ : Finset (Fin N))).card = N.choose c := by
    rw [card_powersetCard, card_univ, Fintype.card_fin]
  rw [← key]
  refine card_nbij' (fun f => univ.filter (fun i => f i = true))
                    (fun S => fun i => decide (i ∈ S)) ?_ ?_ ?_ ?_
  · intro f hf
    simp only [turnPatterns, coe_filter, mem_univ, true_and, Set.mem_setOf_eq] at hf
    simp only [mem_coe, mem_powersetCard]
    exact ⟨filter_subset _ _, hf⟩
  · intro S hS
    simp only [mem_coe, mem_powersetCard] at hS
    simp only [turnPatterns, coe_filter, mem_univ, true_and, Set.mem_setOf_eq,
               decide_eq_true_eq]
    rw [← hS.2]; congr 1; ext i; simp
  · intro f hf
    funext i; simp
  · intro S hS
    ext i; simp

/-- The number of **runs** of a turn pattern: one more than its number of turns
(a walk with `t` sign changes has `t + 1` maximal monotone stretches). -/
def runs {N : ℕ} (f : Fin N → Bool) : ℕ :=
  (univ.filter (fun i => f i = true)).card + 1

/-- **Run-language form.** A turn pattern has `r` runs iff it has `r-1` turns, so
the patterns with exactly `r ≥ 1` runs number `C(N, r-1)`. With `N = n-2` and
`r = d-1` this is the `C(n-2, d-2)` count of `lem:binom`. -/
theorem runs_count (N r : ℕ) (hr : 1 ≤ r) :
    (univ.filter (fun f : Fin N → Bool => runs f = r)).card = N.choose (r - 1) := by
  have : (univ.filter (fun f : Fin N → Bool => runs f = r))
      = turnPatterns N (r - 1) := by
    apply filter_congr
    intro f _
    simp only [runs]
    omega
  rw [this, card_turnPatterns]

/-- **Partition companion.** Summing over the turn number recovers all `2^N` turn
patterns, confirming `turnPatterns N ·` partitions the boolean cube by turn
count. -/
theorem sum_card_turnPatterns (N : ℕ) :
    ∑ c ∈ range (N + 1), (turnPatterns N c).card = 2 ^ N := by
  simp only [card_turnPatterns]
  exact Nat.sum_range_choose N

end OrigamiCone.Sequel
