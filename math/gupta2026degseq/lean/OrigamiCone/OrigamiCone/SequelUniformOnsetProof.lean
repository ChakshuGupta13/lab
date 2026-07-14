import OrigamiCone.SequelContractionType
import OrigamiCone.SequelEd
import OrigamiCone.SequelEdDecompositionLtTwo
import OrigamiCone.SequelEdDecompositionThm

/-!
# Substrate 5/5: Assembly — from `Ed_decomposition` to the uniform-onset witness

This module ships the last substrate step for discharging the paper's
`Lemma 8.5` (`lem:uniform`).  It introduces a **granular axiom
`Ed_decomposition`** — the paper's contraction-bijection identity at
`Ed` — and derives the **`Ed_uniform_onset` statement** as a THEOREM
from it, using substrates 1–4.

## The axiom shift

Prior to this module, `SequelUniformOnset.lean` axiomatised the whole
uniform-onset statement (`Ed d m ·` is a single polynomial of degree
`≤ d` on `n ≥ N_d`, `N_d` `m`-free).  This module axiomatises the
finer **decomposition identity**: for each `d` and every `m ≥ 2`, there
is a finite set `S` of contraction types (each with bounded `width` and
`runCount`) such that

  `Ed d m n = Σ C ∈ S, C.count n`   for `n ≥ 2d + 4`

and each `C ∈ S` satisfies `C.runCount ≤ d + 1` and `C.width ≤ 2d + 3`.
The uniform-onset polynomial witness then falls out of substrate 4's
`sum_eval_of_le` and `sum_natDegree_le`, applied with `D := d`.

The remaining axiom `Ed_decomposition` is the combinatorial content of
the paper's Lemma 8.5 proof body — the frozen-contraction bijection
between `d`-extremum height functions on `G_{m,n}` and pairs `(C, ℓ)`
of a contraction type `C` and a run-length vector `ℓ`.  Formalising
`Ed_decomposition` is a well-defined multi-session task on top of
`SequelEdTransfer`'s column decomposition (5c infrastructure); the
combinatorial primitives it needs are in substrates 1–4.

## Result

`Ed_uniform_onset_of_decomposition : ∀ d, ∃ N, ∀ m ≥ 2, ∃ p : Polynomial ℚ,
  p.natDegree ≤ d ∧ ∀ n ≥ N, (Ed d m n : ℚ) = p.eval (n : ℚ)` — the exact
statement axiomatised in `SequelUniformOnset.lean`, now proved.
`SequelUniformOnset` is updated separately to reuse this theorem, so
`Ed_thm_poly_unconditional`'s axiom trace shifts from `Ed_uniform_onset`
to `Ed_decomposition`.
-/

namespace OrigamiCone.Sequel

open Polynomial

/-- **Ed decomposition axiom** (paper's `Lemma 8.5` proof body).

For each `d : ℕ` and every `m ≥ 2`, there is a finite set `S` of
`ContractionType`s such that:

* every `C ∈ S` has `runCount ≤ d + 1` (from the active-column bound
  `# active middle columns ≤ d` — substrate 3);
* every `C ∈ S` has `width ≤ 2d + 3` (from `2` boundary columns
  `+ ≤ d` active middle columns `+ ≤ d + 1` contracted frozen runs);
* for every `n ≥ 2d + 4` (i.e., strictly greater than every type's
  width), `Ed d m n = ∑ C ∈ S, C.count n`.

The identity `Ed d m n = ∑ C ∈ S, mult C * C.count n` for `n ≥ 2d + 4` is
the frozen-contraction bijection between `d`-extremum height functions on
`G_{m,n}` and pairs `(C, ℓ)` of a contraction type `C` and a run-length
vector `ℓ` — the paper's `Lemma 8.5` proof body.  Substrates 1–4 provide
the combinatorial primitives that a full formalisation of this axiom
would use; the axiom itself is the residual pending obligation.

The `mult : ContractionType → ℕ` captures multiplicity — distinct paper-reduced
forms sharing the same `(width, runCount)` collapse to one element of the
`Finset ContractionType` `S` (since `ContractionType` stores only those
fields), so `mult C` records how many reduced forms have the same `(width,
runCount)` as `C`.  For unique-per-`(W, r)` cases `mult ≡ 1` and this reduces
to the naive sum `∑ C ∈ S, C.count n`.

**Scope narrowing** (2026-07-06): The axiom now assumes `2 ≤ d` (the
nontrivial case).  The `d < 2` case is proved separately in
`SequelEdDecompositionLtTwo.Ed_decomposition_of_lt_two` (trivially: `Ed d m n
= 0` by `Ed_lt_two_eq_zero`, so the empty Finset satisfies the decomposition).
Combining the two yields `Ed_decomposition`, now a THEOREM (below).

**Axiom closure** (2026-07-06, follow-up): The former axiom
`Ed_decomposition_of_ge_two` is now a *theorem* citing
`Ed_decomposition_of_ge_two_thm` (from `SequelEdDecompositionThm`),
which is proved from the fine-grained fiber-cardinality axioms
`fiber_card_axiom` + `fiber_card_zero_axiom` (from
`SequelEdFiberCardAxiom`) plus the partition + disjointness (from
`SequelEdFiberPartition`).  The residual axiomatic content is now the
precise combinatorial identity of paper Lemma 8.5 (the run-length-vector
bijection), not the monolithic decomposition. -/
theorem Ed_decomposition_of_ge_two (d : ℕ) (hd : 2 ≤ d) :
    ∀ m : ℕ, 2 ≤ m →
      ∃ (S : Finset ContractionType) (mult : ContractionType → ℕ),
        (∀ C ∈ S, C.runCount ≤ d + 1) ∧
        (∀ C ∈ S, C.width ≤ 2 * d + 3) ∧
        (∀ n : ℕ, 2 * d + 4 ≤ n →
          (Ed d m n : ℚ) = ∑ C ∈ S, (mult C : ℚ) * (C.count n : ℚ)) :=
  Ed_decomposition_of_ge_two_thm d hd

/-- **`Ed_decomposition`** — combines the trivial `d < 2` case
(`Ed_decomposition_of_lt_two`, from `SequelEdDecompositionLtTwo`) with the
narrower `Ed_decomposition_of_ge_two` axiom.  Preserves the original `∀ d`
API but shifts the axiomatic content to the `2 ≤ d` case only. -/
theorem Ed_decomposition (d : ℕ) :
    ∀ m : ℕ, 2 ≤ m →
      ∃ (S : Finset ContractionType) (mult : ContractionType → ℕ),
        (∀ C ∈ S, C.runCount ≤ d + 1) ∧
        (∀ C ∈ S, C.width ≤ 2 * d + 3) ∧
        (∀ n : ℕ, 2 * d + 4 ≤ n →
          (Ed d m n : ℚ) = ∑ C ∈ S, (mult C : ℚ) * (C.count n : ℚ)) := by
  intro m hm
  by_cases hd : 2 ≤ d
  · exact Ed_decomposition_of_ge_two d hd m hm
  · push_neg at hd
    -- Trivial case: Ed d m n = 0, so ∅ with any mult works.
    refine ⟨∅, fun _ => 0, ?_, ?_, ?_⟩
    · intro C hC; exact absurd hC (Finset.notMem_empty _)
    · intro C hC; exact absurd hC (Finset.notMem_empty _)
    · intro n hn
      rw [Finset.sum_empty]
      have hn_ge : 2 ≤ n := by omega
      have hmn_ge : 2 ≤ m * n := by nlinarith
      have h_zero : Ed d m n = 0 :=
        Ed_lt_two_eq_zero hd (by omega) (by omega) hmn_ge
      exact_mod_cast h_zero

/-- **Uniform onset from decomposition** (paper's `Lemma 8.5`, statement
matching the prior `SequelUniformOnset.Ed_uniform_onset` axiom).

For each `d`, there is a threshold `N` (independent of `m`) such that
for every `m ≥ 2`, `Ed d m ·` agrees on `{n ≥ N}` with a single polynomial
in `n` of natural degree at most `d`.

Proof: instantiate `Ed_decomposition` to get a decomposition of `Ed d m ·`
as a finite sum of contraction-type composition counts; use substrate 4's
`sum_eval_of_le` to package the sum as `(Σ_C poly C).eval` for `n ≥ 2d + 4`;
use substrate 4's `sum_natDegree_le` (with `D := d`, given every
`runCount ≤ d + 1`) to bound the polynomial degree by `d`.  The uniform
threshold `N := 2 * d + 4` follows from the width bound `≤ 2d + 3`
common to every type. -/
theorem Ed_uniform_onset_of_decomposition (d : ℕ) :
    ∃ N : ℕ, ∀ m : ℕ, 2 ≤ m →
      ∃ p : Polynomial ℚ, p.natDegree ≤ d ∧
        ∀ n : ℕ, N ≤ n → (Ed d m n : ℚ) = p.eval (n : ℚ) := by
  refine ⟨2 * d + 4, ?_⟩
  intro m hm
  obtain ⟨S, mult, hrun, hwidth, hdecomp⟩ := Ed_decomposition d m hm
  refine ⟨∑ C ∈ S, (mult C : ℚ) • C.poly, ?_, ?_⟩
  · -- Sum-polynomial degree bound: each term (mult C : ℚ) • C.poly has
    -- natDegree ≤ C.poly.natDegree ≤ C.runCount - 1 ≤ d.  Sum preserves the
    -- bound.
    refine Polynomial.natDegree_sum_le_of_forall_le _ _ ?_
    intro C hCS
    refine (Polynomial.natDegree_smul_le _ _).trans ?_
    have hp := C.poly_natDegree_le
    have hr := hrun C hCS
    omega
  · intro n hn
    -- Widths of every type are ≤ 2d + 3 ≤ n (from hn : 2d + 4 ≤ n).
    have hn_width : ∀ C ∈ S, C.width ≤ n := fun C hCS =>
      (hwidth C hCS).trans (by omega)
    -- Compose the decomposition identity with the sum-eval identity.
    calc (Ed d m n : ℚ)
        = ∑ C ∈ S, (mult C : ℚ) * (C.count n : ℚ) := hdecomp n hn
      _ = ∑ C ∈ S, (mult C : ℚ) * C.poly.eval (n : ℚ) := by
          apply Finset.sum_congr rfl
          intro C hCS
          rw [C.poly_eval_of_le (hn_width C hCS)]
      _ = (∑ C ∈ S, (mult C : ℚ) • C.poly).eval (n : ℚ) := by
          rw [Polynomial.eval_finset_sum]
          apply Finset.sum_congr rfl
          intro C hCS
          rw [Polynomial.eval_smul, smul_eq_mul]

end OrigamiCone.Sequel
