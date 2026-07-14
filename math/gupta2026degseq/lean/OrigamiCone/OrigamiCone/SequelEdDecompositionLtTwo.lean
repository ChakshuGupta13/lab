import OrigamiCone.SequelEd
import OrigamiCone.SequelContractionType

/-!
# Sequel: `Ed_decomposition` — trivial case `d < 2`

For `d < 2` on any nonempty grid with `mn ≥ 2`, `Ed d m n = 0` (via
`Ed_lt_two_eq_zero`).  The decomposition axiom is satisfied by taking the
empty Finset of contraction types.

This converts the `d < 2` branch of `Ed_decomposition` from axiomatic to
theoremhood.  The nontrivial `d ≥ 2` case remains axiomatic (paper's Lemma 8.5
proof body — the frozen-contraction bijection).
-/

namespace OrigamiCone.Sequel

open Polynomial

/-- **`Ed_decomposition` at `d < 2`, proved.**  For `d ∈ {0, 1}` on any grid
with `m ≥ 2`, `Ed d m n = 0` (no height function has fewer than 2 extrema on
a nonempty grid), so the empty Finset satisfies the decomposition trivially. -/
theorem Ed_decomposition_of_lt_two (d : ℕ) (hd : d < 2) :
    ∀ m : ℕ, 2 ≤ m →
      ∃ S : Finset ContractionType,
        (∀ C ∈ S, C.runCount ≤ d + 1) ∧
        (∀ C ∈ S, C.width ≤ 2 * d + 3) ∧
        (∀ n : ℕ, 2 * d + 4 ≤ n →
          (Ed d m n : ℚ) = ∑ C ∈ S, (C.count n : ℚ)) := by
  intro m hm
  refine ⟨∅, ?_, ?_, ?_⟩
  · intro C hC; exact absurd hC (Finset.notMem_empty _)
  · intro C hC; exact absurd hC (Finset.notMem_empty _)
  · intro n hn
    rw [Finset.sum_empty]
    have hn_ge : 2 ≤ n := by omega
    have hmn_ge : 2 ≤ m * n := by nlinarith
    have h_zero : Ed d m n = 0 :=
      Ed_lt_two_eq_zero hd (by omega) (by omega) hmn_ge
    exact_mod_cast h_zero

end OrigamiCone.Sequel
