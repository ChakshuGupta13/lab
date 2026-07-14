import OrigamiCone.SequelEdIterContract

/-!
# Sequel: Paper's iterated contract (stopping at ReducedHF form)

`exists_reduced_form` (from `SequelEdIterContract`) removes ALL frozen columns.
The paper's contraction (`lem:uniform`) is subtly different — it collapses each
maximal frozen run to a single column, leaving ISOLATED frozen columns as "run
markers".  This module supplies the paper's variant.

## Theorem

* **`exists_paper_reduced_form`** — every `IsHeight` function on `Cell m n`
  reduces (via iterated `contractAt`) to one with no ADJACENT frozen columns.
  Isolated frozen columns are allowed (they mark run positions).  Preserves
  the extremum count.  `w ≤ n`.

## Substrate

Imports `SequelEdIterContract` (for `contractAt`, `contractAt_isHeight`,
`contractAt_numExtrema_eq`).  Standalone.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m : ℕ}

/-- **Paper-style iterated contract.**  Reduces any `IsHeight` height function
to one with no adjacent frozen columns (isolated frozen columns are allowed —
these are the "run markers" in the paper's contraction picture).  This is the
substrate for the paper's `ContractionType` bookkeeping (`lem:uniform`), where
the reduced form's isolated frozen columns index the frozen runs of the
original.

Contrast `exists_reduced_form` (from `SequelEdIterContract`), which removes
ALL frozen columns.  That form loses the run-count information; the paper
retains it via the isolated frozen-column markers.

Proof by strong induction on `n`.  If there exist two adjacent frozen columns,
contract one of them via `contractAt` (the pair reduces to a single frozen
column, or both become non-frozen after the shift); invoke the IH on
`Cell m (n - 1)`.  Otherwise `h` is already paper-reduced. -/
theorem exists_paper_reduced_form (hm : 0 < m) :
    ∀ (n : ℕ) (h : Cell m n → ℤ), IsHeight h →
      ∃ (w : ℕ) (h' : Cell m w → ℤ),
        IsHeight h' ∧
        (∀ j : Fin w, ∀ (hj1 : j.val + 1 < w),
          ¬ (frozenColumn h' j ∧ frozenColumn h' ⟨j.val + 1, hj1⟩)) ∧
        (Finset.univ.filter (IsStrictLocalExtremum h')).card =
          (Finset.univ.filter (IsStrictLocalExtremum h)).card ∧
        w ≤ n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro h hh
    by_cases hadj_frz : ∃ j : Fin n, ∃ (hj1 : j.val + 1 < n),
        frozenColumn h j ∧ frozenColumn h ⟨j.val + 1, hj1⟩
    · obtain ⟨j, hj1, hj_frz, _⟩ := hadj_frz
      have hh' := contractAt_isHeight h hh hm j hj_frz
      have hn_ge_2 : 2 ≤ n := by
        obtain ⟨hj0, _, _⟩ := hj_frz
        have := j.isLt; omega
      have hn_sub_lt : n - 1 < n := by omega
      obtain ⟨w, h'', hh'', hnadj, hnum_eq'', hw_le⟩ :=
        ih (n - 1) hn_sub_lt (contractAt h hm j hj_frz) hh'
      refine ⟨w, h'', hh'', hnadj, ?_, ?_⟩
      · rw [hnum_eq'', contractAt_numExtrema_eq h hh hm j hj_frz]
      · omega
    · refine ⟨n, h, hh, ?_, rfl, le_refl _⟩
      intro j hj1 ⟨h_frz, h_frz_next⟩
      exact hadj_frz ⟨j, hj1, h_frz, h_frz_next⟩

end OrigamiCone.Sequel
