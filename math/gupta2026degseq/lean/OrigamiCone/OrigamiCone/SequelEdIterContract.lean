import OrigamiCone.SequelEdContractExtremum

/-!
# Sequel: iterated contract map (Task E.δ.h — full paper `lem:uniform`)

The atomic `contractAt` step removes one frozen column at a time.  Iterating
until no frozen column remains gives the paper's contraction map: every
height function reduces to a "no frozen column" height function on a smaller
grid, preserving the extremum count.

## Theorems

* **`exists_reduced_form`** — the paper's contraction map: every `IsHeight`
  function on `Cell m n` reduces to a "no frozen column" height function on
  `Cell m w` (with `w ≤ n`), preserving the extremum count.

## Substrate

Imports `SequelEdContractExtremum` (for `contractAt`, `contractAt_isHeight`,
`contractAt_numExtrema_eq`).  Standalone.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m : ℕ}

/-- **Iterated contract.**  Every `IsHeight` function on `Cell m n` reduces to
a "no frozen column" height function on `Cell m w` (with `w ≤ n`), preserving
the extremum count.

Proof by strong induction on `n`.  If `h` has any frozen column, apply
`contractAt` to get a height function on `Cell m (n - 1)` (which has the same
extremum count by `contractAt_numExtrema_eq`), then invoke the IH.  Otherwise
`h` itself is reduced. -/
theorem exists_reduced_form (hm : 0 < m) : ∀ (n : ℕ) (h : Cell m n → ℤ), IsHeight h →
    ∃ (w : ℕ) (h' : Cell m w → ℤ),
      IsHeight h' ∧
      (∀ j : Fin w, ¬ frozenColumn h' j) ∧
      (Finset.univ.filter (IsStrictLocalExtremum h')).card =
        (Finset.univ.filter (IsStrictLocalExtremum h)).card ∧
      w ≤ n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro h hh
    by_cases hfrz_exists : ∃ j : Fin n, frozenColumn h j
    · obtain ⟨j, hj_frz⟩ := hfrz_exists
      have hh' := contractAt_isHeight h hh hm j hj_frz
      have hn_ge_2 : 2 ≤ n := by
        obtain ⟨hj0, hj1, _⟩ := hj_frz
        have := j.isLt; omega
      have hn_sub_lt : n - 1 < n := by omega
      obtain ⟨w, h'', hh'', hnofrz, hnum_eq'', hw_le⟩ :=
        ih (n - 1) hn_sub_lt (contractAt h hm j hj_frz) hh'
      refine ⟨w, h'', hh'', hnofrz, ?_, ?_⟩
      · rw [hnum_eq'', contractAt_numExtrema_eq h hh hm j hj_frz]
      · omega
    · push_neg at hfrz_exists
      exact ⟨n, h, hh, hfrz_exists, rfl, le_refl _⟩

end OrigamiCone.Sequel
