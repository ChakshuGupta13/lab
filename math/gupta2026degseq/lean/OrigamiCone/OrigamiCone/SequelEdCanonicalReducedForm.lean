import OrigamiCone.SequelEdPaperReduced
import OrigamiCone.SequelEdCellExpand
import OrigamiCone.SequelEdPaperReducedForm
import OrigamiCone.SequelEdReducedHF

/-!
# Canonical paper-reduced form + `canonicalReducedForm` map

Extends `SequelEdPaperReduced` with the canonical + `2 ≤ w` strengthening
of iterated contraction, and builds the `canonicalReducedForm` map that
sends each canonical height function `h : Cell m n → ℤ` with `numExtrema h
= d` to a `PaperReducedForm m d`.  This is the fiber map for the
`Ed_decomposition_of_ge_two` partition.

## Contents

* `exists_paper_reduced_canonical_ge_two`: strengthened
  `exists_paper_reduced_form` — preserves `IsCanonicalHeight` (via
  `contractAt_isCanonicalHeight`) and concludes `2 ≤ w` when `2 ≤ n` (via
  the observation that adjacent-frozen requires `current_n ≥ 4`, so the
  base case fires at `current_n ≥ 2`).

* `canonicalReducedForm`: `(h : Cell m n → ℤ) → IsCanonicalHeight h →
  numExtrema h = d → 1 ≤ m → 2 ≤ n → 2 ≤ d → PaperReducedForm m d`.  Uses
  `Nonempty.some` (Classical) on the strengthened existence lemma, then
  invokes `ReducedHF.width_le_two_d_sub_one` (via a `ReducedHF` wrapper)
  to close the `hW_upper : W ≤ 2*d + 3` field (with slack; the actual
  bound is `2*d - 1`).

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline
(no new).
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m : ℕ}

/-- **Strengthened paper-reduced iterated contract.**  Same as
`exists_paper_reduced_form` but additionally:

* Preserves `IsCanonicalHeight` (needed for the `PaperReducedForm` structure).
* Concludes `2 ≤ w` when `2 ≤ n` (needed for the `hW : 2 ≤ W` field of
  `PaperReducedForm`; the base case fires at some `current_n ≥ 2` because
  the adjacent-frozen precondition of `contractAt` requires `current_n ≥ 4`,
  so the recursion never enters `n < 4`, and the base case at any given
  `current_n ≥ 2` returns `w = current_n ≥ 2`). -/
theorem exists_paper_reduced_canonical_ge_two (hm : 0 < m) :
    ∀ (n : ℕ) (_ : 2 ≤ n) (h : Cell m n → ℤ), IsCanonicalHeight h →
      ∃ (w : ℕ) (h' : Cell m w → ℤ),
        IsCanonicalHeight h' ∧
        (∀ j : Fin w, ∀ (hj1 : j.val + 1 < w),
          ¬ (frozenColumn h' j ∧ frozenColumn h' ⟨j.val + 1, hj1⟩)) ∧
        (Finset.univ.filter (IsStrictLocalExtremum h')).card =
          (Finset.univ.filter (IsStrictLocalExtremum h)).card ∧
        w ≤ n ∧ 2 ≤ w := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro hn h hCanon
    by_cases hadj_frz : ∃ j : Fin n, ∃ (hj1 : j.val + 1 < n),
        frozenColumn h j ∧ frozenColumn h ⟨j.val + 1, hj1⟩
    · -- Recursive step: adjacent-frozen at some column j
      obtain ⟨j, hj1, hj_frz, hj_frz_next⟩ := hadj_frz
      -- Adjacent-frozen at (j, j+1) requires j+2 < n, hence n ≥ 4.
      have hn_ge_4 : 4 ≤ n := by
        have hj0 : 0 < j.val := hj_frz.1
        have hj_next_next : (j.val + 1) + 1 < n := hj_frz_next.2.1
        omega
      have hCanon_h' : IsCanonicalHeight (contractAt h hm j hj_frz) :=
        contractAt_isCanonicalHeight h hCanon.1 hm j hj_frz hCanon
      have hn_sub_lt : n - 1 < n := by omega
      have hn_sub_ge_2 : 2 ≤ n - 1 := by omega
      obtain ⟨w, h'', hCanon'', hnadj, hnum_eq'', hw_le, hw_ge⟩ :=
        ih (n - 1) hn_sub_lt hn_sub_ge_2 (contractAt h hm j hj_frz) hCanon_h'
      refine ⟨w, h'', hCanon'', hnadj, ?_, ?_, hw_ge⟩
      · rw [hnum_eq'', contractAt_numExtrema_eq h hCanon.1 hm j hj_frz]
      · omega
    · -- Base case: no adjacent frozen at n, so w = n.
      refine ⟨n, h, hCanon, ?_, rfl, le_refl _, hn⟩
      intro j hj1 ⟨h_frz, h_frz_next⟩
      exact hadj_frz ⟨j, hj1, h_frz, h_frz_next⟩

/-- **`canonicalReducedForm h ... : PaperReducedForm m d`** — the fiber-map
target for the `Ed_decomposition_of_ge_two` partition.  Sends each
canonical `d`-extremum height function `h : Cell m n → ℤ` to some
paper-reduced form on `Cell m W → ℤ` with `2 ≤ W ≤ 2*d + 3`.

Uses `Nonempty.some` on the witness produced by
`exists_paper_reduced_canonical_ge_two`; the width bound
`W ≤ 2*d + 3` is discharged by wrapping the reduced form as a `ReducedHF`
and applying `ReducedHF.width_le_two_d_sub_one` (with slack: the actual
bound is `2*d - 1 ≤ 2*d + 3`).

**Ambient hypotheses**: `1 ≤ m`, `2 ≤ n`, and `2 ≤ d` (if `d ≤ 1` then
the `ReducedHF` width bound `W ≤ 2*d - 1 ≤ 1` contradicts `2 ≤ W`, hence
no `PaperReducedForm m d` exists — but the caller already knows this
from `Ed_lt_two_eq_zero`, and this map is only invoked in the `2 ≤ d`
case).

**Implementation note** (2026-07-06 fix): a naive `Nonempty.some hNE`
approach — where `hNE : Nonempty (PaperReducedForm m d)` is built from
the h-dependent existence lemma — collapses to a *constant* function
because `Classical.choice` on `Nonempty α` depends only on `α`, not on
the derivation.  The correct fix is to invoke `Classical.choose`
directly on the h-dependent existential `∃ w, ∃ h', ...`, extracting
`w` and `h'` via `choose_spec` — those results depend on `h` because
the ambient existential statement itself depends on `h`. -/
noncomputable def canonicalReducedForm
    (hm : 1 ≤ m) {n d : ℕ} (hn : 2 ≤ n) (hd : 2 ≤ d)
    (h : Cell m n → ℤ) (hCanon : IsCanonicalHeight h)
    (hExt : numExtrema h = d) : PaperReducedForm m d :=
  -- Extract h-dependent width and reduced height function.  The
  -- existential statement itself depends on `h`, so `Classical.choose`
  -- yields an h-dependent value (unlike `Nonempty.some` on a plain
  -- Nonempty proof).
  let hEx := exists_paper_reduced_canonical_ge_two
    (by omega : 0 < m) n hn h hCanon
  let w : ℕ := Classical.choose hEx
  let hEx2 := Classical.choose_spec hEx
  let h' : Cell m w → ℤ := Classical.choose hEx2
  let hProps := Classical.choose_spec hEx2
  -- `hProps : IsCanonicalHeight h' ∧ (reduced-cond h') ∧
  --   (numExtrema h' = numExtrema h) ∧ w ≤ n ∧ 2 ≤ w`
  let hCanon' : IsCanonicalHeight h' := hProps.1
  let hReduced' :
      ∀ j : Fin w, ∀ (hj1 : j.val + 1 < w),
        ¬ (frozenColumn h' j ∧ frozenColumn h' ⟨j.val + 1, hj1⟩) := hProps.2.1
  let hExt_eq :
      (Finset.univ.filter (IsStrictLocalExtremum h')).card =
        (Finset.univ.filter (IsStrictLocalExtremum h)).card := hProps.2.2.1
  let hw_ge : 2 ≤ w := hProps.2.2.2.2
  -- numExtrema h' = numExtrema h = d
  have hExt_h' : numExtrema h' = d := by
    show (Finset.univ.filter (IsStrictLocalExtremum h')).card = d
    rw [hExt_eq]; exact hExt
  -- Build a ReducedHF to derive w ≤ 2 * d - 1, then w ≤ 2 * d + 3.
  let r : ReducedHF m := {
    W := w, hW := hw_ge, h := h',
    isHeight := hCanon'.1, isCanonical := hCanon'.2,
    reduced := hReduced'
  }
  have hw_upper_reduced : r.W ≤ 2 * d - 1 :=
    ReducedHF.width_le_two_d_sub_one r (by omega) d hExt_h'
  have hw_upper : w ≤ 2 * d + 3 := by
    show r.W ≤ 2 * d + 3
    omega
  ⟨w, hw_ge, hw_upper, h', hCanon'.1, hCanon', hReduced', hExt_h'⟩

end OrigamiCone.Sequel
