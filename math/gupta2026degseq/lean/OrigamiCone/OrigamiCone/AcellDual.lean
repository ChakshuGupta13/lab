import OrigamiCone.AcellMin

/-!
# `-acell` is the cone at the bottom-left corner

Completion of the `acell` foundational story.  The paper's `thm:diam`
construction uses the **two** opposite-corner gradients `h_{++} = acell`
and `h_{--} = -acell`.  `AcellGradient.lean` identified `acell` as the
cone at the top-right corner; this module identifies `-acell` as the
cone at the bottom-left corner:

  `-acell = coneC ⟨0, _⟩ ⟨0, _⟩ 0`        (for `m, n ≥ 1`).

The two opposite corners are at opposite ends of the antidiagonal,
attaining its min and max respectively.

This is the second of the two `cornerGradient_medianMin` witnesses;
together they realize the paper's lower-bound argument symbolically.

Result:
* `negAcell_eq_coneC_bottom` — `(fun v => -acell v) = coneC ⟨0, _⟩ ⟨0, _⟩ 0`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`-acell` is the cone at the bottom-left corner with peak 0.**

For `m, n ≥ 1`, `(fun v => -acell v) = coneC ⟨0, _⟩ ⟨0, _⟩ 0`:

  `-(i + j : ℤ) = 0 − (|0 − i| + |0 − j|) = -(i + j)`.

This is the dual of `acell_eq_coneC_top` (which expresses `acell` as the
cone at the top-right corner with peak `m + n − 2`).  Together the two
identifications cover both of the paper's `h_{++}`, `h_{--}` corner
gradients used in the `thm:diam` lower-bound construction. -/
theorem negAcell_eq_coneC_bottom (hm : 1 ≤ m) (hn : 1 ≤ n) :
    (fun v : Cell m n => -acell v)
      = coneC (⟨0, by omega⟩, ⟨0, by omega⟩) (0 : ℤ) := by
  funext v
  unfold acell coneC gdist
  -- Goal: -((v.1.val : ℤ) + v.2.val) =
  --   (0 : ℤ) - ↑((|(⟨0, _⟩.val : ℤ) - v.1.val|.natAbs +
  --              |(⟨0, _⟩.val : ℤ) - v.2.val|.natAbs : ℕ))
  -- After Fin.mk reduction: ⟨0, _⟩.val = 0; so each natAbs term is
  -- |0 - v.i.val|.natAbs = v.i.val (since v.i.val ≥ 0).
  -- Hence RHS = 0 - (v.1.val + v.2.val) = -acell v.
  have hv1 : (0 : ℤ) ≤ (v.1.val : ℤ) := by positivity
  have hv2 : (0 : ℤ) ≤ (v.2.val : ℤ) := by positivity
  -- After push_cast the goal has |0 - ↑v.1.val| and |0 - ↑v.2.val|.
  -- Each argument is nonpos (since ↑v.i.val ≥ 0), so |·| = -(·).
  push_cast
  rw [abs_of_nonpos (by linarith : (0 : ℤ) - (v.1.val : ℤ) ≤ 0),
      abs_of_nonpos (by linarith : (0 : ℤ) - (v.2.val : ℤ) ≤ 0)]
  ring

end OrigamiCone
