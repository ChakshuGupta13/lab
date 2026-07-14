import OrigamiCone.AcellGdist

/-!
# Manhattan distance from the remaining two corners

Companion to `AcellGdist.lean` (40713e7), which gave Manhattan distance
formulas for the corner `(0, 0)` (bottom-left in the codebase's
row-0-is-bottom convention) and `(m − 1, n − 1)` (top-right).  This module
adds the formulas for the **other two corners**:

  `gdist (⟨0, _⟩, ⟨n − 1, _⟩) v    = i + (n − 1 − j)`,
  `gdist (⟨m − 1, _⟩, ⟨0, _⟩) v    = (m − 1 − i) + j`.

These four formulas together cover every corner-distance computation
the paper uses in its family analyses for `thm:deg4count` (e.g.
"adjacent corners share a coordinate, so their cone-pair envelope is a
tent in the orthogonal direction").

Results:
* `gdist_bottomRight` — distance from `(0, n − 1)`.
* `gdist_topLeft`     — distance from `(m − 1, 0)`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Distance from the bottom-right corner `(0, n − 1)`.**
For `m, n ≥ 1`,
  `gdist (⟨0, _⟩, ⟨n − 1, _⟩) v = i + (n − 1) − j`.

The bottom-right corner shares row coordinate `0` with the bottom-left
(distance `i + j`) but has the opposite column.  The proof unfolds
`gdist`, uses `v.1.val ≥ 0` and `v.2.val ≤ n − 1` to resolve the absolute
values, then `ring` closes. -/
theorem gdist_bottomRight (hm : 1 ≤ m) (hn : 1 ≤ n) (v : Cell m n) :
    gdist (⟨0, by omega⟩, ⟨n - 1, by omega⟩) v
      = (v.1.val : ℤ) + ((n - 1 : ℕ) : ℤ) - v.2.val := by
  unfold gdist
  have h1 : (0 : ℤ) ≤ (v.1.val : ℤ) := by positivity
  have h2 : v.2.val ≤ n - 1 := by have := v.2.isLt; omega
  have h2_cast : (v.2.val : ℤ) ≤ ((n - 1 : ℕ) : ℤ) := by exact_mod_cast h2
  push_cast
  rw [abs_of_nonpos (by linarith : (0 : ℤ) - (v.1.val : ℤ) ≤ 0),
      abs_of_nonneg (by linarith : (0 : ℤ) ≤ ((n - 1 : ℕ) : ℤ) - (v.2.val : ℤ))]
  ring

/-- **Distance from the top-left corner `(m − 1, 0)`.**
For `m, n ≥ 1`,
  `gdist (⟨m − 1, _⟩, ⟨0, _⟩) v = (m − 1) − i + j`.

The top-left corner shares column coordinate `0` with the bottom-left
but has the opposite row.  Symmetric to `gdist_bottomRight`. -/
theorem gdist_topLeft (hm : 1 ≤ m) (hn : 1 ≤ n) (v : Cell m n) :
    gdist (⟨m - 1, by omega⟩, ⟨0, by omega⟩) v
      = ((m - 1 : ℕ) : ℤ) - v.1.val + v.2.val := by
  unfold gdist
  have h1 : v.1.val ≤ m - 1 := by have := v.1.isLt; omega
  have h1_cast : (v.1.val : ℤ) ≤ ((m - 1 : ℕ) : ℤ) := by exact_mod_cast h1
  have h2 : (0 : ℤ) ≤ (v.2.val : ℤ) := by positivity
  push_cast
  rw [abs_of_nonneg (by linarith : (0 : ℤ) ≤ ((m - 1 : ℕ) : ℤ) - (v.1.val : ℤ)),
      abs_of_nonpos (by linarith : (0 : ℤ) - (v.2.val : ℤ) ≤ 0)]
  ring

end OrigamiCone
