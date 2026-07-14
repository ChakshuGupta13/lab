import OrigamiCone.AcellExtremaWitness

/-!
# Bidirectional characterizations of `acell`'s strict-local extrema

This module bundles the existence (`AcellExtremaWitness.lean`) and
uniqueness (`AcellGradient.lean`, `AcellMin.lean`) theorems into compact
`iff` characterizations:

* The strict local maxima of `acell` are exactly `(m − 1, n − 1)`.
* The strict local minima of `acell` are exactly `(0, 0)`.

These wrap-ups let downstream code refer to the **locus** of strict
extrema directly, rather than chaining an existence proof with a
uniqueness proof at each use site.

Hypothesis conventions (inherited from the uniqueness theorems):

* `acell_strictLocalMax_iff` needs `1 ≤ m, 1 ≤ n` (existence & uniqueness
  both hold under this).
* `acell_strictLocalMin_iff` needs `2 ≤ m, 2 ≤ n` (uniqueness fails for
  `m = 1` or `n = 1`: the singleton-row/column grid `Cell 1 n` is a single
  path and every cell with the smallest `acell` value would be a candidate
  for strict local min).

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Characterization of strict local maxima of `acell`.**
For `m, n ≥ 1`, the cell `q : Cell m n` is a strict local maximum of the
antidiagonal `acell` iff it is the top-right corner `(m − 1, n − 1)`. -/
theorem acell_strictLocalMax_iff (hm : 1 ≤ m) (hn : 1 ≤ n) (q : Cell m n) :
    IsStrictLocalMax acell q
      ↔ q = (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩) := by
  refine ⟨acell_unique_max hm hn q, ?_⟩
  rintro rfl
  exact acell_strictMax_topRight hm hn

/-- **Characterization of strict local minima of `acell`.**
For `m, n ≥ 2`, the cell `q : Cell m n` is a strict local minimum of the
antidiagonal `acell` iff it is the origin `(0, 0)`. -/
theorem acell_strictLocalMin_iff (hm : 2 ≤ m) (hn : 2 ≤ n) (q : Cell m n) :
    IsStrictLocalMin acell q
      ↔ q = (⟨0, by omega⟩, ⟨0, by omega⟩) := by
  refine ⟨acell_unique_min hm hn q, ?_⟩
  rintro rfl
  exact acell_strictMin_origin (by omega) (by omega)

end OrigamiCone
