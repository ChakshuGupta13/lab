import OrigamiCone.AcellGdist
import OrigamiCone.ConePair
import OrigamiCone.GdistCorners

/-!
# Cone-pair envelope at opposite-corner apexes (Sub-3 of `thm:deg4count`)

The paper's CC|EE family analysis (Section 3, main.tex L666-672) splits
into two cases for the apex pair of grid corners:

* **Adjacent corners** (share a side): handled by `CpeSharedCoord`
  (b675f8c) — `cpe` is monotone in the orthogonal direction, so the
  apex pair contributes no degree-4 vertex.
* **Opposite corners** (diagonally placed pair): `cpe` reduces to a
  one-dimensional **tent** in the appropriate linear coordinate, and the
  2D maxima problem reduces to finding the peak of that tent.

This module formalises the **opposite-corner reduction** explicitly for
both pairs of opposite corners on a grid:

* BL = `(0, 0)` and TR = `(m − 1, n − 1)` (the main diagonal): both
  distances depend only on `acell v = v.1.val + v.2.val`.
* BR = `(0, n − 1)` and TL = `(m − 1, 0)` (the anti-diagonal): both
  distances depend only on the diagonal coordinate `v.1.val − v.2.val`.

These reductions immediately let downstream analyses (counting admissible
`δ` for degree-4 outcomes) work with one-variable tent functions rather
than 2D `cpe`.

Results:
* `cpe_BL_TR_eq_acell_tent` — `cpe BL TR δ v = min(acell v, δ + (m+n-2) - acell v)`.
* `cpe_BR_TL_eq_diag_tent`  — `cpe BR TL δ v = min(i + (n-1) - j, δ + (m-1) - i + j)`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Cone-pair envelope at the BL–TR opposite-corner pair.**

For `m, n ≥ 1`, with `BL = (0, 0)` and `TR = (m − 1, n − 1)`,

  `cpe BL TR δ v = min (acell v, δ + (m + n − 2) − acell v)`.

This is a tent in `acell v = i + j`: as `acell` ranges from `0` to
`m + n − 2`, the LHS rises linearly with slope `+1` until the two cone
values meet, then falls with slope `−1`.  All cells with the same
`acell` value (i.e. on the same antidiagonal) yield the same `cpe`.

Proof: directly substitute `gdist BL v = acell v`
(`acell_eq_gdist_origin`) and `gdist TR v = (m + n − 2) − acell v`
(`gdist_topRight_eq_complement_acell`). -/
theorem cpe_BL_TR_eq_acell_tent (hm : 1 ≤ m) (hn : 1 ≤ n) (δ : ℤ) (v : Cell m n) :
    cpe (⟨0, by omega⟩, ⟨0, by omega⟩)
        (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩) δ v
      = min (acell v) (δ + ((m + n - 2 : ℕ) : ℤ) - acell v) := by
  unfold cpe
  rw [← acell_eq_gdist_origin hm hn v,
      gdist_topRight_eq_complement_acell hm hn v]
  ring_nf

/-- **Cone-pair envelope at the BR–TL opposite-corner pair.**

For `m, n ≥ 1`, with `BR = (0, n − 1)` and `TL = (m − 1, 0)`,

  `cpe BR TL δ v = min (i + (n − 1) − j, δ + (m − 1) − i + j)`.

This is a tent in the **diagonal coordinate** `v.1.val − v.2.val`: cells
with the same `i − j` value (i.e. on the same diagonal) yield the same
`cpe`.

Proof: directly substitute `gdist BR v = i + (n − 1) − j`
(`gdist_bottomRight`) and `gdist TL v = (m − 1) − i + j` (`gdist_topLeft`). -/
theorem cpe_BR_TL_eq_diag_tent (hm : 1 ≤ m) (hn : 1 ≤ n) (δ : ℤ) (v : Cell m n) :
    cpe (⟨0, by omega⟩, ⟨n - 1, by omega⟩)
        (⟨m - 1, by omega⟩, ⟨0, by omega⟩) δ v
      = min ((v.1.val : ℤ) + ((n - 1 : ℕ) : ℤ) - v.2.val)
            (δ + ((m - 1 : ℕ) : ℤ) - v.1.val + v.2.val) := by
  unfold cpe
  rw [gdist_bottomRight hm hn v, gdist_topLeft hm hn v]
  ring_nf

end OrigamiCone
