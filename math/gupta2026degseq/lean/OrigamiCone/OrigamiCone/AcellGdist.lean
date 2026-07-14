import OrigamiCone.AcellDual

/-!
# `acell` as Manhattan distance from the corner

The antidiagonal `acell v = v.1.val + v.2.val` is exactly the **Manhattan
distance from the bottom-left corner** of the grid; the **complement**
`(m + n − 2) − acell v` is the **Manhattan distance from the top-right
corner**:

  `acell v             = gdist ⟨0, 0⟩ v`,
  `gdist ⟨m−1, n−1⟩ v = (m + n − 2) − acell v`.

These identities are immediate consequences of the cone-form
identifications `negAcell_eq_coneC_bottom` and `acell_eq_coneC_top`
(unfolding the `coneC q C v := C − gdist q v` definition), but are
worth named primitives because they restate the antidiagonal–distance
correspondence in the paper's own geometric language without the cone
wrapping.

Results:
* `acell_eq_gdist_origin` — `acell v = gdist ⟨0, 0⟩ v`.
* `gdist_topRight_eq_complement_acell` — `gdist ⟨m−1, n−1⟩ v = (m + n − 2) − acell v`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **The antidiagonal is the Manhattan distance from the origin.**
For `m, n ≥ 1`, `acell v = gdist ⟨0, 0⟩ v`: the value `i + j` is exactly the
Manhattan distance from the bottom-left corner.  Immediate from
`negAcell_eq_coneC_bottom`: that says `−acell v = 0 − gdist ⟨0, 0⟩ v`, so
`acell v = gdist ⟨0, 0⟩ v`. -/
theorem acell_eq_gdist_origin (hm : 1 ≤ m) (hn : 1 ≤ n) (v : Cell m n) :
    acell v = gdist (⟨0, by omega⟩, ⟨0, by omega⟩) v := by
  -- Unwrap negAcell_eq_coneC_bottom pointwise and simplify.
  have h := congr_fun (negAcell_eq_coneC_bottom hm hn) v
  -- h : -acell v = coneC ⟨0,0⟩ 0 v.  Unfold coneC: = 0 - gdist ⟨0,0⟩ v.
  unfold coneC at h
  linarith

/-- **Distance from the top-right corner equals the antidiagonal complement.**
For `m, n ≥ 1`, `gdist ⟨m−1, n−1⟩ v = (m + n − 2) − acell v`.  Immediate from
`acell_eq_coneC_top`: that says `acell v = (m + n − 2) − gdist ⟨m−1, n−1⟩ v`,
which rearranges to the claim. -/
theorem gdist_topRight_eq_complement_acell (hm : 1 ≤ m) (hn : 1 ≤ n) (v : Cell m n) :
    gdist (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩) v
      = ((m + n - 2 : ℕ) : ℤ) - acell v := by
  -- Unwrap acell_eq_coneC_top pointwise and rearrange.
  have h := congr_fun (acell_eq_coneC_top hm hn) v
  -- h : acell v = coneC ⟨m-1, n-1⟩ (m+n-2 : ℕ) v.  Unfold coneC.
  unfold coneC at h
  linarith

end OrigamiCone
