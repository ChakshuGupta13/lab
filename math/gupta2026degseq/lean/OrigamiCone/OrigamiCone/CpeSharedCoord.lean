import OrigamiCone.ConePair

/-!
# Cone-pair envelope: shared-coordinate monotonicity (Sub-2 of `thm:deg4count`)

When the two apexes `p₁, p₂` of a cone-pair envelope `cpe p₁ p₂ δ` share
their first (resp. second) coordinate, the envelope decomposes additively
as

  `cpe v = (row-distance from shared row) + (function of column only)`

(resp. the symmetric form for shared column).  This means `cpe` is
**monotone** in the absolute distance from the shared coordinate, when
restricted to cells in the same column (resp. row).

The paper uses this in:

* The CC|EE family analysis (main.tex L656-666): two corners on the same
  side share a row, so the cpe is "strictly increasing down each column
  and has a single maximum row" — a cone, never a degree-4 vertex.
* The CE|EE family analysis (L724-734): a corner and an edge cell on the
  same incident side share a coordinate, with similar monotonicity.

Results:
* `cpe_shared_first_mono` — shared first coord ⟹ same-column monotonicity.
* `cpe_shared_second_mono` — shared second coord ⟹ same-row monotonicity.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Shared-row monotonicity of `cpe`.**

If `p₁.1.val = p₂.1.val` (the two apexes share their row coordinate),
then for any two cells `v, w` in the same column with `w` farther from
the shared row than `v` (measured by absolute row-distance),
`cpe v ≤ cpe w`.

Proof: when `p₁.1.val = p₂.1.val`, both `gdist p₁ v` and `gdist p₂ v`
share the term `|p₁.1.val − v.1.val|.natAbs`, so
`cpe v = |p₁.1.val − v.1.val|.natAbs + min(...column terms only...)`.
The column terms agree at `v` and `w` (same column), so the difference
`cpe w − cpe v` equals the row-distance difference. -/
theorem cpe_shared_first_mono {p₁ p₂ : Cell m n} {δ : ℤ}
    (h_row : p₁.1.val = p₂.1.val)
    {v w : Cell m n} (h_col : v.2 = w.2)
    (h_dist : ((p₁.1.val : ℤ) - v.1.val).natAbs
            ≤ ((p₁.1.val : ℤ) - w.1.val).natAbs) :
    cpe p₁ p₂ δ v ≤ cpe p₁ p₂ δ w := by
  unfold cpe gdist
  -- Replace p₂.1.val with p₁.1.val throughout via h_row.
  have hp2 : (p₂.1.val : ℤ) = (p₁.1.val : ℤ) := by exact_mod_cast h_row.symm
  rw [hp2]
  -- Replace v.2.val with w.2.val (same column).
  have hcol : v.2.val = w.2.val := by rw [h_col]
  rw [hcol]
  -- Goal: min (Rv + C₁) (δ + Rv + C₂) ≤ min (Rw + C₁) (δ + Rw + C₂)
  -- where Rv = |p₁.1 - v.1|.natAbs, Rw = |p₁.1 - w.1|.natAbs (Rv ≤ Rw by h_dist),
  -- C_k = |p_k.2 - w.2|.natAbs (column-only terms now using w.2.val).
  -- Both branches of the min increase by (Rw - Rv) ≥ 0 going from v to w.
  omega

/-- **Shared-column monotonicity of `cpe`.**  Symmetric to
`cpe_shared_first_mono`. -/
theorem cpe_shared_second_mono {p₁ p₂ : Cell m n} {δ : ℤ}
    (h_col : p₁.2.val = p₂.2.val)
    {v w : Cell m n} (h_row : v.1 = w.1)
    (h_dist : ((p₁.2.val : ℤ) - v.2.val).natAbs
            ≤ ((p₁.2.val : ℤ) - w.2.val).natAbs) :
    cpe p₁ p₂ δ v ≤ cpe p₁ p₂ δ w := by
  unfold cpe gdist
  have hp2 : (p₂.2.val : ℤ) = (p₁.2.val : ℤ) := by exact_mod_cast h_col.symm
  rw [hp2]
  have hrow : v.1.val = w.1.val := by rw [h_row]
  rw [hrow]
  omega

end OrigamiCone
