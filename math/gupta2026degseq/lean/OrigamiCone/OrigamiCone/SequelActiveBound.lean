import OrigamiCone.SequelEd

/-!
# Substrate 3/5: Active-column count bound (`# active middle columns ≤ d`)

For a function `h : Cell m n → ℤ` (in particular, a height function), a
column `j : Fin n` is **active** if it carries at least one strict local
extremum of `h`.  This module proves the elementary cardinality bound

  `#(active columns) ≤ numExtrema h`,

and its refinement to middle columns

  `#(active middle columns) ≤ numExtrema h`.

The proof is one line: `activeColumns h` is the image of the extremum
finset under `Prod.snd`, so `Finset.card_image_le` gives the bound.  Each
extremum cell projects to its column, and distinct columns come from
distinct extrema (with multiplicity possibly collapsed).

## Downstream role

In the paper's `Lemma 8.5` (`lem:uniform`, "Uniform onset"), the count of
extremum-carrying middle columns of a `d`-extremum height function is
bounded by `d`.  Together with the boundary-column count `2` and the
number of frozen runs `≤ d + 1`, this gives the type-width bound
`w_C ≤ 2d + 3`, which is `m`-free.  `SequelContraction` will pin this
bound onto the contraction/extension bijection; `SequelUniformOnsetProof`
will pair it with `composition_count_as_poly` (substrate 1/5).

No `sorry`, no additional axiom.  Check with
`#print axioms OrigamiCone.Sequel.activeMiddleColumns_card_le_numExtrema`.
-/

namespace OrigamiCone.Sequel

open OrigamiCone

variable {m n : ℕ}

/-- The finset of strict-local-extremum cells of `h`. -/
def extremaFinset (h : Cell m n → ℤ) : Finset (Cell m n) :=
  Finset.univ.filter (fun v => IsStrictLocalExtremum h v)

/-- Sanity: `numExtrema h = (extremaFinset h).card` (definitional). -/
theorem numExtrema_eq_card (h : Cell m n → ℤ) :
    numExtrema h = (extremaFinset h).card := rfl

/-- The finset of **active columns** of `h`: columns of the grid carrying at
least one strict local extremum.  Defined as the image of `extremaFinset h`
under `Prod.snd`. -/
def activeColumns (h : Cell m n → ℤ) : Finset (Fin n) :=
  (extremaFinset h).image (·.2)

/-- Membership characterisation: `j` is an active column iff some cell
`(i, j)` in column `j` is a strict local extremum of `h`. -/
theorem mem_activeColumns_iff (h : Cell m n → ℤ) (j : Fin n) :
    j ∈ activeColumns h ↔ ∃ i : Fin m, IsStrictLocalExtremum h (i, j) := by
  unfold activeColumns extremaFinset
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨⟨i, k⟩, hv, hvj⟩
    simp only at hvj
    subst hvj
    exact ⟨i, hv⟩
  · rintro ⟨i, hi⟩
    exact ⟨(i, j), hi, rfl⟩

/-- **Active-column bound**: the number of active columns of `h` is at most
the total number of strict local extrema of `h`.  Each active column
receives at least one extremum cell under the column projection, so the
image has cardinality at most that of the source. -/
theorem activeColumns_card_le_numExtrema (h : Cell m n → ℤ) :
    (activeColumns h).card ≤ numExtrema h := by
  unfold activeColumns
  rw [numExtrema_eq_card]
  exact Finset.card_image_le

/-- Predicate: `j : Fin n` is a **middle** column (interior, not a boundary
column).  In 0-indexed notation, `0 < j.val` means `j ≠ 0` (not the left
boundary), and `j.val + 1 < n` means `j.val < n - 1` (not the right
boundary). -/
def IsMiddleColumn (j : Fin n) : Prop := 0 < j.val ∧ j.val + 1 < n

instance IsMiddleColumn.decidable (j : Fin n) : Decidable (IsMiddleColumn j) := by
  unfold IsMiddleColumn; infer_instance

/-- The finset of **active middle columns** of `h`: active columns that
are additionally middle (interior). -/
def activeMiddleColumns (h : Cell m n → ℤ) : Finset (Fin n) :=
  (activeColumns h).filter IsMiddleColumn

/-- **Active-middle-column bound**: refinement of
`activeColumns_card_le_numExtrema` restricted to middle columns.  For a
height function `h` with exactly `d = numExtrema h` extrema, at most `d`
middle columns are active — the paper's "at most `d` active middle
columns" claim in the proof of `Lemma 8.5`. -/
theorem activeMiddleColumns_card_le_numExtrema (h : Cell m n → ℤ) :
    (activeMiddleColumns h).card ≤ numExtrema h :=
  le_trans (Finset.card_filter_le _ _) (activeColumns_card_le_numExtrema h)

end OrigamiCone.Sequel
