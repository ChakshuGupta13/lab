import OrigamiCone.CpeSharedCoord

/-!
# Shared-coordinate apexes ⟹ strict maxima at row/column boundary
(Sub-4 of `thm:deg4count`)

When the two cone-pair apexes share a coordinate, the strict local
maxima of `cpe` are confined to the grid boundary in the orthogonal
direction.  This is the *kill* step for the CC|EE adjacent-corner case
of `thm:deg4count` (paper main.tex L656-666): the cone-pair envelope
is "strictly increasing down each column and has a single maximum row",
so the apex pair never produces a degree-4 vertex via maxima in the
interior.

The Lean statement is sharper than the paper's prose: **every** strict
local max of a shared-row cpe is at row 0 or row m-1 — interior rows
contain no strict maxima at all.  Symmetric for shared-column.

Proof: an interior cell (in the row-direction) has both same-column
neighbours `v ± row` in the grid.  By `cpe_shared_first_mono` applied
to whichever of those neighbours is farther from the shared row, that
neighbour has `cpe ≥ cpe v`, contradicting the strict-local-max
requirement that every neighbour have `cpe = cpe v − 1`.

Results:
* `cpe_sharedRow_strictMax_boundary` — strict local max ⟹ first coord ∈ {0, m-1}.
* `cpe_sharedCol_strictMax_boundary` — symmetric.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Shared-row strict-max localisation.**

If the cone-pair apexes share a row (`p₁.1.val = p₂.1.val`), then any
strict local maximum of `cpe p₁ p₂ δ` must lie in the **first or last
row** of the grid (`v.1.val = 0` or `v.1.val + 1 = m`).

Proof: assume `v` is a strict local max but `0 < v.1.val` and
`v.1.val + 1 < m`.  Then both row-neighbours of `v` in the same column
are in the grid.  At least one of them is farther from the shared row
than `v` itself (the one moving "away from the shared row"); by
`cpe_shared_first_mono`, that neighbour's `cpe` value is `≥ cpe v`.
But the strict-local-max condition forces `cpe u = cpe v − 1` for every
neighbour `u`, contradicting `≥`. -/
theorem cpe_sharedRow_strictMax_boundary {p₁ p₂ : Cell m n} {δ : ℤ}
    (h_row : p₁.1.val = p₂.1.val) {v : Cell m n}
    (h_max : IsStrictLocalMax (cpe p₁ p₂ δ) v) :
    v.1.val = 0 ∨ v.1.val + 1 = m := by
  by_contra h_int
  push_neg at h_int
  obtain ⟨h_ne0, h_succ⟩ := h_int
  have h_pos : 0 < v.1.val := Nat.pos_of_ne_zero h_ne0
  have h_lt : v.1.val + 1 < m := lt_of_le_of_ne (by have := v.1.isLt; omega) h_succ
  -- Choose the row-neighbour FARTHER from the shared row.
  -- Case-split on whether v's row is below or at-or-above the shared row.
  by_cases h_case : (p₁.1.val : ℤ) < v.1.val
  · -- v is strictly below the shared row.  The neighbour at v.1.val + 1
    -- is farther from the shared row.
    let u : Cell m n := (⟨v.1.val + 1, h_lt⟩, v.2)
    -- Adjacency: gdist v u = 1 via the row-step.
    have h_adj : adj v u := by
      unfold adj gdist
      show (((((v.1.val : ℤ) - (v.1.val + 1 : ℕ))).natAbs
              + ((v.2.val : ℤ) - v.2.val).natAbs : ℕ) : ℤ) = 1
      have h1 : ((v.1.val : ℤ) - (v.1.val + 1 : ℕ)).natAbs = 1 := by
        push_cast; omega
      have h2 : ((v.2.val : ℤ) - v.2.val).natAbs = 0 := by omega
      rw [h1, h2]; rfl
    have h_eq : cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1 := h_max u h_adj
    -- u is farther from the shared row than v: |p₁.1 - u.1| = |p₁.1 - (v.1+1)|
    -- = (v.1+1) - p₁.1 > v.1 - p₁.1 = |p₁.1 - v.1|.
    have h_dist : ((p₁.1.val : ℤ) - v.1.val).natAbs
                ≤ ((p₁.1.val : ℤ) - u.1.val).natAbs := by
      show ((p₁.1.val : ℤ) - v.1.val).natAbs
         ≤ ((p₁.1.val : ℤ) - (⟨v.1.val + 1, h_lt⟩ : Fin m).val).natAbs
      dsimp only
      omega
    -- By cpe_shared_first_mono with v and u (same column), cpe v ≤ cpe u.
    have h_col : v.2 = u.2 := rfl
    have h_mono : cpe p₁ p₂ δ v ≤ cpe p₁ p₂ δ u :=
      cpe_shared_first_mono h_row h_col h_dist
    omega
  · -- v.1.val ≤ p₁.1.val.  The neighbour at v.1.val - 1 (which exists since v.1.val > 0)
    -- is farther from the shared row.
    push_neg at h_case
    let u : Cell m n := (⟨v.1.val - 1, by omega⟩, v.2)
    have h_adj : adj v u := by
      unfold adj gdist
      show (((((v.1.val : ℤ) - (v.1.val - 1 : ℕ))).natAbs
              + ((v.2.val : ℤ) - v.2.val).natAbs : ℕ) : ℤ) = 1
      have h1 : ((v.1.val : ℤ) - (v.1.val - 1 : ℕ)).natAbs = 1 := by omega
      have h2 : ((v.2.val : ℤ) - v.2.val).natAbs = 0 := by omega
      rw [h1, h2]; rfl
    have h_eq : cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1 := h_max u h_adj
    -- u is farther from the shared row: |p₁.1 - u.1| = p₁.1 - (v.1-1) = p₁.1 - v.1 + 1.
    have h_dist : ((p₁.1.val : ℤ) - v.1.val).natAbs
                ≤ ((p₁.1.val : ℤ) - u.1.val).natAbs := by
      show ((p₁.1.val : ℤ) - v.1.val).natAbs
         ≤ ((p₁.1.val : ℤ) - (⟨v.1.val - 1, _⟩ : Fin m).val).natAbs
      dsimp only
      omega
    have h_col : v.2 = u.2 := rfl
    have h_mono : cpe p₁ p₂ δ v ≤ cpe p₁ p₂ δ u :=
      cpe_shared_first_mono h_row h_col h_dist
    omega

/-- **Shared-column strict-max localisation.**  Symmetric to
`cpe_sharedRow_strictMax_boundary`: shared-column apexes confine strict
maxima to the first or last column. -/
theorem cpe_sharedCol_strictMax_boundary {p₁ p₂ : Cell m n} {δ : ℤ}
    (h_col : p₁.2.val = p₂.2.val) {v : Cell m n}
    (h_max : IsStrictLocalMax (cpe p₁ p₂ δ) v) :
    v.2.val = 0 ∨ v.2.val + 1 = n := by
  by_contra h_int
  push_neg at h_int
  obtain ⟨h_ne0, h_succ⟩ := h_int
  have h_pos : 0 < v.2.val := Nat.pos_of_ne_zero h_ne0
  have h_lt : v.2.val + 1 < n := lt_of_le_of_ne (by have := v.2.isLt; omega) h_succ
  by_cases h_case : (p₁.2.val : ℤ) < v.2.val
  · let u : Cell m n := (v.1, ⟨v.2.val + 1, h_lt⟩)
    have h_adj : adj v u := by
      unfold adj gdist
      show (((((v.1.val : ℤ) - v.1.val)).natAbs
              + ((v.2.val : ℤ) - (v.2.val + 1 : ℕ)).natAbs : ℕ) : ℤ) = 1
      have h1 : ((v.1.val : ℤ) - v.1.val).natAbs = 0 := by omega
      have h2 : ((v.2.val : ℤ) - (v.2.val + 1 : ℕ)).natAbs = 1 := by
        push_cast; omega
      rw [h1, h2]; rfl
    have h_eq : cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1 := h_max u h_adj
    have h_dist : ((p₁.2.val : ℤ) - v.2.val).natAbs
                ≤ ((p₁.2.val : ℤ) - u.2.val).natAbs := by
      show ((p₁.2.val : ℤ) - v.2.val).natAbs
         ≤ ((p₁.2.val : ℤ) - (⟨v.2.val + 1, h_lt⟩ : Fin n).val).natAbs
      dsimp only
      omega
    have h_row : v.1 = u.1 := rfl
    have h_mono : cpe p₁ p₂ δ v ≤ cpe p₁ p₂ δ u :=
      cpe_shared_second_mono h_col h_row h_dist
    omega
  · push_neg at h_case
    let u : Cell m n := (v.1, ⟨v.2.val - 1, by omega⟩)
    have h_adj : adj v u := by
      unfold adj gdist
      show (((((v.1.val : ℤ) - v.1.val)).natAbs
              + ((v.2.val : ℤ) - (v.2.val - 1 : ℕ)).natAbs : ℕ) : ℤ) = 1
      have h1 : ((v.1.val : ℤ) - v.1.val).natAbs = 0 := by omega
      have h2 : ((v.2.val : ℤ) - (v.2.val - 1 : ℕ)).natAbs = 1 := by omega
      rw [h1, h2]; rfl
    have h_eq : cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1 := h_max u h_adj
    have h_dist : ((p₁.2.val : ℤ) - v.2.val).natAbs
                ≤ ((p₁.2.val : ℤ) - u.2.val).natAbs := by
      show ((p₁.2.val : ℤ) - v.2.val).natAbs
         ≤ ((p₁.2.val : ℤ) - (⟨v.2.val - 1, _⟩ : Fin n).val).natAbs
      dsimp only
      omega
    have h_row : v.1 = u.1 := rfl
    have h_mono : cpe p₁ p₂ δ v ≤ cpe p₁ p₂ δ u :=
      cpe_shared_second_mono h_col h_row h_dist
    omega

end OrigamiCone
