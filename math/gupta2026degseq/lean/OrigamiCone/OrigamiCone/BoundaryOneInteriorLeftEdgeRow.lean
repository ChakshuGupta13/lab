import OrigamiCone.BoundaryOneInteriorSideEdges
import OrigamiCone.BoundaryOneInteriorRow
import OrigamiCone.PmOneWalk
import OrigamiCone.Parity

/-!
# Case 2b's third max — sub-case `r ≥ rB` (left edge, downward `p_I`)

Row-symmetric analog of `BoundaryOneInteriorTopEdgeCol.lean`. For
`p_B = (rB, 0)` on the LEFT edge (`1 ≤ rB ≤ m - 2`) in the sub-case
where `p_I`'s row `r` is at least `rB` (p_I weakly below p_B). The last
ROW `m - 1` carries a strict local minimum of `cpe` at column `s`
(= p_I's column); `PmOneWalk` applied to the ±1 walk on that row
delivers a row-strict-max at some column `j₀ < s`. Step-inward-lowers
(row m-1 → m-2) lifts the row-max to grid `IsStrictLocalMax`.

## Cone-dominance calculation

At `(m - 1, s)`:
* `cone_B = gdist p_B (m-1, s) = (m-1-rB) + s`.
* `cone_I = δ + gdist p_I (m-1, s) = δ + (m-1-r)`.

Under `r ≥ rB`, `D = r - rB + s` and parity + right-active give
`cone_B - cone_I = r + s - rB - δ = D - δ ≥ 2`.

## Results

* `oneInterior_LeftEdge_lastRow_stepIn_lower`: cpe(m-2, j) < cpe(m-1, j)
  for any column `j`, under left-edge `p_B` + `p_I` interior.
* `oneInterior_LeftEdge_row_strict_min_right/left`: `(m - 1, s)` strict
  local min in the last row under parity + right-active + `r ≥ rB`.
* `oneInterior_LeftEdge_row_third_max_R`: existence of `j₀ : Fin n` with
  `j₀.val < p_I.2.val` at which `cpe` has a strict local max in row `m - 1`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Step-inward-lowers-cpe for left-edge `p_B` (last row).** -/
theorem oneInterior_LeftEdge_lastRow_stepIn_lower
    (hm : 3 ≤ m) (hn : 2 ≤ n) {rB : ℕ} (hrB : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) (j : Fin n) :
    cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          ((⟨m - 2, by omega⟩ : Fin m), j) <
    cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          ((⟨m - 1, by omega⟩ : Fin m), j) := by
  obtain ⟨hr_pos, hr_bd, _, _⟩ := h_I
  unfold cpe gdist
  dsimp only
  omega

/-- **`(m - 1, s)` strict local min in last row: right direction** under `r ≥ rB`.
Both cones INCREASE by 1 going from (m-1, s) to (m-1, s+1), so the +1 sign is
automatic and h_r_ge_rB / hact are unused (kept for API uniformity). -/
theorem oneInterior_LeftEdge_row_strict_min_right
    (hm : 3 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (_h_r_ge_rB : rB ≤ p_I.1.val)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (_hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨m - 1, by omega⟩ : Fin m), p_I.2) <
    cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨m - 1, by omega⟩ : Fin m),
         (⟨p_I.2.val + 1, by
          have := h_I.2.2.2
          omega⟩ : Fin n)) := by
  have := hrB_pos  -- API uniformity
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  have hisht := cpe_isHeight ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  have hadj : adj ((⟨m - 1, by omega⟩ : Fin m), p_I.2)
                  ((⟨m - 1, by omega⟩ : Fin m),
                   (⟨p_I.2.val + 1, by omega⟩ : Fin n)) := by
    unfold adj gdist; dsimp only; omega
  have h_height := hisht _ _ hadj
  unfold cpe gdist at *
  dsimp only at *
  omega

/-- **`(m - 1, s)` strict local min in last row: left direction** under `r ≥ rB`. -/
theorem oneInterior_LeftEdge_row_strict_min_left
    (hm : 3 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_r_ge_rB : rB ≤ p_I.1.val)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨m - 1, by omega⟩ : Fin m), p_I.2) <
    cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨m - 1, by omega⟩ : Fin m),
         (⟨p_I.2.val - 1, by
          have := p_I.2.isLt
          omega⟩ : Fin n)) := by
  have := hrB_pos  -- API uniformity
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  have hisht := cpe_isHeight ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  have hadj : adj ((⟨m - 1, by omega⟩ : Fin m), p_I.2)
                  ((⟨m - 1, by omega⟩ : Fin m),
                   (⟨p_I.2.val - 1, by have := p_I.2.isLt; omega⟩ : Fin n)) := by
    unfold adj gdist; dsimp only; omega
  have h_height := hisht _ _ hadj
  unfold cpe gdist at *
  dsimp only at *
  omega

/-- **Case 2b's third max (left-edge, sub-case `r ≥ rB`): row strict local
max exists to the left of `p_I`'s column in the last row.** -/
theorem oneInterior_LeftEdge_row_third_max_R
    (hm : 3 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_r_ge_rB : rB ≤ p_I.1.val)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ j₀ : Fin n, j₀.val < p_I.2.val ∧
      (∀ h_j0succ : j₀.val + 1 < n,
        cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
            ((⟨m - 1, by omega⟩ : Fin m), (⟨j₀.val + 1, h_j0succ⟩ : Fin n))
        < cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
              ((⟨m - 1, by omega⟩ : Fin m), j₀)) ∧
      (j₀.val = 0 ∨
        cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
            ((⟨m - 1, by omega⟩ : Fin m),
             (⟨j₀.val - 1, by have := j₀.isLt; omega⟩ : Fin n))
        < cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
              ((⟨m - 1, by omega⟩ : Fin m), j₀)) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  -- Row function guarded by `j < n`; equals cpe at row `m - 1`, column `j`.
  set rowFn : ℕ → ℤ := fun j =>
    if h : j < n then
      cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          ((⟨m - 1, by omega⟩ : Fin m), (⟨j, h⟩ : Fin n))
    else 0 with hrow
  have hisht := cpe_isHeight ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  -- STEP 1: rowFn is a ±1 walk on [0, n - 1].
  have hwalk : ∀ j, j < n - 1 → |rowFn (j + 1) - rowFn j| = 1 := by
    intro j hj
    have hj_lt : j < n := by omega
    have hj_succ_lt : j + 1 < n := by omega
    show |rowFn (j + 1) - rowFn j| = 1
    simp only [hrow, dif_pos hj_lt, dif_pos hj_succ_lt]
    have hadj : adj ((⟨m - 1, by omega⟩ : Fin m), (⟨j, hj_lt⟩ : Fin n))
                    ((⟨m - 1, by omega⟩ : Fin m),
                     (⟨j + 1, hj_succ_lt⟩ : Fin n)) := by
      unfold adj gdist; dsimp only; omega
    have := hisht _ _ hadj
    rw [abs_sub_comm]; exact this
  -- STEP 2: rowFn p_I.2.val < rowFn (p_I.2.val - 1).
  have hj_lt : rowFn p_I.2.val < rowFn (p_I.2.val - 1) := by
    have hp_lt_n : p_I.2.val < n := p_I.2.isLt
    have hpm1_lt_n : p_I.2.val - 1 < n := by omega
    show rowFn p_I.2.val < rowFn (p_I.2.val - 1)
    simp only [hrow, dif_pos hp_lt_n, dif_pos hpm1_lt_n]
    have h_strict :=
      oneInterior_LeftEdge_row_strict_min_left hm hn hrB_pos hrB_lt
        ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ δ h_r_ge_rB hparity hact
    have hpI2_eq : (⟨p_I.2.val, hp_lt_n⟩ : Fin n) = p_I.2 := Fin.eta p_I.2 hp_lt_n
    rw [hpI2_eq]
    exact h_strict
  -- STEP 3: apply PmOneWalk.
  obtain ⟨j₀, hj₀_lt, h_right_row, h_left_row⟩ :=
    pm1_walk_strictMax_before_strictMin (n - 1) rowFn hwalk
      p_I.2.val hs_pos (by have := p_I.2.isLt; omega) hj_lt
  have hj₀_lt_n : j₀ < n := by have := p_I.2.isLt; omega
  refine ⟨(⟨j₀, hj₀_lt_n⟩ : Fin n), hj₀_lt, ?_, ?_⟩
  · intro h_j0succ
    have h := h_right_row
    simp only [hrow, dif_pos hj₀_lt_n, dif_pos h_j0succ] at h
    exact h
  · rcases h_left_row with h | h
    · exact Or.inl h
    · right
      have hj₀m1_lt_n : j₀ - 1 < n := by omega
      simp only [hrow, dif_pos hj₀_lt_n, dif_pos hj₀m1_lt_n] at h
      exact h

end OrigamiCone
