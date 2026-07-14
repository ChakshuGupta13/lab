import OrigamiCone.BoundaryOneInteriorEdge
import OrigamiCone.BoundaryOneInteriorColumn
import OrigamiCone.PmOneWalk
import OrigamiCone.Parity

/-!
# Case 2b's third max — sub-case `s ≥ cB` (rightward `p_I`)

Analog of `BoundaryOneInteriorColumn` for the case 2b setup (top edge
`p_B`) in the sub-case where `p_I`'s column `s` is at least `cB` (i.e.,
`p_I` lies weakly to the right of `p_B`'s column). In this sub-case, the
last column `n - 1` carries a strict local minimum of `cpe` at row `r`
(= p_I's row), and PmOneWalk applied to the ±1 walk on that column
delivers a col-strict-max at some row `i₀ < r`. Step-inward-lowers
lifts the col-max to a grid `IsStrictLocalMax`.

## Cone-dominance calculation

At `(r, n - 1)`:
* `cone_B = gdist p_B (r, n-1) = r + (n-1-cB)`.
* `cone_I = δ + gdist p_I (r, n-1) = δ + (n-1-s)`.

Under `s ≥ cB`, `D = r + s - cB` and parity `(δ - D) % 2 = 0` + activity
`δ < D` give `δ ≤ D - 2`, so `cone_B - cone_I ≥ 2` at `(r, n-1)`.

## Results

* `oneInterior_TopEdge_lastCol_stepIn_lower`: cpe(i, n-2) < cpe(i, n-1)
  for any row `i` under top-edge `p_B` + `p_I` interior.
* `oneInterior_TopEdge_col_strict_min_down/up`: `(r, n - 1)` strict
  local min in column `n - 1` under parity + right-active + `s ≥ cB`.
* `oneInterior_TopEdge_col_third_max_R`: existence of `i₀ : Fin m`
  with `i₀.val < r` at which `cpe` has a strict local max in the last
  column.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Step-inward-lowers-cpe for top-edge `p_B` (last column).** -/
theorem oneInterior_TopEdge_lastCol_stepIn_lower
    (hm : 2 ≤ m) (hn : 3 ≤ n) {cB : ℕ} (hcB : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) (i : Fin m) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
          (i, (⟨n - 2, by omega⟩ : Fin n)) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
          (i, (⟨n - 1, by omega⟩ : Fin n)) := by
  obtain ⟨_, _, hs_pos, hs_bd⟩ := h_I
  unfold cpe gdist
  dsimp only
  omega

/-- **`(r, n - 1)` strict local min (down)** under `s ≥ cB`. -/
theorem oneInterior_TopEdge_col_strict_min_down
    (hm : 2 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_s_ge_cB : cB ≤ p_I.2.val)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
        (p_I.1, (⟨n - 1, by omega⟩ : Fin n)) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
        ((⟨p_I.1.val - 1, by
          have := h_I.1
          have := p_I.1.isLt
          omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  have := hcB_pos  -- API uniformity: case 2b requires 1 ≤ cB (paper's edge case)
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                             p_I δ hparity
  have hadj : adj (p_I.1, (⟨n - 1, by omega⟩ : Fin n))
                  ((⟨p_I.1.val - 1, by have := p_I.1.isLt; omega⟩ : Fin m),
                   (⟨n - 1, by omega⟩ : Fin n)) := by
    unfold adj gdist
    dsimp only
    omega
  have h_height := hisht _ _ hadj
  unfold cpe gdist at *
  dsimp only at *
  omega

/-- **`(r, n - 1)` strict local min (up)** under `s ≥ cB`. -/
theorem oneInterior_TopEdge_col_strict_min_up
    (hm : 2 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_s_ge_cB : cB ≤ p_I.2.val)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
        (p_I.1, (⟨n - 1, by omega⟩ : Fin n)) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
        ((⟨p_I.1.val + 1, by
          have := h_I.2.1
          omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  have := hcB_pos  -- API uniformity: case 2b requires 1 ≤ cB (paper's edge case)
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                             p_I δ hparity
  have hadj : adj (p_I.1, (⟨n - 1, by omega⟩ : Fin n))
                  ((⟨p_I.1.val + 1, by omega⟩ : Fin m),
                   (⟨n - 1, by omega⟩ : Fin n)) := by
    unfold adj gdist
    dsimp only
    omega
  have h_height := hisht _ _ hadj
  unfold cpe gdist at *
  dsimp only at *
  omega

/-- **Case 2b's third maximum (sub-case `s ≥ cB`): column strict local max
above `p_I`'s row in the last column.** Assembly of `oneInterior_TopEdge_
col_strict_min_down` with `PmOneWalk.pm1_walk_strictMax_before_strictMin`
on the ±1 walk `cpe(·, n - 1)`. Analog of `oneInterior_TLcorner_col_second
_max` for the top-edge sub-case. -/
theorem oneInterior_TopEdge_col_third_max_R
    (hm : 2 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_s_ge_cB : cB ≤ p_I.2.val)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    ∃ i₀ : Fin m, i₀.val < p_I.1.val ∧
      (∀ h_i0succ : i₀.val + 1 < m,
        cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
            ((⟨i₀.val + 1, h_i0succ⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n))
        < cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
              (i₀, (⟨n - 1, by omega⟩ : Fin n))) ∧
      (i₀.val = 0 ∨
        cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
            ((⟨i₀.val - 1, by have := i₀.isLt; omega⟩ : Fin m),
             (⟨n - 1, by omega⟩ : Fin n))
        < cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
              (i₀, (⟨n - 1, by omega⟩ : Fin n))) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  -- Column function guarded by `i < m`; equals cpe at row `i`, column `n - 1`.
  set columnFn : ℕ → ℤ := fun i =>
    if h : i < m then
      cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ
          ((⟨i, h⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n))
    else 0 with hcol
  -- STEP 1: columnFn is a ±1 walk on [0, m - 1].
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                             p_I δ hparity
  have hwalk : ∀ i, i < m - 1 → |columnFn (i + 1) - columnFn i| = 1 := by
    intro i hi
    have hi_lt : i < m := by omega
    have hi_succ_lt : i + 1 < m := by omega
    show |columnFn (i + 1) - columnFn i| = 1
    simp only [hcol, dif_pos hi_lt, dif_pos hi_succ_lt]
    have hadj : adj ((⟨i, hi_lt⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n))
                    ((⟨i + 1, hi_succ_lt⟩ : Fin m),
                     (⟨n - 1, by omega⟩ : Fin n)) := by
      unfold adj gdist; dsimp only; omega
    have := hisht _ _ hadj
    rw [abs_sub_comm]; exact this
  -- STEP 2: columnFn p_I.1.val < columnFn (p_I.1.val - 1).
  have hj_lt : columnFn p_I.1.val < columnFn (p_I.1.val - 1) := by
    have hp_lt_m : p_I.1.val < m := p_I.1.isLt
    have hpm1_lt_m : p_I.1.val - 1 < m := by omega
    show columnFn p_I.1.val < columnFn (p_I.1.val - 1)
    simp only [hcol, dif_pos hp_lt_m, dif_pos hpm1_lt_m]
    have h_strict :=
      oneInterior_TopEdge_col_strict_min_down hm hn hcB_pos hcB_lt
        ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ δ h_s_ge_cB hparity hact
    have hpI1_eq : (⟨p_I.1.val, hp_lt_m⟩ : Fin m) = p_I.1 := Fin.eta p_I.1 hp_lt_m
    rw [hpI1_eq]
    exact h_strict
  -- STEP 3: apply PmOneWalk.
  obtain ⟨i₀, hi₀_lt, h_right_col, h_left_col⟩ :=
    pm1_walk_strictMax_before_strictMin (m - 1) columnFn hwalk
      p_I.1.val hr_pos (by have := p_I.1.isLt; omega) hj_lt
  have hi₀_lt_m : i₀ < m := by have := p_I.1.isLt; omega
  -- STEP 4: repackage.
  refine ⟨(⟨i₀, hi₀_lt_m⟩ : Fin m), hi₀_lt, ?_, ?_⟩
  · intro h_i0succ
    have h := h_right_col
    simp only [hcol, dif_pos hi₀_lt_m, dif_pos h_i0succ] at h
    exact h
  · rcases h_left_col with h | h
    · exact Or.inl h
    · right
      have hi₀m1_lt_m : i₀ - 1 < m := by omega
      simp only [hcol, dif_pos hi₀_lt_m, dif_pos hi₀m1_lt_m] at h
      exact h

end OrigamiCone
