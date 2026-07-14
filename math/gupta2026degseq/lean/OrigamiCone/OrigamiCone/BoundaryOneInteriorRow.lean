import OrigamiCone.BoundaryOneInterior
import OrigamiCone.BoundaryOneInteriorColumn
import OrigamiCone.PmOneWalk
import OrigamiCone.Parity

/-!
# Case 2a assembly: 3rd strict local max in the last row

Row-symmetric twin of `BoundaryOneInteriorColumn.oneInterior_TLcorner_col_
second_max`. With `p_B = (0, 0)`, `p_I` interior at `(r, s)`, and the
configuration active + parity-valid, there is a column `j' < s` at which
`cpe` has a strict local max ALONG the last row `m - 1`. Combined with
`BoundaryOneInterior.oneInterior_TLcorner_lastRow_stepIn_lower` (the inward
neighbour `(m - 2, j')` is strictly smaller), this lifts to a grid strict
local max — the paper's 3rd of 3 maxima in `lem:boundary` case 2a.

Structure is identical to the column case, with rows/cols swapped:

1. `oneInterior_TLcorner_row_strict_min_right/left` — `(m-1, s)` is a
   strict local min in the last row.
2. `oneInterior_TLcorner_row_third_max` — assembly via `PmOneWalk`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`(m - 1, s)` is a strict local min in the last row: right direction.**
Row analog of `oneInterior_TLcorner_col_strict_min_down`. Note: unlike the
left direction, both cones strictly INCREASE from `(m-1, s)` to `(m-1, s+1)`,
so the `+1` sign of the height-function difference is forced without needing
activity. The `hact` hypothesis is kept for API uniformity. -/
theorem oneInterior_TLcorner_row_strict_min_right
    (hm : 3 ≤ m) (hn : 2 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (_hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨m - 1, by omega⟩ : Fin m), p_I.2) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨m - 1, by omega⟩ : Fin m),
         (⟨p_I.2.val + 1, by have := h_I.2.2.2; omega⟩ : Fin n)) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  have hadj : adj ((⟨m - 1, by omega⟩ : Fin m), p_I.2)
                  ((⟨m - 1, by omega⟩ : Fin m),
                   (⟨p_I.2.val + 1, by omega⟩ : Fin n)) := by
    unfold adj gdist; dsimp only; omega
  have h_height := hisht _ _ hadj
  unfold cpe gdist at *
  dsimp only at *
  omega

/-- **`(m - 1, s)` is a strict local min in the last row: left direction.** -/
theorem oneInterior_TLcorner_row_strict_min_left
    (hm : 3 ≤ m) (hn : 2 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨m - 1, by omega⟩ : Fin m), p_I.2) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨m - 1, by omega⟩ : Fin m),
         (⟨p_I.2.val - 1, by have := p_I.2.isLt; omega⟩ : Fin n)) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  have hadj : adj ((⟨m - 1, by omega⟩ : Fin m), p_I.2)
                  ((⟨m - 1, by omega⟩ : Fin m),
                   (⟨p_I.2.val - 1, by have := p_I.2.isLt; omega⟩ : Fin n)) := by
    unfold adj gdist; dsimp only; omega
  have h_height := hisht _ _ hadj
  unfold cpe gdist at *
  dsimp only at *
  omega

/-- **Case 2a's third maximum: row strict local max exists left of `p_I`'s
column.** With `p_B = (0, 0)`, `p_I` interior, parity + activity, there is a
column `j₀ : Fin n` with `j₀.val < p_I.2.val` at which `cpe` has a strict
local max ALONG the last row of the grid. Combined with `oneInterior_TLcorner
_lastRow_stepIn_lower`, this lifts to a strict local max in the grid —
the paper's third of three maxima in `lem:boundary` case 2a. -/
theorem oneInterior_TLcorner_row_third_max
    (hm : 3 ≤ m) (hn : 2 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ j₀ : Fin n, j₀.val < p_I.2.val ∧
      (∀ h_j0succ : j₀.val + 1 < n,
        cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
            ((⟨m - 1, by omega⟩ : Fin m), (⟨j₀.val + 1, h_j0succ⟩ : Fin n))
        < cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
              ((⟨m - 1, by omega⟩ : Fin m), j₀)) ∧
      (j₀.val = 0 ∨
        cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
            ((⟨m - 1, by omega⟩ : Fin m),
             (⟨j₀.val - 1, by have := j₀.isLt; omega⟩ : Fin n))
        < cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
              ((⟨m - 1, by omega⟩ : Fin m), j₀)) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  -- Row function guarded by `j < n`; equals cpe at row `m - 1`, column `j`.
  set rowFn : ℕ → ℤ := fun j =>
    if h : j < n then
      cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          ((⟨m - 1, by omega⟩ : Fin m), (⟨j, h⟩ : Fin n))
    else 0 with hrow
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
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
      oneInterior_TLcorner_row_strict_min_left hm hn
        ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ δ hparity hact
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
