import OrigamiCone.BoundaryOneInterior
import OrigamiCone.BoundaryOneInteriorColumn
import OrigamiCone.BoundaryOneInteriorRow
import OrigamiCone.Parity

/-!
# Case 2a assembly: 3 strict local maxima in the grid

Lifts the "strict local max in a line" results (opposite corner + column
strict local max + row strict local max) to `IsStrictLocalMax` in the full
grid, via:
  cpe is IsHeight (Parity) + strict inequality at neighbour ⟹ neighbour is
  cpe(v) - 1

For each candidate max cell `v = (i₀, n - 1)` (column case) or
`v = (m - 1, j₀)` (row case), we prove `∀ u, adj v u → cpe u = cpe v - 1`
by case-splitting on the direction of adjacency and appealing to:
  * the in-line strict-min primitive (for lateral neighbours), or
  * the step-inward-lowers primitive (for inward neighbour).

## Results

* `case2a_TL_col_strictLocalMax`: for column 2nd max at `(i₀, n - 1)`.
* `case2a_TL_row_strictLocalMax`: for row 3rd max at `(m - 1, j₀)`.
* `case2a_TL_opposite_strictLocalMax`: opposite corner (m - 1, n - 1) —
  already proved as `oneInterior_TLcorner_opposite_max` in
  BoundaryOneInterior; re-exported here in the case-2a assembly namespace
  for uniformity.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- Bridge: cpe strict less-than at adjacent cells implies cpe = cpe - 1,
under parity (so cpe is IsHeight). -/
private lemma cpe_lt_adj_eq_pred
    {m n : ℕ} (p_B p_I : Cell m n) (δ : ℤ)
    (hparity : (δ - gdist p_B p_I) % 2 = 0)
    (v u : Cell m n) (hadj : adj v u)
    (hlt : cpe p_B p_I δ u < cpe p_B p_I δ v) :
    cpe p_B p_I δ u = cpe p_B p_I δ v - 1 := by
  have hisht := cpe_isHeight p_B p_I δ hparity
  have habs := hisht _ _ hadj
  -- |cpe v - cpe u| = 1 + strict-less ⟹ cpe v - cpe u = 1
  rw [abs_eq (by norm_num : (0:ℤ) ≤ 1)] at habs
  rcases habs with h | h
  · linarith
  · linarith

/-- **Case 2a's column 2nd max is a strict local max in the grid.** -/
theorem case2a_TL_col_strictLocalMax
    (hm : 2 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ i₀ : Fin m, i₀.val < p_I.1.val ∧
      IsStrictLocalMax
        (cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
        (i₀, (⟨n - 1, by omega⟩ : Fin n)) := by
  obtain ⟨i₀, hi₀_lt, h_right_col, h_left_col⟩ :=
    oneInterior_TLcorner_col_second_max hm hn h_I δ hparity hact
  refine ⟨i₀, hi₀_lt, ?_⟩
  intro u hadj
  -- Every adj neighbour u of (i₀, n-1) is in one of three positions.
  -- Unfold adj to a Manhattan-1 constraint and case-split on directions.
  set v : Cell m n := (i₀, (⟨n - 1, by omega⟩ : Fin n)) with hv
  -- Convert adj to `cpe u < cpe v` via case-analysis. IsHeight then gives the equality.
  -- Approach: enumerate u.1 and u.2 relative to i₀ and n-1.
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  have habs := hisht v u hadj
  -- Show cpe u < cpe v by case-split. IsHeight + strict-less ⟹ = pred.
  rw [abs_eq (by norm_num : (0:ℤ) ≤ 1)] at habs
  suffices h_lt :
      cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ u <
      cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ v by
    rcases habs with h | h
    · linarith
    · linarith
  -- adj v u = (|v.1 - u.1| + |v.2 - u.2| = 1). Case-split on u position.
  have hadj' : (((v.1.val : ℤ) - u.1.val).natAbs +
                ((v.2.val : ℤ) - u.2.val).natAbs : ℤ) = 1 := by
    have := hadj; unfold adj gdist at this; exact_mod_cast this
  have hu1 := u.1.isLt
  have hu2 := u.2.isLt
  -- Case: u.1 = i₀, u.2 = n-2 (inward)
  by_cases h1 : u.1.val = i₀.val
  · -- Row same; column must be n - 2 (n cannot be reached since u.2 < n)
    by_cases h2 : u.2.val = n - 2
    · -- u = (i₀, n - 2): use step-inward-lowers.
      have hu_eq : u = (i₀, (⟨n - 2, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext h1
        · exact Fin.ext h2
      rw [hu_eq]
      simpa [hv] using oneInterior_TLcorner_lastCol_stepIn_lower hm hn h_I δ i₀
    · -- u.1 = i₀, u.2 ≠ n - 2. Then either u.2 = n (impossible since u.2 < n)
      -- or u.2 = n - 1 (same as v.2, but then adj forces |0| + |0| = 0 ≠ 1).
      exfalso
      have : (v.2.val : ℤ) - u.2.val = 0 ∨
             ((v.2.val : ℤ) - u.2.val).natAbs = 1 := by
        -- from hadj' with h1
        have hrow : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by
          simp [hv, h1]
        omega
      -- v.2.val = n - 1. u.2 < n. So u.2 ∈ {n-2, n-1}. h2 excludes n-2. n-1 gives diff 0.
      have hv2 : v.2.val = n - 1 := by simp [hv]
      have hd0_or_1 : u.2.val = n - 1 ∨ u.2.val = n - 2 ∨ u.2.val + 2 ≤ n - 1 := by
        omega
      rcases hd0_or_1 with h | h | h
      · -- u.2 = n - 1 = v.2 ⇒ adj forces distance 0
        have : ((v.2.val : ℤ) - u.2.val).natAbs = 0 := by
          simp [hv, h]
        have hrow0 : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by simp [hv, h1]
        omega
      · exact h2 h
      · -- u.2 ≤ n - 3, so |v.2 - u.2| ≥ 2
        have : ((v.2.val : ℤ) - u.2.val).natAbs ≥ 2 := by
          simp only [hv2]; omega
        have hrow0 : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by simp [hv, h1]
        omega
  · -- u.1 ≠ i₀: column stays at n-1 (else adj forces distance > 1)
    have hu2_eq_v2 : u.2.val = n - 1 := by
      -- If not, both coords differ so natAbs sum ≥ 2
      by_contra hne
      have hv2 : v.2.val = n - 1 := by simp [hv]
      have hcol_ne : ((v.2.val : ℤ) - u.2.val).natAbs ≥ 1 := by
        simp only [hv2]; omega
      have hrow_ne : ((v.1.val : ℤ) - u.1.val).natAbs ≥ 1 := by
        have : v.1.val = i₀.val := by simp [hv]
        omega
      omega
    -- Now u = (u.1, n - 1) and |u.1 - i₀| = 1
    have hrow_diff : ((v.1.val : ℤ) - u.1.val).natAbs = 1 := by
      have hcol0 : ((v.2.val : ℤ) - u.2.val).natAbs = 0 := by
        have hv2 : v.2.val = n - 1 := by simp [hv]
        simp [hv2, hu2_eq_v2]
      omega
    have hi₀val : v.1.val = i₀.val := by simp [hv]
    have hu1_case : u.1.val = i₀.val + 1 ∨ u.1.val + 1 = i₀.val := by
      have : ((i₀.val : ℤ) - u.1.val).natAbs = 1 := by rw [← hi₀val]; exact hrow_diff
      omega
    rcases hu1_case with h | h
    · -- u.1 = i₀ + 1: down neighbour
      have h_i0succ : i₀.val + 1 < m := by
        have := u.1.isLt; omega
      have := h_right_col h_i0succ
      have hu_eq : u = (⟨i₀.val + 1, h_i0succ⟩, (⟨n - 1, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext h
        · exact Fin.ext hu2_eq_v2
      rw [hu_eq]
      simpa [hv] using this
    · -- u.1 + 1 = i₀: up neighbour (requires i₀ > 0)
      have hi₀pos : 0 < i₀.val := by omega
      have hi₀_eq : i₀.val - 1 = u.1.val := by omega
      have hu1_lt_m : u.1.val < m := u.1.isLt
      have hi₀m1_lt_m : i₀.val - 1 < m := by omega
      rcases h_left_col with h_z | h_up
      · -- i₀.val = 0 contradicts hi₀pos
        omega
      · -- Use up neighbour inequality
        have hu_eq : u = (⟨i₀.val - 1, hi₀m1_lt_m⟩, (⟨n - 1, by omega⟩ : Fin n)) := by
          apply Prod.ext
          · exact Fin.ext hi₀_eq.symm
          · exact Fin.ext hu2_eq_v2
        rw [hu_eq]
        simpa [hv] using h_up

/-- **Case 2a's row 3rd max is a strict local max in the grid.** Row twin
of `case2a_TL_col_strictLocalMax`. -/
theorem case2a_TL_row_strictLocalMax
    (hm : 3 ≤ m) (hn : 2 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ j₀ : Fin n, j₀.val < p_I.2.val ∧
      IsStrictLocalMax
        (cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
        ((⟨m - 1, by omega⟩ : Fin m), j₀) := by
  obtain ⟨j₀, hj₀_lt, h_right_row, h_left_row⟩ :=
    oneInterior_TLcorner_row_third_max hm hn h_I δ hparity hact
  refine ⟨j₀, hj₀_lt, ?_⟩
  intro u hadj
  set v : Cell m n := ((⟨m - 1, by omega⟩ : Fin m), j₀) with hv
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  have habs := hisht v u hadj
  rw [abs_eq (by norm_num : (0:ℤ) ≤ 1)] at habs
  suffices h_lt :
      cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ u <
      cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ v by
    rcases habs with h | h
    · linarith
    · linarith
  have hadj' : (((v.1.val : ℤ) - u.1.val).natAbs +
                ((v.2.val : ℤ) - u.2.val).natAbs : ℤ) = 1 := by
    have := hadj; unfold adj gdist at this; exact_mod_cast this
  have hu1 := u.1.isLt
  have hu2 := u.2.isLt
  -- Case: u.2 = j₀ (column same), u.1 = m-2 (inward)
  by_cases h2 : u.2.val = j₀.val
  · by_cases h1 : u.1.val = m - 2
    · have hu_eq : u = ((⟨m - 2, by omega⟩ : Fin m), j₀) := by
        apply Prod.ext
        · exact Fin.ext h1
        · exact Fin.ext h2
      rw [hu_eq]
      simpa [hv] using oneInterior_TLcorner_lastRow_stepIn_lower hm hn h_I δ j₀
    · exfalso
      have hv1 : v.1.val = m - 1 := by simp [hv]
      have hd0_or_1 : u.1.val = m - 1 ∨ u.1.val = m - 2 ∨ u.1.val + 2 ≤ m - 1 := by
        omega
      rcases hd0_or_1 with h | h | h
      · have : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by simp [hv, h]
        have hcol0 : ((v.2.val : ℤ) - u.2.val).natAbs = 0 := by simp [hv, h2]
        omega
      · exact h1 h
      · have : ((v.1.val : ℤ) - u.1.val).natAbs ≥ 2 := by
          simp only [hv1]; omega
        have hcol0 : ((v.2.val : ℤ) - u.2.val).natAbs = 0 := by simp [hv, h2]
        omega
  · -- u.2 ≠ j₀: row stays at m - 1
    have hu1_eq_v1 : u.1.val = m - 1 := by
      by_contra hne
      have hv1 : v.1.val = m - 1 := by simp [hv]
      have hrow_ne : ((v.1.val : ℤ) - u.1.val).natAbs ≥ 1 := by
        simp only [hv1]; omega
      have hcol_ne : ((v.2.val : ℤ) - u.2.val).natAbs ≥ 1 := by
        have : v.2.val = j₀.val := by simp [hv]
        omega
      omega
    have hcol_diff : ((v.2.val : ℤ) - u.2.val).natAbs = 1 := by
      have hrow0 : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by
        have hv1 : v.1.val = m - 1 := by simp [hv]
        simp [hv1, hu1_eq_v1]
      omega
    have hj₀val : v.2.val = j₀.val := by simp [hv]
    have hu2_case : u.2.val = j₀.val + 1 ∨ u.2.val + 1 = j₀.val := by
      have : ((j₀.val : ℤ) - u.2.val).natAbs = 1 := by rw [← hj₀val]; exact hcol_diff
      omega
    rcases hu2_case with h | h
    · have h_j0succ : j₀.val + 1 < n := by
        have := u.2.isLt; omega
      have := h_right_row h_j0succ
      have hu_eq : u = ((⟨m - 1, by omega⟩ : Fin m), ⟨j₀.val + 1, h_j0succ⟩) := by
        apply Prod.ext
        · exact Fin.ext hu1_eq_v1
        · exact Fin.ext h
      rw [hu_eq]
      simpa [hv] using this
    · have hj₀pos : 0 < j₀.val := by omega
      have hj₀_eq : j₀.val - 1 = u.2.val := by omega
      have hu2_lt_n : u.2.val < n := u.2.isLt
      have hj₀m1_lt_n : j₀.val - 1 < n := by omega
      rcases h_left_row with h_z | h_left
      · omega
      · have hu_eq : u = ((⟨m - 1, by omega⟩ : Fin m), ⟨j₀.val - 1, hj₀m1_lt_n⟩) := by
          apply Prod.ext
          · exact Fin.ext hu1_eq_v1
          · exact Fin.ext hj₀_eq.symm
        rw [hu_eq]
        simpa [hv] using h_left

/-- **Capstone: case 2a's three pairwise-distinct strict local maxima.**
With `p_B = (0, 0)`, `p_I` interior, parity + right-side activity, the
envelope `cpe p_B p_I δ` has three pairwise-distinct cells `v₁`, `v₂`, `v₃`
in the grid, each a `IsStrictLocalMax`:
  * `v₁ = (m-1, n-1)` (opposite corner)
  * `v₂ = (i₀, n-1)` for some `i₀ : Fin m` with `i₀.val < p_I.1.val`
  * `v₃ = (m-1, j₀)` for some `j₀ : Fin n` with `j₀.val < p_I.2.val`

This is the paper's `lem:boundary` case 2a conclusion (three-maxima
structural result), formalised at the strongest level: `IsStrictLocalMax`
in the grid graph, with distinctness explicit. `hm : 3 ≤ m` and `hn : 3 ≤ n`
(both strictly greater than the interior lemma's individual minima) are the
joint lower bounds required to house all three cells. -/
theorem case2a_TL_three_strictLocalMax
    (hm : 3 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ (i₀ : Fin m) (j₀ : Fin n),
      i₀.val < p_I.1.val ∧ j₀.val < p_I.2.val ∧
      IsStrictLocalMax
        (cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
        ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ∧
      IsStrictLocalMax
        (cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
        (i₀, (⟨n - 1, by omega⟩ : Fin n)) ∧
      IsStrictLocalMax
        (cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
        ((⟨m - 1, by omega⟩ : Fin m), j₀) ∧
      -- Pairwise distinctness.
      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ≠
        (i₀, ((⟨n - 1, by omega⟩ : Fin n))) ∧
      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ≠
        (((⟨m - 1, by omega⟩ : Fin m)), j₀) ∧
      (i₀, ((⟨n - 1, by omega⟩ : Fin n))) ≠
        (((⟨m - 1, by omega⟩ : Fin m)), j₀) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  obtain ⟨i₀, hi₀_lt, h_col_max⟩ :=
    case2a_TL_col_strictLocalMax (by omega) hn ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩
      δ hparity hact
  obtain ⟨j₀, hj₀_lt, h_row_max⟩ :=
    case2a_TL_row_strictLocalMax hm (by omega) ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩
      δ hparity hact
  have h_opp := oneInterior_TLcorner_opposite_max (by omega) (by omega)
                  ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ δ
  refine ⟨i₀, j₀, hi₀_lt, hj₀_lt, h_opp, h_col_max, h_row_max, ?_, ?_, ?_⟩
  · -- (m-1, n-1) ≠ (i₀, n-1): first coord differs since i₀ < r < m - 1.
    intro heq
    have := congrArg (fun c : Cell m n => c.1.val) heq
    dsimp at this
    -- (m - 1) = i₀.val, but i₀ < r < m - 1
    omega
  · -- (m-1, n-1) ≠ (m-1, j₀): second coord differs since j₀ < s < n - 1.
    intro heq
    have := congrArg (fun c : Cell m n => c.2.val) heq
    dsimp at this
    omega
  · -- (i₀, n-1) ≠ (m-1, j₀): both coords differ. First: i₀ < m - 1.
    intro heq
    have h1 := congrArg (fun c : Cell m n => c.1.val) heq
    dsimp at h1
    omega

/-- **Case 2a: three-element Finset of strict local maxima.** Uniform-shape
counterpart to `Boundary.bothInterior_exists_four_maxima`. With `p_B = TL
corner` and `p_I` interior, under parity + right-side activity, `cpe` has a
three-element Finset of strict local maxima. This is the ready-to-use form
for the counting bridge (a degree-4 vertex admits exactly two maxima, so
three-or-more rules the configuration out). -/
theorem case2a_TL_exists_three_maxima
    (hm : 3 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s,
        IsStrictLocalMax
          (cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ) c := by
  obtain ⟨i₀, j₀, hi₀_lt, hj₀_lt, h_opp, h_col, h_row, hne1, hne2, hne3⟩ :=
    case2a_TL_three_strictLocalMax hm hn h_I δ hparity hact
  refine ⟨{((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)),
           (i₀, (⟨n - 1, by omega⟩ : Fin n)),
           ((⟨m - 1, by omega⟩ : Fin m), j₀)}, ?_, ?_⟩
  · -- Cardinality 3 from pairwise distinctness.
    rw [Finset.card_insert_of_notMem (by simp [hne1, hne2]),
        Finset.card_insert_of_notMem (by simp [hne3])]
    simp
  · intro c hc
    simp only [Finset.mem_insert, Finset.mem_singleton] at hc
    rcases hc with rfl | rfl | rfl
    · exact h_opp
    · exact h_col
    · exact h_row

end OrigamiCone
