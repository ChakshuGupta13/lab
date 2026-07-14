import OrigamiCone.BoundaryOneInterior
import OrigamiCone.PmOneWalk
import OrigamiCone.Parity

/-!
# Case 2a assembly: 2nd strict local max in the last column

Assembles the case 2a "there is a strict local max above `p_I`'s row in the
last column" argument of the paper's Boundary Lemma (`lem:boundary` case 2a):

  With `p_B = (0, 0)`, `p_I` interior at `(r, s)`, and the configuration
  active (`-D < δ < D` with `D = d(p_B, p_I)`) and parity-valid, there is a
  row `i' < r` at which the cone-pair envelope `cpe` attains a strict local
  MAXIMUM at cell `(i', n - 1)` in the grid.

Combined with `BoundaryOneInterior.oneInterior_TLcorner_opposite_max`
(the opposite corner is a strict local max) and the symmetric last-row
argument (out of scope here), this delivers `lem:boundary`'s three-maxima
conclusion.

## The three ingredients

1. **`cpe` is a height function** — under the parity condition
   (`(δ - d(p_B, p_I)) % 2 = 0`), `cpe = min (gdist p_B) (δ + gdist p_I)` is
   a height function on the grid (`Parity.parity_isHeight`).
2. **`(r, n-1)` is a strict local minimum in the last column.** Under
   activity and parity, cone `p_I` dominates at `(r, n-1)` since `cone_I`
   attains its column minimum there; the height-function property then
   fixes the sign of `cpe(r ± 1, n-1) - cpe(r, n-1)` to `+1` (the other
   choice would need `cone_B` to attain a strictly smaller value, which
   activity rules out).
3. **1D primitive + step-inward lift.** `PmOneWalk.pm1_walk_strictMax_
   before_strictMin` applied to `cpe` restricted to the column produces a
   column strict local max at some `i' < r`. Combining with
   `BoundaryOneInterior.oneInterior_TLcorner_lastCol_stepIn_lower` (the
   inward neighbour `(i', n - 2)` is strictly smaller) lifts the column
   strict local max to a grid strict local max.

Scope: this module builds the assembly for the WLOG `p_B = TL` sub-case's
2nd max. The symmetric 3rd max on the last row and the reflections for
`p_B` at other corners are separate work.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`cpe` is a height function under parity.** Direct wrapper of
`Parity.parity_isHeight` unfolding the `cpe` definition. -/
theorem cpe_isHeight (p₁ p₂ : Cell m n) (δ : ℤ)
    (hparity : (δ - gdist p₁ p₂) % 2 = 0) :
    IsHeight (cpe p₁ p₂ δ) := by
  unfold cpe
  exact parity_isHeight hparity

/-- **Cone-I is active (dominates) at `(r, n - 1)`** with `p_B = (0, 0)` and
`p_I` interior at `(r, s)`. This is the paper's "cone_I is least, hence
active" observation: cone_I attains its column minimum at row `r`, and
under activity `δ < D` this makes cone_I strictly smaller than cone_B at
`(r, n - 1)` (as the paper's proof requires). -/
private theorem coneI_dominates_at_pI_row_lastCol
    (hm : 2 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    δ + gdist p_I (p_I.1, (⟨n - 1, by omega⟩ : Fin n))
      ≤ gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
              (p_I.1, (⟨n - 1, by omega⟩ : Fin n)) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  unfold gdist at *
  dsimp only at *
  -- hact says δ < p_I.1.val + p_I.2.val (the gdist from (0,0) to p_I).
  -- Goal: δ + |p_I.2.val - (n-1)| ≤ p_I.1.val + |0 - (n-1)|.
  -- After natAbs unfolding: δ + (n-1-p_I.2.val) ≤ p_I.1.val + (n-1).
  -- I.e., δ ≤ p_I.1.val + p_I.2.val. True from hact (strict).
  omega

/-- **`(r, n - 1)` is strict local min in the last column: down direction.**
Under parity and right-side activity (`δ < D = d(p_B, p_I)`), the envelope
`cpe` is strictly larger at `(r - 1, n - 1)` than at `(r, n - 1)`. Proof: the
height-function property gives `cpe (r - 1, n - 1) - cpe (r, n - 1) ∈ {+1, -1}`;
activity + cone-I dominance at `(r, n - 1)` rules out the `-1` case (both `min`
components at `(r - 1, n - 1)` are `≥ cpe (r, n - 1)`), so it is `+1`. -/
theorem oneInterior_TLcorner_col_strict_min_down
    (hm : 2 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        (p_I.1, (⟨n - 1, by omega⟩ : Fin n)) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨p_I.1.val - 1, by
          have := h_I.1
          have := p_I.1.isLt
          omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  -- Height-function property: |cpe(A) - cpe(B)| = 1 at adjacent cells.
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  have hadj : adj (p_I.1, (⟨n - 1, by omega⟩ : Fin n))
                  ((⟨p_I.1.val - 1, by have := p_I.1.isLt; omega⟩ : Fin m),
                   (⟨n - 1, by omega⟩ : Fin n)) := by
    unfold adj gdist
    dsimp only
    omega
  have h_height := hisht _ _ hadj
  -- Unfold cpe to expose the `min` and gdist components.
  unfold cpe gdist at *
  dsimp only at *
  -- hact + activity gives δ ≤ p_I.1.val + p_I.2.val - 1 (integer strict).
  -- Combined with height-fn (abs = 1), omega closes.
  omega
theorem oneInterior_TLcorner_col_strict_min_up
    (hm : 2 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        (p_I.1, (⟨n - 1, by omega⟩ : Fin n)) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
        ((⟨p_I.1.val + 1, by
          have := h_I.2.1
          omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
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

/-- **Case 2a's second maximum: column strict local max exists above `p_I`'s
row.** With `p_B = (0, 0)`, `p_I` interior, parity + activity, there is a
row `i₀ : Fin m` with `i₀.val < p_I.1.val` at which `cpe` has a strict local
maximum ALONG the last column of the grid. Combined with `BoundaryOneInterior.
oneInterior_TLcorner_lastCol_stepIn_lower` (the inward neighbour
`(i₀, n - 2)` also has smaller `cpe`), this lifts to a strict local max in
the grid — the paper's second of three maxima in `lem:boundary` case 2a. -/
theorem oneInterior_TLcorner_col_second_max
    (hm : 2 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ i₀ : Fin m, i₀.val < p_I.1.val ∧
      (∀ h_i0succ : i₀.val + 1 < m,
        cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
            ((⟨i₀.val + 1, h_i0succ⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n))
        < cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
              (i₀, (⟨n - 1, by omega⟩ : Fin n))) ∧
      (i₀.val = 0 ∨
        cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
            ((⟨i₀.val - 1, by have := i₀.isLt; omega⟩ : Fin m),
             (⟨n - 1, by omega⟩ : Fin n))
        < cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
              (i₀, (⟨n - 1, by omega⟩ : Fin n))) := by
  obtain ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ := h_I
  -- Column function guarded by `i < m`; equals cpe at row `i`, column `n - 1`.
  set columnFn : ℕ → ℤ := fun i =>
    if h : i < m then
      cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          ((⟨i, h⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n))
    else 0 with hcol
  -- STEP 1: columnFn is a ±1 walk on [0, m - 1].
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
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
      oneInterior_TLcorner_col_strict_min_down hm hn
        ⟨hr_pos, hr_bd, hs_pos, hs_bd⟩ δ hparity hact
    -- Row Fin `p_I.1` is defeq to `⟨p_I.1.val, hp_lt_m⟩` via Fin.eta.
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
