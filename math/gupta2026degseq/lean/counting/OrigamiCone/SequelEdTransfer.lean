import OrigamiCone.SequelEd
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Data.Matrix.Basic

namespace OrigamiCone.Sequel

def PathColouring (m : ℕ) : Type := Fin m → ZMod 3

instance {m : ℕ} : Fintype (PathColouring m) := Pi.instFintype

instance {m : ℕ} : DecidableEq (PathColouring m) := by
  unfold PathColouring; infer_instance

def IsPathProperColouring {m : ℕ} (c : PathColouring m) : Prop :=
  ∀ i : Fin m, ∀ h : i.val + 1 < m, c i ≠ c ⟨i.val + 1, h⟩

instance {m : ℕ} (c : PathColouring m) : Decidable (IsPathProperColouring c) := by
  unfold IsPathProperColouring; infer_instance

def PathAdjacent {m : ℕ} (u v : PathColouring m) : Prop :=
  IsPathProperColouring u ∧ IsPathProperColouring v ∧ (∀ i, u i ≠ v i)

instance {m : ℕ} (u v : PathColouring m) : Decidable (PathAdjacent u v) := by
  unfold PathAdjacent; infer_instance

def TransferState (m : ℕ) : Type :=
  { p : PathColouring m × PathColouring m // PathAdjacent p.1 p.2 }

instance {m : ℕ} : Fintype (TransferState m) := by
  unfold TransferState; infer_instance

instance {m : ℕ} : DecidableEq (TransferState m) := by
  unfold TransferState; infer_instance

def IsColExtremum {m : ℕ} (u v w : PathColouring m) (i : Fin m) : Prop :=
  u i = w i ∧
    (∀ h : 0 < i.val, v ⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt h Nat.one_pos) i.isLt.le⟩ = u i) ∧
    (∀ h : i.val + 1 < m, v ⟨i.val + 1, h⟩ = u i)

instance {m : ℕ} (u v w : PathColouring m) (i : Fin m) :
    Decidable (IsColExtremum u v w i) := by
  unfold IsColExtremum; infer_instance

def columnExtremaCount {m : ℕ} (u v w : PathColouring m) : ℕ :=
  (Finset.univ.filter (fun i : Fin m => IsColExtremum u v w i)).card

theorem columnExtremaCount_le {m : ℕ} (u v w : PathColouring m) :
    columnExtremaCount u v w ≤ m := by
  have h_le : columnExtremaCount u v w ≤ (Finset.univ : Finset (Fin m)).card :=
    Finset.card_filter_le _ _
  rwa [Finset.card_univ, Fintype.card_fin] at h_le

noncomputable def transferMatrix (m : ℕ) :
    Matrix (TransferState m) (TransferState m) (Polynomial ℤ) :=
  fun s t =>
    if s.val.2 = t.val.1 then
      Polynomial.X ^ (columnExtremaCount s.val.1 s.val.2 t.val.2)
    else 0

theorem transferMatrix_apply_eq {m : ℕ} (s t : TransferState m)
    (h : s.val.2 = t.val.1) :
    transferMatrix m s t = Polynomial.X ^ (columnExtremaCount s.val.1 s.val.2 t.val.2) := by
  unfold transferMatrix
  rw [if_pos h]

theorem transferMatrix_apply_ne {m : ℕ} (s t : TransferState m)
    (h : s.val.2 ≠ t.val.1) :
    transferMatrix m s t = 0 := by
  unfold transferMatrix
  rw [if_neg h]

noncomputable def cnPoly (m n : ℕ) : Polynomial ℤ :=
  ∑ d ∈ Finset.range (m * n + 1), Polynomial.monomial d (Ed d m n : ℤ)

theorem Ed_eq_cnPoly_coeff (d m n : ℕ) :
    (Ed d m n : ℤ) = (cnPoly m n).coeff d := by
  unfold cnPoly
  rw [Polynomial.finset_sum_coeff]
  by_cases hd : d ≤ m * n
  ·
    have hd_mem : d ∈ Finset.range (m * n + 1) := by
      rw [Finset.mem_range]; omega
    rw [← Finset.sum_erase_add _ _ hd_mem]
    have h_others : ∑ k ∈ (Finset.range (m * n + 1)).erase d,
        (Polynomial.monomial k ((Ed k m n : ℤ))).coeff d = 0 := by
      refine Finset.sum_eq_zero ?_
      intro k hk
      rw [Finset.mem_erase] at hk
      rw [Polynomial.coeff_monomial]
      simp only [if_neg hk.1]
    rw [h_others, zero_add, Polynomial.coeff_monomial, if_pos rfl]
  ·
    push_neg at hd
    have h_ed_zero : Ed d m n = 0 :=
      Ed_gt_mn_eq_zero hd
    rw [h_ed_zero, Nat.cast_zero]
    refine (Finset.sum_eq_zero ?_).symm
    intro k hk
    rw [Finset.mem_range] at hk
    rw [Polynomial.coeff_monomial]
    have hkd : k ≠ d := by omega
    simp only [if_neg hkd]

theorem cnPoly_eq_sum_over_heights (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    cnPoly m n
      = ∑ h ∈ (CanonicalHeights_finite hm hn).toFinset,
          Polynomial.monomial (numExtrema h) (1 : ℤ) := by
  unfold cnPoly

  have h_maps : ∀ h ∈ (CanonicalHeights_finite hm hn).toFinset,
      numExtrema h ∈ Finset.range (m * n + 1) := by
    intro h _
    rw [Finset.mem_range]
    exact Nat.lt_succ_of_le (numExtrema_le_mn h)

  rw [← Finset.sum_fiberwise_of_maps_to h_maps
        (fun h => Polynomial.monomial (numExtrema h) (1 : ℤ))]

  refine Finset.sum_congr rfl fun d _ => ?_

  have h_inner_eq :
      (∑ h ∈ (CanonicalHeights_finite hm hn).toFinset with numExtrema h = d,
        Polynomial.monomial (numExtrema h) (1 : ℤ))
      = ∑ h ∈ (CanonicalHeights_finite hm hn).toFinset with numExtrema h = d,
          Polynomial.monomial d (1 : ℤ) := by
    refine Finset.sum_congr rfl fun h hh => ?_
    rw [Finset.mem_filter] at hh
    rw [hh.2]
  rw [h_inner_eq, Finset.sum_const]

  rw [Polynomial.smul_monomial]
  congr 1

  have h_ed_card :
      Ed d m n
        = ((CanonicalHeights_finite hm hn).toFinset.filter
            (fun h => numExtrema h = d)).card := by
    rw [Ed_eq_finset_card d hm hn]
    congr 1
    ext h
    simp only [Set.Finite.mem_toFinset, Finset.mem_filter, Set.mem_setOf_eq]
    tauto
  rw [← h_ed_card]
  simp

def hCol {m n : ℕ} (h : Cell m n → ℤ) (j : Fin n) : PathColouring m :=
  fun i => ((h (i, j) : ℤ) : ZMod 3)

private theorem adj_row {m n : ℕ} (i : ℕ) (hi : i < m)
    (h_succ : i + 1 < m) (j : Fin n) :
    adj ((⟨i, hi⟩, j) : Cell m n) (⟨i + 1, h_succ⟩, j) := by
  show gdist ((⟨i, hi⟩, j) : Cell m n) (⟨i + 1, h_succ⟩, j) = 1
  show (((((⟨i, hi⟩ : Fin m).val : ℤ) - ((⟨i + 1, h_succ⟩ : Fin m).val : ℤ)).natAbs
        + (((j.val : ℤ) - j.val).natAbs : ℕ)) : ℤ) = 1
  simp

private theorem adj_col {m n : ℕ} (i : Fin m) (j : ℕ) (hj : j < n)
    (h_succ : j + 1 < n) :
    adj ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j + 1, h_succ⟩) := by
  show gdist ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j + 1, h_succ⟩) = 1
  show ((((i.val : ℤ) - i.val).natAbs
          + (((⟨j, hj⟩ : Fin n).val : ℤ) - ((⟨j + 1, h_succ⟩ : Fin n).val : ℤ)).natAbs : ℕ) : ℤ) = 1
  simp

private theorem adj_row_up {m n : ℕ} (i : ℕ) (hi : i < m)
    (hi_lo : 0 < i) (j : Fin n) :
    adj ((⟨i, hi⟩, j) : Cell m n) (⟨i - 1, by omega⟩, j) := by
  show gdist ((⟨i, hi⟩, j) : Cell m n) (⟨i - 1, by omega⟩, j) = 1
  show ((((i : ℤ) - ((i - 1 : ℕ) : ℤ)).natAbs
          + (((j.val : ℤ) - j.val).natAbs) : ℕ) : ℤ) = 1
  rw [Nat.cast_sub (by omega : 1 ≤ i)]
  push_cast; simp

private theorem adj_col_left {m n : ℕ} (i : Fin m) (j : ℕ) (hj : j < n)
    (hj_lo : 0 < j) :
    adj ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) := by
  show gdist ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) = 1
  show ((((i.val : ℤ) - i.val).natAbs
          + (((j : ℤ) - ((j - 1 : ℕ) : ℤ)).natAbs) : ℕ) : ℤ) = 1
  rw [Nat.cast_sub (by omega : 1 ≤ j)]
  push_cast; simp

private theorem adj_neighbour_cases {m n : ℕ} (p u : Cell m n) (hadj : adj p u) :
    (∃ hi : p.1.val + 1 < m, u = (⟨p.1.val + 1, hi⟩, p.2)) ∨
    (∃ hi : 0 < p.1.val, u = (⟨p.1.val - 1, by omega⟩, p.2)) ∨
    (∃ hj : p.2.val + 1 < n, u = (p.1, ⟨p.2.val + 1, hj⟩)) ∨
    (∃ hj : 0 < p.2.val, u = (p.1, ⟨p.2.val - 1, by omega⟩)) := by
  have hdist : (((((p.1.val : ℤ) - u.1.val).natAbs
                    + ((p.2.val : ℤ) - u.2.val).natAbs : ℕ)) : ℤ) = 1 := hadj

  have h_split : (((p.1.val : ℤ) - u.1.val).natAbs = 1 ∧ ((p.2.val : ℤ) - u.2.val).natAbs = 0) ∨
                 (((p.1.val : ℤ) - u.1.val).natAbs = 0 ∧ ((p.2.val : ℤ) - u.2.val).natAbs = 1) := by
    have : ((p.1.val : ℤ) - u.1.val).natAbs + ((p.2.val : ℤ) - u.2.val).natAbs = 1 := by
      exact_mod_cast hdist
    omega
  rcases h_split with ⟨hrow_ne, hcol_eq⟩ | ⟨hrow_eq, hcol_ne⟩
  ·
    have hcol : p.2 = u.2 := by
      apply Fin.ext
      have : (p.2.val : ℤ) = u.2.val := by
        have : ((p.2.val : ℤ) - u.2.val).natAbs = 0 := hcol_eq
        omega
      exact_mod_cast this

    have h_row_pm : p.1.val = u.1.val + 1 ∨ u.1.val = p.1.val + 1 := by
      have : ((p.1.val : ℤ) - u.1.val).natAbs = 1 := hrow_ne
      omega
    rcases h_row_pm with hup | hdown
    ·
      right; left
      have hi_lo : 0 < p.1.val := by omega
      refine ⟨hi_lo, ?_⟩
      apply Prod.ext
      · show u.1 = ⟨p.1.val - 1, _⟩
        apply Fin.ext; simp; omega
      · exact hcol.symm
    ·
      left
      have hi_hi : p.1.val + 1 < m := by
        have := u.1.isLt
        omega
      refine ⟨hi_hi, ?_⟩
      apply Prod.ext
      · show u.1 = ⟨p.1.val + 1, _⟩
        apply Fin.ext; simp; omega
      · exact hcol.symm
  ·
    have hrow : p.1 = u.1 := by
      apply Fin.ext
      have : (p.1.val : ℤ) = u.1.val := by
        have : ((p.1.val : ℤ) - u.1.val).natAbs = 0 := hrow_eq
        omega
      exact_mod_cast this
    have h_col_pm : p.2.val = u.2.val + 1 ∨ u.2.val = p.2.val + 1 := by
      have : ((p.2.val : ℤ) - u.2.val).natAbs = 1 := hcol_ne
      omega
    rcases h_col_pm with hleft | hright
    · right; right; right
      have hj_lo : 0 < p.2.val := by omega
      refine ⟨hj_lo, ?_⟩
      apply Prod.ext
      · exact hrow.symm
      · show u.2 = ⟨p.2.val - 1, _⟩
        apply Fin.ext; simp; omega
    · right; right; left
      have hj_hi : p.2.val + 1 < n := by
        have := u.2.isLt
        omega
      refine ⟨hj_hi, ?_⟩
      apply Prod.ext
      · exact hrow.symm
      · show u.2 = ⟨p.2.val + 1, _⟩
        apply Fin.ext; simp; omega

private theorem ne_of_abs_diff_one {a b : ℤ} (h : |a - b| = 1) :
    (a : ZMod 3) ≠ (b : ZMod 3) := by

  rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp h with heq | heq
  ·
    intro habs
    have h_ab : a = b + 1 := by linarith
    rw [h_ab] at habs
    push_cast at habs

    have : (1 : ZMod 3) = 0 := by linear_combination habs
    exact absurd this (by decide)
  ·
    intro habs
    have h_ab : a = b - 1 := by linarith
    rw [h_ab] at habs
    push_cast at habs
    have : (1 : ZMod 3) = 0 := by linear_combination -habs
    exact absurd this (by decide)

theorem hCol_isPathProperColouring {m n : ℕ} {h : Cell m n → ℤ}
    (hh : IsHeight h) (j : Fin n) :
    IsPathProperColouring (hCol h j) := by
  intro i h_succ
  unfold hCol
  have h_adj : adj ((i, j) : Cell m n) (⟨i.val + 1, h_succ⟩, j) :=
    adj_row i.val i.isLt h_succ j
  have h_diff : |h (i, j) - h (⟨i.val + 1, h_succ⟩, j)| = 1 :=
    hh _ _ h_adj

  have hi_eta : (i, j) = ((⟨i.val, i.isLt⟩ : Fin m), j) := by
    apply Prod.ext <;> simp
  rw [hi_eta] at h_diff
  exact ne_of_abs_diff_one h_diff

theorem hCol_pathAdjacent {m n : ℕ} {h : Cell m n → ℤ}
    (hh : IsHeight h) {j : ℕ} (hj : j < n) (h_succ : j + 1 < n) :
    PathAdjacent (hCol h ⟨j, hj⟩) (hCol h ⟨j + 1, h_succ⟩) := by
  refine ⟨hCol_isPathProperColouring hh _, hCol_isPathProperColouring hh _, ?_⟩
  intro i
  unfold hCol
  have h_adj : adj ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j + 1, h_succ⟩) :=
    adj_col i j hj h_succ
  exact ne_of_abs_diff_one (hh _ _ h_adj)

theorem IsColExtremum_of_IsStrictLocalMax_interior
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n)
    (hmax : IsStrictLocalMax h (i, ⟨j, by omega⟩)) :
    IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i := by

  set c : ℤ := h (i, ⟨j, by omega⟩) with hc_def

  have h_adj_L : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) :=
    adj_col_left i j (by omega) hj_lo
  have h_L : h (i, ⟨j - 1, by omega⟩) = c - 1 := hmax _ h_adj_L

  have h_adj_R : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j + 1, hj_hi⟩) :=
    adj_col i j (by omega) hj_hi
  have h_R : h (i, ⟨j + 1, hj_hi⟩) = c - 1 := hmax _ h_adj_R
  refine ⟨?_, ?_, ?_⟩
  ·
    show (hCol h ⟨j - 1, by omega⟩) i = (hCol h ⟨j + 1, hj_hi⟩) i
    unfold hCol
    rw [h_L, h_R]
  ·
    intro hi_lo
    show (hCol h ⟨j, by omega⟩)
              ⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt hi_lo Nat.one_pos) i.isLt.le⟩
          = (hCol h ⟨j - 1, by omega⟩) i
    have h_adj_U : adj ((i, ⟨j, by omega⟩) : Cell m n)
                        (⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt hi_lo Nat.one_pos)
                          i.isLt.le⟩, ⟨j, by omega⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨j, by omega⟩
    have h_U : h (⟨i.val - 1, _⟩, ⟨j, by omega⟩) = c - 1 := hmax _ h_adj_U
    unfold hCol
    rw [h_U, h_L]
  ·
    intro hi_hi
    show (hCol h ⟨j, by omega⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨j - 1, by omega⟩) i
    have h_adj_D : adj ((i, ⟨j, by omega⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) :=
      adj_row i.val i.isLt hi_hi ⟨j, by omega⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c - 1 := hmax _ h_adj_D
    unfold hCol
    rw [h_D, h_L]

theorem IsColExtremum_of_IsStrictLocalMin_interior
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n)
    (hmin : IsStrictLocalMin h (i, ⟨j, by omega⟩)) :
    IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i := by

  set c : ℤ := h (i, ⟨j, by omega⟩) with hc_def
  have h_adj_L : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) :=
    adj_col_left i j (by omega) hj_lo
  have h_L : h (i, ⟨j - 1, by omega⟩) = c + 1 := hmin _ h_adj_L
  have h_adj_R : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j + 1, hj_hi⟩) :=
    adj_col i j (by omega) hj_hi
  have h_R : h (i, ⟨j + 1, hj_hi⟩) = c + 1 := hmin _ h_adj_R
  refine ⟨?_, ?_, ?_⟩
  · show (hCol h ⟨j - 1, by omega⟩) i = (hCol h ⟨j + 1, hj_hi⟩) i
    unfold hCol
    rw [h_L, h_R]
  · intro hi_lo
    show (hCol h ⟨j, by omega⟩)
              ⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt hi_lo Nat.one_pos) i.isLt.le⟩
          = (hCol h ⟨j - 1, by omega⟩) i
    have h_adj_U : adj ((i, ⟨j, by omega⟩) : Cell m n)
                        (⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt hi_lo Nat.one_pos)
                          i.isLt.le⟩, ⟨j, by omega⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨j, by omega⟩
    have h_U : h (⟨i.val - 1, _⟩, ⟨j, by omega⟩) = c + 1 := hmin _ h_adj_U
    unfold hCol
    rw [h_U, h_L]
  · intro hi_hi
    show (hCol h ⟨j, by omega⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨j - 1, by omega⟩) i
    have h_adj_D : adj ((i, ⟨j, by omega⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) :=
      adj_row i.val i.isLt hi_hi ⟨j, by omega⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c + 1 := hmin _ h_adj_D
    unfold hCol
    rw [h_D, h_L]

theorem IsColExtremum_of_IsStrictLocalExtremum_interior
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n)
    (hext : IsStrictLocalExtremum h (i, ⟨j, by omega⟩)) :
    IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i := by
  rcases hext with hmax | hmin
  · exact IsColExtremum_of_IsStrictLocalMax_interior hh i hj_lo hj_hi hmax
  · exact IsColExtremum_of_IsStrictLocalMin_interior hh i hj_lo hj_hi hmin

private theorem eq_of_pm1_and_mod_eq {c a b : ℤ}
    (ha : a = c - 1 ∨ a = c + 1) (hb : b = c - 1 ∨ b = c + 1)
    (hmod : (a : ZMod 3) = (b : ZMod 3)) : a = b := by
  rcases ha with rfl | rfl <;> rcases hb with rfl | rfl
  · rfl
  · exfalso; push_cast at hmod
    have h2 : (2 : ZMod 3) = 0 := by linear_combination -hmod
    exact absurd h2 (by decide)
  · exfalso; push_cast at hmod
    have h2 : (2 : ZMod 3) = 0 := by linear_combination hmod
    exact absurd h2 (by decide)
  · rfl

theorem IsStrictLocalExtremum_of_IsColExtremum_interior
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n)
    (hcol : IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i) :
    IsStrictLocalExtremum h (i, ⟨j, by omega⟩) := by
  set c : ℤ := h (i, ⟨j, by omega⟩) with hc_def

  have h_adj_L : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) :=
    adj_col_left i j (by omega) hj_lo
  have h_adj_R : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j + 1, hj_hi⟩) :=
    adj_col i j (by omega) hj_hi

  have h_L_pm : h (i, ⟨j - 1, by omega⟩) = c - 1 ∨ h (i, ⟨j - 1, by omega⟩) = c + 1 := by
    rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_L) with h1 | h1
    · left; linarith
    · right; linarith
  have h_R_pm : h (i, ⟨j + 1, hj_hi⟩) = c - 1 ∨ h (i, ⟨j + 1, hj_hi⟩) = c + 1 := by
    rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_R) with h1 | h1
    · left; linarith
    · right; linarith

  have h_LR_mod : (h (i, ⟨j - 1, by omega⟩) : ZMod 3) = (h (i, ⟨j + 1, hj_hi⟩) : ZMod 3) := by
    have := hcol.1
    unfold hCol at this
    exact this
  have h_LR_eq : h (i, ⟨j - 1, by omega⟩) = h (i, ⟨j + 1, hj_hi⟩) :=
    eq_of_pm1_and_mod_eq h_L_pm h_R_pm h_LR_mod

  have h_L_pm' : h (i, ⟨j - 1, by omega⟩) = c - 1 ∨ h (i, ⟨j - 1, by omega⟩) = c + 1 := h_L_pm

  rcases h_L_pm with hL | hL
  ·
    left
    intro u hu

    rcases adj_neighbour_cases ((i, ⟨j, by omega⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj2_hi, hu_eq⟩ | ⟨hj2_lo, hu_eq⟩
    ·
      have h_down_cond := hcol.2.2 hi_hi
      unfold hCol at h_down_cond

      have h_adj_D : adj ((i, ⟨j, by omega⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) :=
        adj_row i.val i.isLt hi_hi ⟨j, by omega⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = h (i, ⟨j - 1, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_L_pm' h_down_cond
      rw [hu_eq, h_D_eq, hL]
    ·
      have h_up_cond := hcol.2.1 hi_lo
      unfold hCol at h_up_cond
      have h_adj_U : adj ((i, ⟨j, by omega⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨j, by omega⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = h (i, ⟨j - 1, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_L_pm' h_up_cond
      rw [hu_eq, h_U_eq, hL]
    ·
      rw [hu_eq, ← h_LR_eq, hL]
    ·
      rw [hu_eq, hL]
  ·
    right
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨j, by omega⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj2_hi, hu_eq⟩ | ⟨hj2_lo, hu_eq⟩
    · have h_down_cond := hcol.2.2 hi_hi
      unfold hCol at h_down_cond
      have h_adj_D : adj ((i, ⟨j, by omega⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) :=
        adj_row i.val i.isLt hi_hi ⟨j, by omega⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = h (i, ⟨j - 1, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_L_pm' h_down_cond
      rw [hu_eq, h_D_eq, hL]
    · have h_up_cond := hcol.2.1 hi_lo
      unfold hCol at h_up_cond
      have h_adj_U : adj ((i, ⟨j, by omega⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨j, by omega⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = h (i, ⟨j - 1, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_L_pm' h_up_cond
      rw [hu_eq, h_U_eq, hL]
    · rw [hu_eq, ← h_LR_eq, hL]
    · rw [hu_eq, hL]

theorem IsStrictLocalExtremum_iff_IsColExtremum_interior
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n) :
    IsStrictLocalExtremum h (i, ⟨j, by omega⟩) ↔
    IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i :=
  ⟨IsColExtremum_of_IsStrictLocalExtremum_interior hh i hj_lo hj_hi,
   IsStrictLocalExtremum_of_IsColExtremum_interior hh i hj_lo hj_hi⟩

def IsBoundaryExtremum {m : ℕ} (own other : PathColouring m) (i : Fin m) : Prop :=
  (∀ _ : 0 < i.val, own ⟨i.val - 1, by omega⟩ = other i) ∧
  (∀ h : i.val + 1 < m, own ⟨i.val + 1, h⟩ = other i)

instance {m : ℕ} (own other : PathColouring m) (i : Fin m) :
    Decidable (IsBoundaryExtremum own other i) := by
  unfold IsBoundaryExtremum; infer_instance

theorem IsBoundaryExtremum_of_IsStrictLocalMax_left
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n)
    (hmax : IsStrictLocalMax h (i, ⟨0, hn⟩)) :
    IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i := by
  set c : ℤ := h (i, ⟨0, hn⟩) with hc_def

  have h_adj_R : adj ((i, ⟨0, hn⟩) : Cell m n) (i, ⟨1, hn_succ⟩) :=
    adj_col i 0 hn hn_succ
  have h_R : h (i, ⟨1, hn_succ⟩) = c - 1 := hmax _ h_adj_R
  refine ⟨?_, ?_⟩
  ·
    intro hi_lo
    have h_adj_U : adj ((i, ⟨0, hn⟩) : Cell m n)
                        (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨0, hn⟩
    have h_U : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c - 1 := hmax _ h_adj_U
    show (hCol h ⟨0, hn⟩) ⟨i.val - 1, by omega⟩ = (hCol h ⟨1, hn_succ⟩) i
    unfold hCol
    rw [h_U, h_R]
  ·
    intro hi_hi
    have h_adj_D : adj ((i, ⟨0, hn⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) :=
      adj_row i.val i.isLt hi_hi ⟨0, hn⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c - 1 := hmax _ h_adj_D
    show (hCol h ⟨0, hn⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨1, hn_succ⟩) i
    unfold hCol
    rw [h_D, h_R]

theorem IsBoundaryExtremum_of_IsStrictLocalMin_left
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n)
    (hmin : IsStrictLocalMin h (i, ⟨0, hn⟩)) :
    IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i := by
  set c : ℤ := h (i, ⟨0, hn⟩) with hc_def
  have h_adj_R : adj ((i, ⟨0, hn⟩) : Cell m n) (i, ⟨1, hn_succ⟩) :=
    adj_col i 0 hn hn_succ
  have h_R : h (i, ⟨1, hn_succ⟩) = c + 1 := hmin _ h_adj_R
  refine ⟨?_, ?_⟩
  · intro hi_lo
    have h_adj_U : adj ((i, ⟨0, hn⟩) : Cell m n)
                        (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨0, hn⟩
    have h_U : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c + 1 := hmin _ h_adj_U
    show (hCol h ⟨0, hn⟩) ⟨i.val - 1, by omega⟩ = (hCol h ⟨1, hn_succ⟩) i
    unfold hCol
    rw [h_U, h_R]
  · intro hi_hi
    have h_adj_D : adj ((i, ⟨0, hn⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) :=
      adj_row i.val i.isLt hi_hi ⟨0, hn⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c + 1 := hmin _ h_adj_D
    show (hCol h ⟨0, hn⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨1, hn_succ⟩) i
    unfold hCol
    rw [h_D, h_R]

theorem IsBoundaryExtremum_of_IsStrictLocalExtremum_left
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n)
    (hext : IsStrictLocalExtremum h (i, ⟨0, hn⟩)) :
    IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i := by
  rcases hext with hmax | hmin
  · exact IsBoundaryExtremum_of_IsStrictLocalMax_left hh i hn hn_succ hmax
  · exact IsBoundaryExtremum_of_IsStrictLocalMin_left hh i hn hn_succ hmin

theorem IsStrictLocalExtremum_of_IsBoundaryExtremum_left
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n)
    (hbdy : IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i) :
    IsStrictLocalExtremum h (i, ⟨0, hn⟩) := by
  set c : ℤ := h (i, ⟨0, hn⟩) with hc_def

  have h_adj_R : adj ((i, ⟨0, hn⟩) : Cell m n) (i, ⟨1, hn_succ⟩) :=
    adj_col i 0 hn hn_succ
  have h_R_pm : h (i, ⟨1, hn_succ⟩) = c - 1 ∨ h (i, ⟨1, hn_succ⟩) = c + 1 := by
    rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_R) with h1 | h1
    · left; linarith
    · right; linarith

  have h_R_pm' : h (i, ⟨1, hn_succ⟩) = c - 1 ∨ h (i, ⟨1, hn_succ⟩) = c + 1 := h_R_pm

  rcases h_R_pm with hR | hR
  ·
    left
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨0, hn⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
    ·
      have h_adj_D : adj ((i, ⟨0, hn⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) :=
        adj_row i.val i.isLt hi_hi ⟨0, hn⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_D := hbdy.2 hi_hi
      unfold hCol at h_bdy_D
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = h (i, ⟨1, hn_succ⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_R_pm' h_bdy_D
      rw [hu_eq, h_D_eq, hR]
    ·
      have h_adj_U : adj ((i, ⟨0, hn⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨0, hn⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_U := hbdy.1 hi_lo
      unfold hCol at h_bdy_U
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = h (i, ⟨1, hn_succ⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_R_pm' h_bdy_U
      rw [hu_eq, h_U_eq, hR]
    ·
      rw [hu_eq]
      have h_fin : (⟨(0 : ℕ) + 1, hj_hi⟩ : Fin n) = ⟨1, hn_succ⟩ := by
        apply Fin.ext; simp
      rw [h_fin]; exact hR
    ·
      exact absurd hj_lo (by simp)
  ·
    right
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨0, hn⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
    · have h_adj_D : adj ((i, ⟨0, hn⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) :=
        adj_row i.val i.isLt hi_hi ⟨0, hn⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_D := hbdy.2 hi_hi
      unfold hCol at h_bdy_D
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = h (i, ⟨1, hn_succ⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_R_pm' h_bdy_D
      rw [hu_eq, h_D_eq, hR]
    · have h_adj_U : adj ((i, ⟨0, hn⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨0, hn⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_U := hbdy.1 hi_lo
      unfold hCol at h_bdy_U
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = h (i, ⟨1, hn_succ⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_R_pm' h_bdy_U
      rw [hu_eq, h_U_eq, hR]
    · rw [hu_eq]
      have h_fin : (⟨(0 : ℕ) + 1, hj_hi⟩ : Fin n) = ⟨1, hn_succ⟩ := by
        apply Fin.ext; simp
      rw [h_fin]; exact hR
    · exact absurd hj_lo (by simp)

theorem IsStrictLocalExtremum_iff_IsBoundaryExtremum_left
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n) :
    IsStrictLocalExtremum h (i, ⟨0, hn⟩) ↔
    IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i :=
  ⟨IsBoundaryExtremum_of_IsStrictLocalExtremum_left hh i hn hn_succ,
   IsStrictLocalExtremum_of_IsBoundaryExtremum_left hh i hn hn_succ⟩

theorem IsBoundaryExtremum_of_IsStrictLocalMax_right
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n)
    (hmax : IsStrictLocalMax h (i, ⟨n - 1, by omega⟩)) :
    IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i := by
  set c : ℤ := h (i, ⟨n - 1, by omega⟩) with hc_def

  have h_adj_L : adj ((i, ⟨n - 1, by omega⟩) : Cell m n) (i, ⟨n - 2, by omega⟩) := by
    have := @adj_col_left m n i (n - 1) (by omega) (by omega : 0 < n - 1)

    have h_lt1 : n - 1 - 1 < n := by omega
    have h_lt2 : n - 2 < n := by omega
    have h_fin : (⟨n - 1 - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
      apply Fin.ext; simp; omega
    rw [h_fin] at this
    exact this
  have h_L : h (i, ⟨n - 2, by omega⟩) = c - 1 := hmax _ h_adj_L
  refine ⟨?_, ?_⟩
  ·
    intro hi_lo
    have h_adj_U : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                        (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨n - 1, by omega⟩
    have h_U : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c - 1 := hmax _ h_adj_U
    show (hCol h ⟨n - 1, by omega⟩) ⟨i.val - 1, by omega⟩
          = (hCol h ⟨n - 2, by omega⟩) i
    unfold hCol
    rw [h_U, h_L]
  ·
    intro hi_hi
    have h_adj_D : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) :=
      adj_row i.val i.isLt hi_hi ⟨n - 1, by omega⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c - 1 := hmax _ h_adj_D
    show (hCol h ⟨n - 1, by omega⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨n - 2, by omega⟩) i
    unfold hCol
    rw [h_D, h_L]

theorem IsBoundaryExtremum_of_IsStrictLocalMin_right
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n)
    (hmin : IsStrictLocalMin h (i, ⟨n - 1, by omega⟩)) :
    IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i := by
  set c : ℤ := h (i, ⟨n - 1, by omega⟩) with hc_def
  have h_adj_L : adj ((i, ⟨n - 1, by omega⟩) : Cell m n) (i, ⟨n - 2, by omega⟩) := by
    have := @adj_col_left m n i (n - 1) (by omega) (by omega : 0 < n - 1)
    have h_lt1 : n - 1 - 1 < n := by omega
    have h_lt2 : n - 2 < n := by omega
    have h_fin : (⟨n - 1 - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
      apply Fin.ext; simp; omega
    rw [h_fin] at this
    exact this
  have h_L : h (i, ⟨n - 2, by omega⟩) = c + 1 := hmin _ h_adj_L
  refine ⟨?_, ?_⟩
  · intro hi_lo
    have h_adj_U : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                        (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨n - 1, by omega⟩
    have h_U : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c + 1 := hmin _ h_adj_U
    show (hCol h ⟨n - 1, by omega⟩) ⟨i.val - 1, by omega⟩
          = (hCol h ⟨n - 2, by omega⟩) i
    unfold hCol
    rw [h_U, h_L]
  · intro hi_hi
    have h_adj_D : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) :=
      adj_row i.val i.isLt hi_hi ⟨n - 1, by omega⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c + 1 := hmin _ h_adj_D
    show (hCol h ⟨n - 1, by omega⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨n - 2, by omega⟩) i
    unfold hCol
    rw [h_D, h_L]

theorem IsBoundaryExtremum_of_IsStrictLocalExtremum_right
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n)
    (hext : IsStrictLocalExtremum h (i, ⟨n - 1, by omega⟩)) :
    IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i := by
  rcases hext with hmax | hmin
  · exact IsBoundaryExtremum_of_IsStrictLocalMax_right hh i hn_succ hmax
  · exact IsBoundaryExtremum_of_IsStrictLocalMin_right hh i hn_succ hmin

theorem IsStrictLocalExtremum_of_IsBoundaryExtremum_right
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n)
    (hbdy : IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i) :
    IsStrictLocalExtremum h (i, ⟨n - 1, by omega⟩) := by
  set c : ℤ := h (i, ⟨n - 1, by omega⟩) with hc_def

  have h_adj_L : adj ((i, ⟨n - 1, by omega⟩) : Cell m n) (i, ⟨n - 2, by omega⟩) := by
    have := @adj_col_left m n i (n - 1) (by omega) (by omega : 0 < n - 1)
    have h_lt1 : n - 1 - 1 < n := by omega
    have h_lt2 : n - 2 < n := by omega
    have h_fin : (⟨n - 1 - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
      apply Fin.ext; simp; omega
    rw [h_fin] at this
    exact this
  have h_L_pm : h (i, ⟨n - 2, by omega⟩) = c - 1 ∨ h (i, ⟨n - 2, by omega⟩) = c + 1 := by
    rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_L) with h1 | h1
    · left; linarith
    · right; linarith
  have h_L_pm' : h (i, ⟨n - 2, by omega⟩) = c - 1 ∨ h (i, ⟨n - 2, by omega⟩) = c + 1 := h_L_pm
  rcases h_L_pm with hL | hL
  · left
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨n - 1, by omega⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
    · have h_adj_D : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) :=
        adj_row i.val i.isLt hi_hi ⟨n - 1, by omega⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_D := hbdy.2 hi_hi
      unfold hCol at h_bdy_D
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = h (i, ⟨n - 2, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_L_pm' h_bdy_D
      rw [hu_eq, h_D_eq, hL]
    · have h_adj_U : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨n - 1, by omega⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_U := hbdy.1 hi_lo
      unfold hCol at h_bdy_U
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = h (i, ⟨n - 2, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_L_pm' h_bdy_U
      rw [hu_eq, h_U_eq, hL]
    ·
      exfalso; simp at hj_hi; omega
    ·
      rw [hu_eq]
      have h_lt1 : (n - 1 : ℕ) - 1 < n := by omega
      have h_lt2 : n - 2 < n := by omega
      have h_fin : (⟨(n - 1 : ℕ) - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
        apply Fin.ext; simp; omega
      rw [h_fin]; exact hL
  · right
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨n - 1, by omega⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
    · have h_adj_D : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) :=
        adj_row i.val i.isLt hi_hi ⟨n - 1, by omega⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_D := hbdy.2 hi_hi
      unfold hCol at h_bdy_D
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = h (i, ⟨n - 2, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_L_pm' h_bdy_D
      rw [hu_eq, h_D_eq, hL]
    · have h_adj_U : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨n - 1, by omega⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_U := hbdy.1 hi_lo
      unfold hCol at h_bdy_U
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = h (i, ⟨n - 2, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_L_pm' h_bdy_U
      rw [hu_eq, h_U_eq, hL]
    · exfalso; simp at hj_hi; omega
    · rw [hu_eq]
      have h_lt1 : (n - 1 : ℕ) - 1 < n := by omega
      have h_lt2 : n - 2 < n := by omega
      have h_fin : (⟨(n - 1 : ℕ) - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
        apply Fin.ext; simp; omega
      rw [h_fin]; exact hL

theorem IsStrictLocalExtremum_iff_IsBoundaryExtremum_right
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n) :
    IsStrictLocalExtremum h (i, ⟨n - 1, by omega⟩) ↔
    IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i :=
  ⟨IsBoundaryExtremum_of_IsStrictLocalExtremum_right hh i hn_succ,
   IsStrictLocalExtremum_of_IsBoundaryExtremum_right hh i hn_succ⟩

noncomputable def heightToPath {m n : ℕ} (h : Cell m n → ℤ)
    (hh : IsHeight h) (hn : 2 ≤ n) (j : Fin (n - 1)) : TransferState m :=
  ⟨(hCol h ⟨j.val, by have := j.isLt; omega⟩,
    hCol h ⟨j.val + 1, by have := j.isLt; omega⟩),
   hCol_pathAdjacent hh (by have := j.isLt; omega)
                        (by have := j.isLt; omega)⟩

theorem heightToPath_fst {m n : ℕ} (h : Cell m n → ℤ)
    (hh : IsHeight h) (hn : 2 ≤ n) (j : Fin (n - 1)) :
    (heightToPath h hh hn j).val.1
      = hCol h ⟨j.val, by have := j.isLt; omega⟩ := rfl

theorem heightToPath_snd {m n : ℕ} (h : Cell m n → ℤ)
    (hh : IsHeight h) (hn : 2 ≤ n) (j : Fin (n - 1)) :
    (heightToPath h hh hn j).val.2
      = hCol h ⟨j.val + 1, by have := j.isLt; omega⟩ := rfl

theorem heightToPath_matches {m n : ℕ} (h : Cell m n → ℤ)
    (hh : IsHeight h) (hn : 2 ≤ n) (j : Fin (n - 1))
    (hj_succ : j.val + 1 < n - 1) :
    (heightToPath h hh hn j).val.2
      = (heightToPath h hh hn ⟨j.val + 1, hj_succ⟩).val.1 := rfl

def modSign (a b : ZMod 3) : ℤ :=
  if a + 1 = b then 1 else -1

theorem modSign_eq_pos_or_neg_one (a b : ZMod 3) :
    modSign a b = 1 ∨ modSign a b = -1 := by
  unfold modSign
  by_cases h : a + 1 = b
  · left; rw [if_pos h]
  · right; rw [if_neg h]

theorem modSign_abs (a b : ZMod 3) : |modSign a b| = 1 := by
  rcases modSign_eq_pos_or_neg_one a b with h | h <;> rw [h] <;> decide

theorem modSign_of_succ {a b : ZMod 3} (h : a + 1 = b) :
    modSign a b = 1 := by
  unfold modSign; rw [if_pos h]

theorem modSign_of_pred {a b : ZMod 3} (h : a + 1 ≠ b) :
    modSign a b = -1 := by
  unfold modSign; rw [if_neg h]

theorem modSign_cast_eq_diff {a b : ZMod 3} (hab : a ≠ b) :
    ((modSign a b : ℤ) : ZMod 3) = b - a := by
  revert hab
  revert a b
  decide

theorem modSign_symm {a b : ZMod 3} (hab : a ≠ b) :
    modSign b a = - modSign a b := by
  revert hab
  revert a b
  decide

theorem modSign_cycle_sum {a b d e : ZMod 3}
    (hab : a ≠ b) (hbe : b ≠ e) (hed : e ≠ d) (hda : d ≠ a) :
    modSign a b + modSign b e + modSign e d + modSign d a = 0 := by
  revert hab hbe hed hda
  revert a b d e
  decide

noncomputable def pathToHeight {m n : ℕ} (c : Fin n → PathColouring m)
    (p : Cell m n) : ℤ :=
  have hm : 0 < m := Nat.lt_of_le_of_lt (Nat.zero_le _) p.1.isLt

  (∑ k : Fin p.2.val,
      modSign (c ⟨k.val, by have := k.isLt; have := p.2.isLt; omega⟩ ⟨0, hm⟩)
              (c ⟨k.val + 1, by have := k.isLt; have := p.2.isLt; omega⟩ ⟨0, hm⟩))
  +

  (∑ r : Fin p.1.val,
      modSign (c p.2 ⟨r.val, by have := r.isLt; have := p.1.isLt; omega⟩)
              (c p.2 ⟨r.val + 1, by have := r.isLt; have := p.1.isLt; omega⟩))

theorem pathToHeight_origin {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (c : Fin n → PathColouring m) :
    pathToHeight c ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n) = 0 := by
  unfold pathToHeight
  simp

theorem pathToHeight_vstep {m n : ℕ} (c : Fin n → PathColouring m)
    (i : ℕ) (hi_succ : i + 1 < m) (j : Fin n) :
    pathToHeight c ((⟨i + 1, hi_succ⟩, j) : Cell m n)
      - pathToHeight c ((⟨i, by omega⟩, j) : Cell m n)
      = modSign (c j ⟨i, by omega⟩) (c j ⟨i + 1, hi_succ⟩) := by
  unfold pathToHeight
  simp only [Fin.sum_univ_castSucc, Fin.val_castSucc, Fin.val_last]
  ring

theorem pathToHeight_hstep {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r)
    (j : ℕ) (hj_succ : j + 1 < n)
    (i : ℕ) (hi : i < m) :
    pathToHeight c ((⟨i, hi⟩, ⟨j + 1, hj_succ⟩) : Cell m n)
      - pathToHeight c ((⟨i, hi⟩, ⟨j, by omega⟩) : Cell m n)
      = modSign (c ⟨j, by omega⟩ ⟨i, hi⟩)
                (c ⟨j + 1, hj_succ⟩ ⟨i, hi⟩) := by
  induction i with
  | zero =>

    unfold pathToHeight
    simp only [Fin.sum_univ_castSucc, Fin.val_castSucc, Fin.val_last]
    ring
  | succ i ih =>
    have hi_prev : i < m := by omega

    have vstep_jp := pathToHeight_vstep c i hi ⟨j + 1, hj_succ⟩

    have vstep_j := pathToHeight_vstep c i hi ⟨j, by omega⟩

    have hab : c ⟨j, by omega⟩ ⟨i, hi_prev⟩ ≠ c ⟨j + 1, hj_succ⟩ ⟨i, hi_prev⟩ :=
      hca j hj_succ ⟨i, hi_prev⟩
    have hbe : c ⟨j + 1, hj_succ⟩ ⟨i, hi_prev⟩ ≠ c ⟨j + 1, hj_succ⟩ ⟨i + 1, hi⟩ :=
      hcp ⟨j + 1, hj_succ⟩ ⟨i, hi_prev⟩ hi
    have hed : c ⟨j + 1, hj_succ⟩ ⟨i + 1, hi⟩ ≠ c ⟨j, by omega⟩ ⟨i + 1, hi⟩ :=
      (hca j hj_succ ⟨i + 1, hi⟩).symm
    have hda : c ⟨j, by omega⟩ ⟨i + 1, hi⟩ ≠ c ⟨j, by omega⟩ ⟨i, hi_prev⟩ :=
      (hcp ⟨j, by omega⟩ ⟨i, hi_prev⟩ hi).symm

    have cycle := modSign_cycle_sum hab hbe hed hda

    have hed_symm : modSign (c ⟨j + 1, hj_succ⟩ ⟨i + 1, hi⟩)
                            (c ⟨j, by omega⟩ ⟨i + 1, hi⟩)
                = - modSign (c ⟨j, by omega⟩ ⟨i + 1, hi⟩)
                            (c ⟨j + 1, hj_succ⟩ ⟨i + 1, hi⟩) :=
      modSign_symm hed.symm

    have hda_symm : modSign (c ⟨j, by omega⟩ ⟨i + 1, hi⟩)
                            (c ⟨j, by omega⟩ ⟨i, hi_prev⟩)
                = - modSign (c ⟨j, by omega⟩ ⟨i, hi_prev⟩)
                            (c ⟨j, by omega⟩ ⟨i + 1, hi⟩) :=
      modSign_symm hda.symm

    have ih' := ih hi_prev
    linarith [cycle, hed_symm, hda_symm, vstep_jp, vstep_j, ih']

theorem pathToHeight_isHeight {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r) :
    IsHeight (pathToHeight c) := by
  intro p q hadj
  rcases adj_neighbour_cases p q hadj with
    ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
  ·
    rw [hu_eq]
    have vstep := pathToHeight_vstep c p.1.val hi_hi p.2

    have h_eq : (⟨p.1.val, by have := p.1.isLt; omega⟩ : Fin m) = p.1 := by
      apply Fin.ext; rfl
    rw [h_eq] at vstep
    have : (p.1, p.2) = p := rfl
    rw [this] at vstep
    have : |pathToHeight c p - pathToHeight c (⟨p.1.val + 1, hi_hi⟩, p.2)|
         = |modSign (c p.2 p.1) (c p.2 ⟨p.1.val + 1, hi_hi⟩)| := by
      rw [show pathToHeight c p - pathToHeight c (⟨p.1.val + 1, hi_hi⟩, p.2)
            = -(pathToHeight c (⟨p.1.val + 1, hi_hi⟩, p.2) - pathToHeight c p) from by ring,
          vstep, abs_neg]
    rw [this]; exact modSign_abs _ _
  ·
    rw [hu_eq]

    have h_iprev : p.1.val - 1 + 1 = p.1.val := by omega
    have h_iprev_lt : p.1.val - 1 + 1 < m := by
      rw [h_iprev]; exact p.1.isLt
    have vstep := pathToHeight_vstep c (p.1.val - 1) h_iprev_lt p.2

    have h_fin1 : (⟨p.1.val - 1 + 1, h_iprev_lt⟩ : Fin m) = p.1 := by
      apply Fin.ext; simp; omega
    rw [h_fin1] at vstep
    have : (p.1, p.2) = p := rfl
    rw [this] at vstep
    have : |pathToHeight c p
            - pathToHeight c (⟨p.1.val - 1, by omega⟩, p.2)|
         = |modSign (c p.2 ⟨p.1.val - 1, by omega⟩) (c p.2 p.1)| := by
      rw [vstep]
    rw [this]; exact modSign_abs _ _
  ·
    rw [hu_eq]
    have hstep := pathToHeight_hstep c hcp hca p.2.val hj_hi p.1.val p.1.isLt

    have h_fin_col : (⟨p.2.val, by have := p.2.isLt; omega⟩ : Fin n) = p.2 := by
      apply Fin.ext; rfl
    have h_fin_row : (⟨p.1.val, p.1.isLt⟩ : Fin m) = p.1 := by
      apply Fin.ext; rfl
    rw [h_fin_col, h_fin_row] at hstep
    have : (p.1, p.2) = p := rfl
    rw [this] at hstep
    have : |pathToHeight c p - pathToHeight c (p.1, ⟨p.2.val + 1, hj_hi⟩)|
         = |modSign (c p.2 p.1) (c ⟨p.2.val + 1, hj_hi⟩ p.1)| := by
      rw [show pathToHeight c p - pathToHeight c (p.1, ⟨p.2.val + 1, hj_hi⟩)
            = -(pathToHeight c (p.1, ⟨p.2.val + 1, hj_hi⟩) - pathToHeight c p) from by ring,
          hstep, abs_neg]
    rw [this]; exact modSign_abs _ _
  ·
    rw [hu_eq]
    have h_jprev : p.2.val - 1 + 1 = p.2.val := by omega
    have h_jprev_lt : p.2.val - 1 + 1 < n := by
      rw [h_jprev]; exact p.2.isLt
    have hstep := pathToHeight_hstep c hcp hca (p.2.val - 1) h_jprev_lt p.1.val p.1.isLt
    have h_fin_col : (⟨p.2.val - 1 + 1, h_jprev_lt⟩ : Fin n) = p.2 := by
      apply Fin.ext; simp; omega
    have h_fin_row : (⟨p.1.val, p.1.isLt⟩ : Fin m) = p.1 := by
      apply Fin.ext; rfl
    rw [h_fin_col, h_fin_row] at hstep
    have : (p.1, p.2) = p := rfl
    rw [this] at hstep
    have : |pathToHeight c p
            - pathToHeight c (p.1, ⟨p.2.val - 1, by omega⟩)|
         = |modSign (c ⟨p.2.val - 1, by omega⟩ p.1) (c p.2 p.1)| := by
      rw [hstep]
    rw [this]; exact modSign_abs _ _

theorem hCol_pathToHeight_vstep {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (i : ℕ) (hi_succ : i + 1 < m) (j : Fin n) :
    ((pathToHeight c ((⟨i + 1, hi_succ⟩, j) : Cell m n) : ℤ) : ZMod 3)
      - ((pathToHeight c ((⟨i, by omega⟩, j) : Cell m n) : ℤ) : ZMod 3)
      = c j ⟨i + 1, hi_succ⟩ - c j ⟨i, by omega⟩ := by
  have vstep := pathToHeight_vstep c i hi_succ j
  have hab : c j ⟨i, by omega⟩ ≠ c j ⟨i + 1, hi_succ⟩ :=
    hcp j ⟨i, by omega⟩ hi_succ
  have h_cast_step :
      ((pathToHeight c ((⟨i + 1, hi_succ⟩, j) : Cell m n) : ℤ) : ZMod 3)
        - ((pathToHeight c ((⟨i, by omega⟩, j) : Cell m n) : ℤ) : ZMod 3)
        = ((modSign (c j ⟨i, by omega⟩) (c j ⟨i + 1, hi_succ⟩) : ℤ) : ZMod 3) := by
    rw [← Int.cast_sub, vstep]
  rw [h_cast_step, modSign_cast_eq_diff hab]

theorem hCol_pathToHeight_hstep {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r)
    (j : ℕ) (hj_succ : j + 1 < n)
    (i : ℕ) (hi : i < m) :
    ((pathToHeight c ((⟨i, hi⟩, ⟨j + 1, hj_succ⟩) : Cell m n) : ℤ) : ZMod 3)
      - ((pathToHeight c ((⟨i, hi⟩, ⟨j, by omega⟩) : Cell m n) : ℤ) : ZMod 3)
      = c ⟨j + 1, hj_succ⟩ ⟨i, hi⟩ - c ⟨j, by omega⟩ ⟨i, hi⟩ := by
  have hstep := pathToHeight_hstep c hcp hca j hj_succ i hi
  have hab : c ⟨j, by omega⟩ ⟨i, hi⟩ ≠ c ⟨j + 1, hj_succ⟩ ⟨i, hi⟩ :=
    hca j hj_succ ⟨i, hi⟩
  have h_cast_step :
      ((pathToHeight c ((⟨i, hi⟩, ⟨j + 1, hj_succ⟩) : Cell m n) : ℤ) : ZMod 3)
        - ((pathToHeight c ((⟨i, hi⟩, ⟨j, by omega⟩) : Cell m n) : ℤ) : ZMod 3)
        = ((modSign (c ⟨j, by omega⟩ ⟨i, hi⟩) (c ⟨j + 1, hj_succ⟩ ⟨i, hi⟩) : ℤ) : ZMod 3) := by
    rw [← Int.cast_sub, hstep]
  rw [h_cast_step, modSign_cast_eq_diff hab]

theorem hCol_pathToHeight_row0 {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r)
    (hm : 0 < m) (j : ℕ) (hj : j < n) :
    ((pathToHeight c ((⟨0, hm⟩, ⟨j, hj⟩) : Cell m n) : ℤ) : ZMod 3)
      = c ⟨j, hj⟩ ⟨0, hm⟩ - c ⟨0, by omega⟩ ⟨0, hm⟩ := by
  induction j with
  | zero =>

    have h_hn : (0 : ℕ) < n := by omega
    rw [pathToHeight_origin hm h_hn c]
    push_cast; ring
  | succ j ih =>
    have hj_prev : j < n := by omega
    have hstep := hCol_pathToHeight_hstep c hcp hca j hj (0 : ℕ) hm

    have ih' := ih hj_prev

    linear_combination hstep + ih'

theorem hCol_pathToHeight {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r)
    (i : ℕ) (hi : i < m) (j : ℕ) (hj : j < n) :
    ((pathToHeight c ((⟨i, hi⟩, ⟨j, hj⟩) : Cell m n) : ℤ) : ZMod 3)
      = c ⟨j, hj⟩ ⟨i, hi⟩ - c ⟨0, by omega⟩ ⟨0, by omega⟩ := by
  induction i with
  | zero =>

    exact hCol_pathToHeight_row0 c hcp hca hi j hj
  | succ i ih =>
    have hi_prev : i < m := by omega
    have vstep := hCol_pathToHeight_vstep c hcp i hi ⟨j, hj⟩

    have ih' := ih hi_prev

    linear_combination vstep + ih'

private theorem modSign_hCol_v {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : ℕ) (hi_succ : i + 1 < m) (j : Fin n) :
    modSign ((hCol h) j ⟨i, by omega⟩) ((hCol h) j ⟨i + 1, hi_succ⟩)
      = h ((⟨i + 1, hi_succ⟩, j) : Cell m n) - h ((⟨i, by omega⟩, j) : Cell m n) := by
  have h_adj : adj ((⟨i, by omega⟩, j) : Cell m n) (⟨i + 1, hi_succ⟩, j) :=
    adj_row i (by omega) hi_succ j
  have h_diff := hh _ _ h_adj

  rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp h_diff with h1 | h1
  ·

    unfold hCol
    rw [modSign_of_pred]
    · linarith
    ·
      intro hcast
      have h_eq : h (⟨i + 1, hi_succ⟩, j) = h (⟨i, by omega⟩, j) - 1 := by linarith
      rw [h_eq] at hcast
      push_cast at hcast

      have h2 : (2 : ZMod 3) = 0 := by linear_combination hcast
      exact absurd h2 (by decide)
  ·
    unfold hCol
    rw [modSign_of_succ]
    · linarith
    ·
      have h_eq : h (⟨i + 1, hi_succ⟩, j) = h (⟨i, by omega⟩, j) + 1 := by linarith
      rw [h_eq]; push_cast; ring

private theorem modSign_hCol_h {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (j : ℕ) (hj_succ : j + 1 < n) (i : Fin m) :
    modSign ((hCol h) ⟨j, by omega⟩ i) ((hCol h) ⟨j + 1, hj_succ⟩ i)
      = h ((i, ⟨j + 1, hj_succ⟩) : Cell m n) - h ((i, ⟨j, by omega⟩) : Cell m n) := by
  have h_adj : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j + 1, hj_succ⟩) :=
    adj_col i j (by omega) hj_succ
  have h_diff := hh _ _ h_adj
  rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp h_diff with h1 | h1
  · unfold hCol
    rw [modSign_of_pred]
    · linarith
    · intro hcast
      have h_eq : h (i, ⟨j + 1, hj_succ⟩) = h (i, ⟨j, by omega⟩) - 1 := by linarith
      rw [h_eq] at hcast
      push_cast at hcast
      have h2 : (2 : ZMod 3) = 0 := by linear_combination hcast
      exact absurd h2 (by decide)
  · unfold hCol
    rw [modSign_of_succ]
    · linarith
    · have h_eq : h (i, ⟨j + 1, hj_succ⟩) = h (i, ⟨j, by omega⟩) + 1 := by linarith
      rw [h_eq]; push_cast; ring

private theorem pathToHeight_hCol_row0 {m n : ℕ} {h : Cell m n → ℤ}
    (hh : IsHeight h) (hm : 0 < m) (j : ℕ) (hj : j < n) :
    pathToHeight (hCol h) ((⟨0, hm⟩, ⟨j, hj⟩) : Cell m n)
      = h ((⟨0, hm⟩, ⟨j, hj⟩) : Cell m n) - h ((⟨0, hm⟩, ⟨0, by omega⟩) : Cell m n) := by
  induction j with
  | zero =>
    have h_hn : (0 : ℕ) < n := by omega
    rw [pathToHeight_origin hm h_hn (hCol h)]; ring
  | succ j ih =>
    have hj_prev : j < n := by omega

    have hcp_h : ∀ k : Fin n, IsPathProperColouring ((hCol h) k) :=
      fun k => hCol_isPathProperColouring hh k
    have hca_h : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
        (hCol h) ⟨k, by omega⟩ r ≠ (hCol h) ⟨k + 1, hk⟩ r := fun k hk r => by
      have := hCol_pathAdjacent hh (by omega : k < n) hk
      exact this.2.2 r
    have hstep := pathToHeight_hstep (hCol h) hcp_h hca_h j hj (0 : ℕ) hm

    have modSign_match := modSign_hCol_h hh j hj ⟨0, hm⟩

    have ih' := ih hj_prev
    linarith [hstep, modSign_match, ih']

theorem pathToHeight_hCol {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : ℕ) (hi : i < m) (j : ℕ) (hj : j < n) :
    pathToHeight (hCol h) ((⟨i, hi⟩, ⟨j, hj⟩) : Cell m n)
      = h ((⟨i, hi⟩, ⟨j, hj⟩) : Cell m n)
        - h ((⟨0, by omega⟩, ⟨0, by omega⟩) : Cell m n) := by
  induction i with
  | zero =>
    exact pathToHeight_hCol_row0 hh hi j hj
  | succ i ih =>
    have hi_prev : i < m := by omega
    have vstep := pathToHeight_vstep (hCol h) i hi ⟨j, hj⟩
    have modSign_match := modSign_hCol_v hh i hi ⟨j, hj⟩
    have ih' := ih hi_prev
    linarith [vstep, modSign_match, ih']

theorem numExtrema_eq_sum_over_cols {m n : ℕ} (h : Cell m n → ℤ) :
    numExtrema h = ∑ j : Fin n,
        (Finset.univ.filter (fun i : Fin m => IsStrictLocalExtremum h (i, j))).card := by
  unfold numExtrema
  simp_rw [Finset.card_filter]
  rw [Fintype.sum_prod_type (f := fun (p : Fin m × Fin n) =>
    (if IsStrictLocalExtremum h p then (1 : ℕ) else 0))]
  exact Finset.sum_comm

theorem col_count_interior_eq {m n : ℕ}
    {h : Cell m n → ℤ} (hh : IsHeight h)
    (j : ℕ) (hj_lo : 0 < j) (hj_hi : j + 1 < n) :
    (Finset.univ.filter
        (fun i : Fin m => IsStrictLocalExtremum h (i, ⟨j, by omega⟩))).card
      = columnExtremaCount (hCol h ⟨j - 1, by omega⟩)
                           (hCol h ⟨j, by omega⟩)
                           (hCol h ⟨j + 1, hj_hi⟩) := by
  unfold columnExtremaCount
  congr 1
  apply Finset.filter_congr
  intro i _
  exact (IsStrictLocalExtremum_iff_IsColExtremum_interior hh i hj_lo hj_hi)

theorem col_count_left_boundary_eq {m n : ℕ}
    {h : Cell m n → ℤ} (hh : IsHeight h)
    (hn : 0 < n) (hn_succ : 1 < n) :
    (Finset.univ.filter
        (fun i : Fin m => IsStrictLocalExtremum h (i, ⟨0, hn⟩))).card
      = (Finset.univ.filter
          (fun i : Fin m => IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i)).card := by
  congr 1
  apply Finset.filter_congr
  intro i _
  exact (IsStrictLocalExtremum_iff_IsBoundaryExtremum_left hh i hn hn_succ)

theorem col_count_right_boundary_eq {m n : ℕ}
    {h : Cell m n → ℤ} (hh : IsHeight h)
    (hn_succ : 1 < n) :
    (Finset.univ.filter
        (fun i : Fin m => IsStrictLocalExtremum h (i, ⟨n - 1, by omega⟩))).card
      = (Finset.univ.filter
          (fun i : Fin m => IsBoundaryExtremum
              (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i)).card := by
  congr 1
  apply Finset.filter_congr
  intro i _
  exact (IsStrictLocalExtremum_iff_IsBoundaryExtremum_right hh i hn_succ)

noncomputable def leftBdyVec (m : ℕ) (s : TransferState m) : Polynomial ℤ :=
  Polynomial.monomial
    ((Finset.univ.filter
        (fun i : Fin m => IsBoundaryExtremum s.val.1 s.val.2 i)).card)
    (1 : ℤ)

noncomputable def rightBdyVec (m : ℕ) (s : TransferState m) : Polynomial ℤ :=
  Polynomial.monomial
    ((Finset.univ.filter
        (fun i : Fin m => IsBoundaryExtremum s.val.2 s.val.1 i)).card)
    (1 : ℤ)

theorem leftBdyVec_ne_zero (m : ℕ) (s : TransferState m) :
    leftBdyVec m s ≠ 0 := by
  unfold leftBdyVec
  simp [Polynomial.monomial_eq_zero_iff]

theorem rightBdyVec_ne_zero (m : ℕ) (s : TransferState m) :
    rightBdyVec m s ≠ 0 := by
  unfold rightBdyVec
  simp [Polynomial.monomial_eq_zero_iff]

def IsNormalizedLeft {m : ℕ} (s : TransferState m) : Prop :=
  ∃ h : 0 < m, s.val.1 ⟨0, h⟩ = 0

instance {m : ℕ} (s : TransferState m) : Decidable (IsNormalizedLeft s) := by
  unfold IsNormalizedLeft
  by_cases h : 0 < m
  · exact decidable_of_iff _ ⟨fun heq => ⟨h, heq⟩, fun ⟨_, heq⟩ => heq⟩
  · exact isFalse (fun ⟨h', _⟩ => h h')

noncomputable def leftBdyVecNorm (m : ℕ) (s : TransferState m) : Polynomial ℤ :=
  if IsNormalizedLeft s then leftBdyVec m s else 0

theorem leftBdyVecNorm_of_not_normalized {m : ℕ} {s : TransferState m}
    (h : ¬ IsNormalizedLeft s) : leftBdyVecNorm m s = 0 := by
  unfold leftBdyVecNorm; rw [if_neg h]

theorem leftBdyVecNorm_of_normalized {m : ℕ} {s : TransferState m}
    (h : IsNormalizedLeft s) : leftBdyVecNorm m s = leftBdyVec m s := by
  unfold leftBdyVecNorm; rw [if_pos h]

end OrigamiCone.Sequel
