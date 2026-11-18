import OrigamiCone.SequelEdFrozenBridge
import OrigamiCone.SequelEdFrozenForward

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m n : ℕ}

theorem frozenColumn_dL_const_adjacent (h : Cell m n → ℤ) (hh : IsHeight h)
    (j : Fin n) (hfrz : frozenColumn h j) (i : ℕ) (hi1 : i + 1 < m) :
    h (⟨i + 1, hi1⟩, ⟨j.val - 1, by
      obtain ⟨hj0, _, _⟩ := hfrz; omega⟩)
      - h (⟨i + 1, hi1⟩, j)
    = h (⟨i, by omega⟩, ⟨j.val - 1, by
      obtain ⟨hj0, _, _⟩ := hfrz; omega⟩)
      - h (⟨i, by omega⟩, j) := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  set jL : Fin n := ⟨j.val - 1, by omega⟩ with hjLdef
  set jR : Fin n := ⟨j.val + 1, hj1⟩ with hjRdef
  set v : ℕ → ℤ := fun k => h (⟨min k (m-1), by omega⟩, j) with hvdef
  set Lc : ℕ → ℤ := fun k => h (⟨min k (m-1), by omega⟩, jL) with hLdef
  set Rc : ℕ → ℤ := fun k => h (⟨min k (m-1), by omega⟩, jR) with hRdef
  have hvk : ∀ k (hk : k < m), v k = h (⟨k, hk⟩, j) := by
    intro k hk; have hmin : min k (m-1) = k := by omega
    simp only [hvdef, hmin]
  have hLk : ∀ k (hk : k < m), Lc k = h (⟨k, hk⟩, jL) := by
    intro k hk; have hmin : min k (m-1) = k := by omega
    simp only [hLdef, hmin]
  have hRk : ∀ k (hk : k < m), Rc k = h (⟨k, hk⟩, jR) := by
    intro k hk; have hmin : min k (m-1) = k := by omega
    simp only [hRdef, hmin]
  set e : ℕ → ℤ := fun k => v (k+1) - v k with hedef
  set dL : ℕ → ℤ := fun k => Lc k - v k with hdLdef
  set dR : ℕ → ℤ := fun k => Rc k - v k with hdRdef
  have hev : ∀ k (hk1 : k + 1 < m), e k = h (⟨k+1, hk1⟩, j) - h (⟨k, by omega⟩, j) := by
    intro k hk1; simp only [hedef]; rw [hvk (k+1) hk1, hvk k (by omega)]
  have hdLv : ∀ k (hk : k < m), dL k = h (⟨k, hk⟩, jL) - h (⟨k, hk⟩, j) := by
    intro k hk; simp only [hdLdef]; rw [hLk k hk, hvk k hk]
  have hdRv : ∀ k (hk : k < m), dR k = h (⟨k, hk⟩, jR) - h (⟨k, hk⟩, j) := by
    intro k hk; simp only [hdRdef]; rw [hRk k hk, hvk k hk]
  have he_pm : ∀ i, i < m - 1 → e i = 1 ∨ e i = -1 := by
    intro i hi
    rw [hev i (by omega)]
    have hadj : adj ((⟨i+1, by omega⟩, j) : Cell m n) (⟨i, by omega⟩, j) := by
      unfold adj gdist; simp only [Fin.val_mk]; omega
    have hvv := hh _ _ hadj
    rcases abs_cases (h (⟨i+1, by omega⟩, j) - h (⟨i, by omega⟩, j))
        with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hLwalk : ∀ i, i < m - 1 →
      e i + (dL (i+1) - dL i) = 1 ∨ e i + (dL (i+1) - dL i) = -1 := by
    intro i hi
    have hstep : e i + (dL (i+1) - dL i)
        = h (⟨i+1, by omega⟩, jL) - h (⟨i, by omega⟩, jL) := by
      rw [hev i (by omega), hdLv (i+1) (by omega), hdLv i (by omega)]; ring
    rw [hstep]
    have hadj : adj ((⟨i+1, by omega⟩, jL) : Cell m n) (⟨i, by omega⟩, jL) := by
      unfold adj gdist; simp only [hjLdef, Fin.val_mk]; omega
    have hvv := hh _ _ hadj
    rcases abs_cases (h (⟨i+1, by omega⟩, jL) - h (⟨i, by omega⟩, jL))
        with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hRwalk : ∀ i, i < m - 1 →
      e i + (dR (i+1) - dR i) = 1 ∨ e i + (dR (i+1) - dR i) = -1 := by
    intro i hi
    have hstep : e i + (dR (i+1) - dR i)
        = h (⟨i+1, by omega⟩, jR) - h (⟨i, by omega⟩, jR) := by
      rw [hev i (by omega), hdRv (i+1) (by omega), hdRv i (by omega)]; ring
    rw [hstep]
    have hadj : adj ((⟨i+1, by omega⟩, jR) : Cell m n) (⟨i, by omega⟩, jR) := by
      unfold adj gdist; simp only [hjRdef, Fin.val_mk]; omega
    have hvv := hh _ _ hadj
    rcases abs_cases (h (⟨i+1, by omega⟩, jR) - h (⟨i, by omega⟩, jR))
        with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hrainbow : ∀ i, i ≤ m - 1 → dL i + dR i = 0 := by
    intro i hi
    have hi' : i < m := by omega
    rw [hdLv i hi', hdRv i hi']
    have hs := hsym ⟨i, hi'⟩
    simp only [hjLdef, hjRdef] at hs ⊢
    omega
  have hconst := rainbow_dL_const he_pm hLwalk hRwalk hrainbow i (by omega)
  rw [hdLv (i+1) hi1, hdLv i (by omega)] at hconst
  simp only [hjLdef] at hconst ⊢
  linarith

theorem frozenColumn_dL_eq_row0 (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (i : Fin m) :
    h (i, ⟨j.val - 1, by obtain ⟨hj0, _, _⟩ := hfrz; omega⟩) - h (i, j)
    = h (⟨0, hm⟩, ⟨j.val - 1, by obtain ⟨hj0, _, _⟩ := hfrz; omega⟩)
        - h (⟨0, hm⟩, j) := by
  obtain ⟨k, hk⟩ := i
  induction k with
  | zero => rfl
  | succ k ih =>
    have hk1 : k < m := by omega
    have hstep := frozenColumn_dL_const_adjacent h hh j hfrz k hk
    have := ih hk1
    linarith

theorem frozenColumn_dL_const (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (i1 i2 : Fin m) :
    h (i1, ⟨j.val - 1, by obtain ⟨hj0, _, _⟩ := hfrz; omega⟩) - h (i1, j)
    = h (i2, ⟨j.val - 1, by obtain ⟨hj0, _, _⟩ := hfrz; omega⟩) - h (i2, j) := by
  rw [frozenColumn_dL_eq_row0 h hh hm j hfrz i1,
      frozenColumn_dL_eq_row0 h hh hm j hfrz i2]

theorem frozenColumn_slope_exists (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) :
    ∃ k : ℤ, (k = 1 ∨ k = -1) ∧
      ∀ i : Fin m, h (i, j) - h (i, ⟨j.val - 1, by
        obtain ⟨hj0, _, _⟩ := hfrz; omega⟩) = k := by
  obtain ⟨hj0, hj1, _⟩ := hfrz
  set jL : Fin n := ⟨j.val - 1, by omega⟩ with hjLdef
  set k : ℤ := h (⟨0, hm⟩, j) - h (⟨0, hm⟩, jL) with hkdef
  have hk_pm : k = 1 ∨ k = -1 := by
    have hadj : adj ((⟨0, hm⟩, j) : Cell m n) (⟨0, hm⟩, jL) := by
      unfold adj gdist; simp only [hjLdef, Fin.val_mk]; omega
    have hvv := hh _ _ hadj
    rcases abs_cases (h (⟨0, hm⟩, j) - h (⟨0, hm⟩, jL)) with ⟨he, _⟩ | ⟨he, _⟩ <;>
      first | (left; omega) | (right; omega)
  refine ⟨k, hk_pm, ?_⟩
  intro i
  have := frozenColumn_dL_eq_row0 h hh hm j ⟨hj0, hj1, ‹_›⟩ i
  simp only [hjLdef] at this ⊢
  linarith

end OrigamiCone.Sequel
