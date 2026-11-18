import Mathlib
import OrigamiCone.SequelEdActiveCol

namespace OrigamiCone.Sequel

open OrigamiCone

variable {m n : ℕ}

def frozenColumn (h : Cell m n → ℤ) (j : Fin n) : Prop :=
  ∃ (_ : 0 < j.val) (hj1 : j.val + 1 < n),
    ∀ i : Fin m,
      h (i, ⟨j.val - 1, by omega⟩) + h (i, ⟨j.val + 1, hj1⟩) = 2 * h (i, j)

theorem frozenColumn_not_active (h : Cell m n → ℤ)
    (j : Fin n) (hfrz : frozenColumn h j) :
    ¬ activeColumn h j := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  rintro ⟨i, hext⟩
  set jL : Fin n := ⟨j.val - 1, by omega⟩ with hjL
  set jR : Fin n := ⟨j.val + 1, hj1⟩ with hjR
  have hadjL : adj ((i, j) : Cell m n) (i, jL) := by
    unfold adj gdist; simp only [hjL]; omega
  have hadjR : adj ((i, j) : Cell m n) (i, jR) := by
    unfold adj gdist; simp only [hjR]; omega
  have hsum : (h (i, jL) - h (i, j)) + (h (i, jR) - h (i, j)) = 0 := by
    have hs := hsym i; omega
  rcases hext with hmax | hmin
  ·
    have hL := hmax (i, jL) hadjL
    have hR := hmax (i, jR) hadjR
    omega
  ·
    have hL := hmin (i, jL) hadjL
    have hR := hmin (i, jR) hadjR
    omega

theorem active_not_frozenColumn (h : Cell m n → ℤ) (j : Fin n)
    (hact : activeColumn h j) : ¬ frozenColumn h j :=
  fun hfrz => frozenColumn_not_active h j hfrz hact

end OrigamiCone.Sequel
