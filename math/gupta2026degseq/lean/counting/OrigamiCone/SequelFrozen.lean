import Mathlib

namespace OrigamiCone.Sequel

def isExtremum (u v w : ℕ → ZMod 3) (m i : ℕ) : Prop :=
  u i = w i ∧ (0 < i → v (i - 1) = u i) ∧ (i + 1 < m → v (i + 1) = u i)

def IsFrozen (u v w : ℕ → ZMod 3) : Prop :=
  ∃ k : ZMod 3, k ≠ 0 ∧ (∀ i, u i = v i - k) ∧ (∀ i, w i = v i + k)

theorem frozen_imp_extremumFree (u v w : ℕ → ZMod 3) (m : ℕ)
    (hf : IsFrozen u v w) : ∀ i, ¬ isExtremum u v w m i := by
  obtain ⟨k, hk, hu, hw⟩ := hf
  intro i hext
  have heq : u i = w i := hext.1
  rw [hu i, hw i] at heq
  have hne : ∀ x y : ZMod 3, y ≠ 0 → x - y ≠ x + y := by decide
  exact hne (v i) k hk heq

theorem extremum_imp_aligned (u v w : ℕ → ZMod 3) (m i : ℕ)
    (hext : isExtremum u v w m i) : u i = w i :=
  hext.1

theorem rainbow_imp_const_slope (u v w : ℕ → ZMod 3) (i : ℕ)
    (hsv : v (i + 1) ≠ v i) (hu : u (i + 1) ≠ u i) (hw : w (i + 1) ≠ w i)
    (hrain : ∀ j, w j - v j = -(u j - v j)) :
    u (i + 1) - v (i + 1) = u i - v i := by
  set sv := v (i + 1) - v i with hsvdef
  set a := u i - v i with hadef
  set x := u (i + 1) - v (i + 1) with hxdef
  have hs : sv ≠ 0 := sub_ne_zero.mpr hsv
  have h1 : x ≠ a - sv := by
    intro h; apply hu
    have hux : u (i + 1) = x + v (i + 1) := by rw [hxdef]; ring
    rw [hux, h, hsvdef, hadef]; ring
  have h2 : x ≠ a + sv := by
    intro h; apply hw
    have hwi1 : w (i + 1) = -(u (i + 1) - v (i + 1)) + v (i + 1) := by
      rw [← hrain (i + 1)]; ring
    have hwi : w i = -(u i - v i) + v i := by rw [← hrain i]; ring
    rw [hwi1, hwi, ← hxdef, ← hadef, h, hsvdef]; ring
  have key : ∀ p q r : ZMod 3, r ≠ 0 → p ≠ q - r → p ≠ q + r → p = q := by decide
  exact key x a sv hs h1 h2

end OrigamiCone.Sequel
