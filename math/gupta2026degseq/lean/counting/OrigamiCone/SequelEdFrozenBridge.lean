import OrigamiCone.SequelEdFrozenForward
import OrigamiCone.SequelEdFrozenCol

namespace OrigamiCone.Sequel

open OrigamiCone

variable {m n : ℕ}

theorem adj_interior_cases (i : Fin m) (j : Fin n) (hj0 : 0 < j.val)
    (hj1 : j.val + 1 < n) (u : Cell m n) (hu : adj ((i, j) : Cell m n) u) :
    (∃ (_ : 0 < i.val), u = (⟨i.val - 1, by omega⟩, j))
    ∨ (∃ (hp : i.val + 1 < m), u = (⟨i.val + 1, hp⟩, j))
    ∨ (u = (i, ⟨j.val - 1, by omega⟩))
    ∨ (u = (i, ⟨j.val + 1, hj1⟩)) := by
  obtain ⟨⟨p, hp⟩, ⟨q, hq⟩⟩ := u
  unfold adj gdist at hu
  have hu' : ((i.val : ℤ) - p).natAbs + ((j.val : ℤ) - q).natAbs = 1 := by exact_mod_cast hu
  by_cases hcol : q = j.val
  ·
    subst hcol
    have hrow : ((i.val : ℤ) - p).natAbs = 1 := by simpa using hu'
    rcases (Int.natAbs_eq_iff).mp hrow with hpi | hpi
    · left; exact ⟨by omega, by ext <;> simp <;> omega⟩
    · right; left; exact ⟨by omega, by ext <;> simp <;> omega⟩
  ·
    have hcz : ((j.val : ℤ) - q).natAbs = 1 ∧ ((i.val : ℤ) - p).natAbs = 0 := by
      constructor
      · by_contra hc
        have hjq0 : ((j.val : ℤ) - q).natAbs = 0 := by omega
        exact hcol (by have := Int.natAbs_eq_zero.mp hjq0; omega)
      · omega
    have hpi : p = i.val := by have := Int.natAbs_eq_zero.mp hcz.2; omega
    have hqc : q = j.val - 1 ∨ q = j.val + 1 := by
      rcases (Int.natAbs_eq_iff).mp hcz.1 with h | h
      · left; omega
      · right; omega
    rcases hqc with hq1 | hq1
    · right; right; left; ext <;> simp <;> omega
    · right; right; right; ext <;> simp <;> omega

theorem inactive_imp_frozenColumn (h : Cell m n → ℤ) (hh : IsHeight h)
    (j : Fin n) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n) (hm : 0 < m)
    (hnact : ¬ activeColumn h j) : frozenColumn h j := by
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
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i+1, by omega⟩, j) - h (⟨i, by omega⟩, j)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hdL_pm : ∀ i, i ≤ m - 1 → dL i = 1 ∨ dL i = -1 := by
    intro i hi
    rw [hdLv i (by omega)]
    have hadj : adj ((⟨i, by omega⟩, j) : Cell m n) (⟨i, by omega⟩, jL) := by
      unfold adj gdist; simp only [hjLdef, Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i, by omega⟩, j) - h (⟨i, by omega⟩, jL)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hdR_pm : ∀ i, i ≤ m - 1 → dR i = 1 ∨ dR i = -1 := by
    intro i hi
    rw [hdRv i (by omega)]
    have hadj : adj ((⟨i, by omega⟩, j) : Cell m n) (⟨i, by omega⟩, jR) := by
      unfold adj gdist; simp only [hjRdef, Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i, by omega⟩, j) - h (⟨i, by omega⟩, jR)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega

  have hLwalk : ∀ i, i < m - 1 →
      e i + (dL (i+1) - dL i) = 1 ∨ e i + (dL (i+1) - dL i) = -1 := by
    intro i hi
    have hstep : e i + (dL (i+1) - dL i)
        = h (⟨i+1, by omega⟩, jL) - h (⟨i, by omega⟩, jL) := by
      rw [hev i (by omega), hdLv (i+1) (by omega), hdLv i (by omega)]; ring
    rw [hstep]
    have hadj : adj ((⟨i+1, by omega⟩, jL) : Cell m n) (⟨i, by omega⟩, jL) := by
      unfold adj gdist; simp only [hjLdef, Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i+1, by omega⟩, jL) - h (⟨i, by omega⟩, jL)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hRwalk : ∀ i, i < m - 1 →
      e i + (dR (i+1) - dR i) = 1 ∨ e i + (dR (i+1) - dR i) = -1 := by
    intro i hi
    have hstep : e i + (dR (i+1) - dR i)
        = h (⟨i+1, by omega⟩, jR) - h (⟨i, by omega⟩, jR) := by
      rw [hev i (by omega), hdRv (i+1) (by omega), hdRv i (by omega)]; ring
    rw [hstep]
    have hadj : adj ((⟨i+1, by omega⟩, jR) : Cell m n) (⟨i, by omega⟩, jR) := by
      unfold adj gdist; simp only [hjRdef, Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i+1, by omega⟩, jR) - h (⟨i, by omega⟩, jR)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega

  have hbridge : ∀ σ, (σ = 1 ∨ σ = -1) → ∀ i (hi' : i < m),
      alignedExt (m-1) e dL dR σ i → IsStrictLocalExtremum h (⟨i, hi'⟩, j) := by
    intro σ hσ i hi' hae
    obtain ⟨hae_up, hae_dn, hae_L, hae_R⟩ := hae

    have hLval : h (⟨i, hi'⟩, jL) = h (⟨i, hi'⟩, j) + σ := by
      have hd := hdLv i hi'; rw [hae_L] at hd; omega
    have hRval : h (⟨i, hi'⟩, jR) = h (⟨i, hi'⟩, j) + σ := by
      have hd := hdRv i hi'; rw [hae_R] at hd; omega
    have hup_val : ∀ (hpos : 0 < i) (P : i - 1 < m),
        h (⟨i - 1, P⟩, j) = h (⟨i, hi'⟩, j) + σ := by
      intro hpos P
      have hk1 : (i - 1) + 1 < m := by omega
      have hst : e (i - 1) = v ((i - 1) + 1) - v (i - 1) := by simp only [hedef]
      rw [hvk ((i - 1) + 1) hk1, hvk (i - 1) P, hae_up hpos] at hst
      have hc : h (⟨(i - 1) + 1, hk1⟩, j) = h (⟨i, hi'⟩, j) := by
        rw [show (⟨(i - 1) + 1, hk1⟩ : Fin m) = ⟨i, hi'⟩ from
          Fin.ext (show (i - 1) + 1 = i by omega)]
      rw [hc] at hst
      omega
    have hdn_val : ∀ (hlt : i < m - 1) (P : i + 1 < m),
        h (⟨i + 1, P⟩, j) = h (⟨i, hi'⟩, j) + σ := by
      intro hlt P
      have hst : e i = v (i + 1) - v i := by simp only [hedef]
      rw [hvk (i + 1) P, hvk i hi', hae_dn hlt] at hst
      omega
    rcases hσ with hσ1 | hσ1
    ·
      subst hσ1; right
      intro u hu
      rcases adj_interior_cases ⟨i, hi'⟩ j hj0 hj1 u hu with
        ⟨hup, rfl⟩ | ⟨hdn, rfl⟩ | rfl | rfl
      · exact hup_val hup _
      · exact hdn_val (by have h2 : i + 1 < m := hdn; omega) hdn
      · exact hLval
      · exact hRval
    ·
      subst hσ1; left
      intro u hu
      rcases adj_interior_cases ⟨i, hi'⟩ j hj0 hj1 u hu with
        ⟨hup, rfl⟩ | ⟨hdn, rfl⟩ | rfl | rfl
      · exact hup_val hup _
      · exact hdn_val (by have h2 : i + 1 < m := hdn; omega) hdn
      · exact hLval
      · exact hRval

  have hef : ∀ σ, (σ = 1 ∨ σ = -1) → ∀ i, i ≤ m - 1 →
      ¬ alignedExt (m-1) e dL dR σ i := by
    intro σ hσ i hi hae
    have hi' : i < m := by omega
    exact hnact ⟨⟨i, hi'⟩, hbridge σ hσ i hi' hae⟩

  have hrainbow := extremumFree_rainbow he_pm hdL_pm hdR_pm hLwalk hRwalk hef
  refine ⟨hj0, hj1, ?_⟩
  intro i
  have hi := i.isLt
  have hr := hrainbow i.val (by omega)
  rw [hdLv i.val hi, hdRv i.val hi] at hr
  have hiv : (⟨i.val, hi⟩ : Fin m) = i := Fin.ext rfl
  rw [hiv] at hr

  show h (i, jL) + h (i, jR) = 2 * h (i, j)
  omega

theorem frozenColumn_iff_inactive (h : Cell m n → ℤ) (hh : IsHeight h)
    (j : Fin n) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n) (hm : 0 < m) :
    frozenColumn h j ↔ ¬ activeColumn h j :=
  ⟨frozenColumn_not_active h j, inactive_imp_frozenColumn h hh j hj0 hj1 hm⟩

end OrigamiCone.Sequel
