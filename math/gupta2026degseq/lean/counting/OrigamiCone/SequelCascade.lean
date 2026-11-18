import Mathlib

namespace OrigamiCone.Sequel

theorem zmod3_double (x : ZMod 3) : x + x = -x := by revert x; decide

theorem zmod3_neg_sub (x : ZMod 3) : -x - x = x := by revert x; decide

def IsExtremum (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m j : ℕ) : Prop :=
  α j = γ j ∧ (0 < j → s (j-1) = -α j) ∧ (j + 1 < m → s j = α j)

def AtR (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (r : ZMod 3) (j : ℕ) : Prop :=
  α j = r ∧ γ j = r ∧ (0 < j → s (j-1) = -r)

theorem cascade_step
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m : ℕ)
    (hα : ∀ i, i < m → α i ≠ 0)
    (hγ : ∀ i, i < m → γ i ≠ 0)
    (hs : ∀ i, i + 1 < m → s i ≠ 0)
    (hpu : ∀ i, i + 1 < m → α (i+1) ≠ α i - s i)
    (hpw : ∀ i, i + 1 < m → γ (i+1) ≠ γ i - s i)
    (hNoExt : ∀ j, j < m → ¬ IsExtremum α γ s m j)
    (r : ZMod 3) (hr : r ≠ 0)
    (j : ℕ) (hj : j + 1 < m)
    (hAt : AtR α γ s r j) :
    AtR α γ s r (j + 1) := by
  obtain ⟨hαj, hγj, hsPrev⟩ := hAt
  have hjLt : j < m := by omega
  have hne := hNoExt j hjLt

  have hsj_ne_r : s j ≠ r := by
    intro h
    apply hne
    refine ⟨hαj.trans hγj.symm, ?_, ?_⟩
    · intro hjp; rw [hαj]; exact hsPrev hjp
    · intro _; rw [hαj]; exact h
  have hsj_ne_0 : s j ≠ 0 := hs _ hj
  have key₁ : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ b → a = -b := by decide
  have hsj : s j = -r := key₁ _ _ hsj_ne_0 hr hsj_ne_r

  have hα1_ne_0 : α (j+1) ≠ 0 := hα _ hj
  have hγ1_ne_0 : γ (j+1) ≠ 0 := hγ _ hj
  have hα1_ne : α (j+1) ≠ -r := by
    intro h; apply hpu _ hj
    rw [h, hsj, hαj, sub_neg_eq_add, zmod3_double]
  have hγ1_ne : γ (j+1) ≠ -r := by
    intro h; apply hpw _ hj
    rw [h, hsj, hγj, sub_neg_eq_add, zmod3_double]
  have key₂ : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ -b → a = b := by decide
  refine ⟨key₂ _ _ hα1_ne_0 hr hα1_ne, key₂ _ _ hγ1_ne_0 hr hγ1_ne, ?_⟩
  intro _
  show s (j + 1 - 1) = -r
  have hsub : j + 1 - 1 = j := by omega
  rw [hsub]; exact hsj

theorem cascade_extend
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m : ℕ)
    (hα : ∀ i, i < m → α i ≠ 0)
    (hγ : ∀ i, i < m → γ i ≠ 0)
    (hs : ∀ i, i + 1 < m → s i ≠ 0)
    (hpu : ∀ i, i + 1 < m → α (i+1) ≠ α i - s i)
    (hpw : ∀ i, i + 1 < m → γ (i+1) ≠ γ i - s i)
    (hNoExt : ∀ j, j < m → ¬ IsExtremum α γ s m j)
    (r : ZMod 3) (hr : r ≠ 0)
    (p : ℕ) (hAtP : AtR α γ s r p) :
    ∀ k, p + k < m → AtR α γ s r (p + k) := by
  intro k hk
  induction k with
  | zero => simpa using hAtP
  | succ k IH =>
    have hk' : p + k + 1 < m := hk
    have hpk : p + k < m := by omega
    have ih := IH (by omega)
    exact cascade_step α γ s m hα hγ hs hpu hpw hNoExt r hr (p + k) hk' ih

theorem up_rule_base
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (p : ℕ)
    (hαp1 : α (p-1) ≠ 0) (hγp1 : γ (p-1) ≠ 0)
    (hsp1_ne_0 : s (p-1) ≠ 0)
    (hpu : α p ≠ α (p-1) - s (p-1))
    (hpw : γ p ≠ γ (p-1) - s (p-1))
    (r : ZMod 3) (hr : r ≠ 0)
    (hAlP : α p = r ∧ γ p = r)
    (hNotAlPrev : α (p-1) ≠ γ (p-1)) :
    s (p-1) = -r := by

  have hsp1_ne_r : s (p-1) ≠ r := by
    intro h
    have hα_ne_neg : α (p-1) ≠ -r := by
      intro hα; apply hpu; rw [hα, h, hAlP.1, zmod3_neg_sub]
    have hγ_ne_neg : γ (p-1) ≠ -r := by
      intro hγ; apply hpw; rw [hγ, h, hAlP.2, zmod3_neg_sub]
    have key : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ -b → a = b := by decide
    exact hNotAlPrev ((key _ _ hαp1 hr hα_ne_neg).trans (key _ _ hγp1 hr hγ_ne_neg).symm)
  have key₁ : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ b → a = -b := by decide
  exact key₁ _ _ hsp1_ne_0 hr hsp1_ne_r

private theorem bottom_contradiction
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m : ℕ) (hm : 1 ≤ m)
    (r : ZMod 3)
    (hNoExt : ∀ j, j < m → ¬ IsExtremum α γ s m j)
    (hAt : AtR α γ s r (m - 1)) : False := by
  apply hNoExt (m - 1) (by omega)
  obtain ⟨hα, hγ, hsPrev⟩ := hAt
  refine ⟨hα.trans hγ.symm, ?_, ?_⟩
  · intro hpos
    have hsub : m - 1 - 1 = m - 2 := by omega
    rw [hα]; rw [show (m - 1 : ℕ) - 1 = m - 1 - 1 from rfl, hsub]
    have := hsPrev hpos
    rw [hsub] at this
    exact this
  · intro h; omega

theorem cascade
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m : ℕ)
    (hα : ∀ i, i < m → α i ≠ 0)
    (hγ : ∀ i, i < m → γ i ≠ 0)
    (hs : ∀ i, i + 1 < m → s i ≠ 0)
    (hpu : ∀ i, i + 1 < m → α (i+1) ≠ α i - s i)
    (hpw : ∀ i, i + 1 < m → γ (i+1) ≠ γ i - s i)
    (hNoExt : ∀ j, j < m → ¬ IsExtremum α γ s m j) :
    ∀ i, i < m → α i ≠ γ i := by
  intro i hi hAl
  classical

  let P : ℕ → Prop := fun j => j < m ∧ α j = γ j
  have hexists : ∃ j, P j := ⟨i, hi, hAl⟩
  set p := Nat.find hexists with hp_def
  obtain ⟨hp_lt, hp_al⟩ : P p := Nat.find_spec hexists
  have hp_min : ∀ q, q < p → ¬ P q := fun q hq => Nat.find_min hexists hq

  set r := α p with hr_def
  have hr_ne : r ≠ 0 := hα p hp_lt
  have hAlP : α p = r ∧ γ p = r := ⟨rfl, hp_al.symm⟩
  have hAtP : AtR α γ s r p := by
    refine ⟨rfl, hp_al.symm, ?_⟩
    intro hpos

    have hNotAlPrev : α (p - 1) ≠ γ (p - 1) := by
      intro hEq; apply hp_min (p - 1) (by omega); exact ⟨by omega, hEq⟩
    have hp1_lt : p - 1 < m := by omega
    have hp_step : p - 1 + 1 < m := by omega
    have hp_eq : p - 1 + 1 = p := by omega
    have hpu_p : α p ≠ α (p - 1) - s (p - 1) := by
      have := hpu (p - 1) hp_step
      rw [hp_eq] at this; exact this
    have hpw_p : γ p ≠ γ (p - 1) - s (p - 1) := by
      have := hpw (p - 1) hp_step
      rw [hp_eq] at this; exact this
    exact up_rule_base α γ s p (hα _ hp1_lt) (hγ _ hp1_lt) (hs _ hp_step)
      hpu_p hpw_p r hr_ne hAlP hNotAlPrev

  have hk : p + (m - 1 - p) < m := by omega
  have hAtBottom : AtR α γ s r (p + (m - 1 - p)) :=
    cascade_extend α γ s m hα hγ hs hpu hpw hNoExt r hr_ne p hAtP _ hk
  have hsum : p + (m - 1 - p) = m - 1 := by omega
  rw [hsum] at hAtBottom
  exact bottom_contradiction α γ s m (by omega) r hNoExt hAtBottom

end OrigamiCone.Sequel
