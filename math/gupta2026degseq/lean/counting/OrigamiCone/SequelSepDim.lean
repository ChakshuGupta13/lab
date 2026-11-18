import Mathlib

namespace OrigamiCone.Sequel

lemma dim_identity (ρ γ ρ' γ' : ℕ)
    (hρ : 1 ≤ ρ) (hγ : 1 ≤ γ) (hρ' : 1 ≤ ρ') (hγ' : 1 ≤ γ') :
    ρ * γ + ρ' * γ'
      = ((ρ + ρ' - 2) + (γ + γ' - 2)) + 2
        + ((ρ - 1) * (γ - 1) + (ρ' - 1) * (γ' - 1)) := by
  have h2ρ : 2 ≤ ρ + ρ' := by omega
  have h2γ : 2 ≤ γ + γ' := by omega
  zify [hρ, hγ, hρ', hγ', h2ρ, h2γ]
  ring

lemma dim_codim (ρ γ ρ' γ' : ℕ)
    (hρ : 1 ≤ ρ) (hγ : 1 ≤ γ) (hρ' : 1 ≤ ρ') (hγ' : 1 ≤ γ') :
    (ρ * γ + ρ' * γ' - 2) - ((ρ + ρ' - 2) + (γ + γ' - 2))
      = (ρ - 1) * (γ - 1) + (ρ' - 1) * (γ' - 1) := by
  have hid := dim_identity ρ γ ρ' γ' hρ hγ hρ' hγ'
  omega

lemma dim_le (ρ γ ρ' γ' : ℕ)
    (hρ : 1 ≤ ρ) (hγ : 1 ≤ γ) (hρ' : 1 ≤ ρ') (hγ' : 1 ≤ γ') :
    (ρ + ρ' - 2) + (γ + γ' - 2) ≤ ρ * γ + ρ' * γ' - 2 := by
  have hid := dim_identity ρ γ ρ' γ' hρ hγ hρ' hγ'
  omega

def SingleEdge (ρ γ ρ' γ' : ℕ) : Prop := (ρ = 1 ∧ ρ' = 1) ∨ (γ = 1 ∧ γ' = 1)

theorem sep_dim_bound (ρ γ ρ' γ' : ℕ)
    (hρ : 1 ≤ ρ) (hγ : 1 ≤ γ) (hρ' : 1 ≤ ρ') (hγ' : 1 ≤ γ')
    (hrow : ρ ≤ ρ' + 1 ∧ ρ' ≤ ρ + 1)
    (hcol : γ ≤ γ' + 1 ∧ γ' ≤ γ + 1)
    (hd : 5 ≤ ρ * γ + ρ' * γ')
    (hmax : (ρ + ρ' - 2) + (γ + γ' - 2) = ρ * γ + ρ' * γ' - 2) :
    SingleEdge ρ γ ρ' γ' := by

  have hid := dim_identity ρ γ ρ' γ' hρ hγ hρ' hγ'
  have hP1 : (ρ - 1) * (γ - 1) = 0 := by omega
  have hP2 : (ρ' - 1) * (γ' - 1) = 0 := by omega
  rcases Nat.mul_eq_zero.mp hP1 with h1 | h1
  ·
    have hρ1 : ρ = 1 := by omega
    rcases Nat.mul_eq_zero.mp hP2 with h2 | h2
    ·
      exact Or.inl ⟨hρ1, by omega⟩
    ·
      exfalso
      have hγ'1 : γ' = 1 := by omega
      subst hρ1; subst hγ'1
      simp only [one_mul, mul_one] at hd
      omega
  ·
    have hγ1 : γ = 1 := by omega
    rcases Nat.mul_eq_zero.mp hP2 with h2 | h2
    ·
      exfalso
      have hρ'1 : ρ' = 1 := by omega
      subst hγ1; subst hρ'1
      simp only [one_mul, mul_one] at hd
      omega
    ·
      exact Or.inr ⟨hγ1, by omega⟩

end OrigamiCone.Sequel
