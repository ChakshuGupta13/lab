import OrigamiCone.Cone

namespace OrigamiCone

variable {m n : ℕ}

lemma gdist_parity (p v : Cell m n) :
    (gdist p v - ((p.1.val : ℤ) + p.2.val + v.1.val + v.2.val)) % 2 = 0 := by
  unfold gdist
  omega

lemma cone_pair_same_parity {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) (v : Cell m n) :
    (gdist p₁ v - (δ + gdist p₂ v)) % 2 = 0 := by
  have e1 := gdist_parity p₁ v
  have e2 := gdist_parity p₂ v
  have e3 := gdist_parity p₁ p₂
  omega

theorem parity_isHeight {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) :
    IsHeight (fun v => min (gdist p₁ v) (δ + gdist p₂ v)) := by
  intro v v' hvv'

  set a := gdist p₁ v with ha
  set b := δ + gdist p₂ v with hb
  set a' := gdist p₁ v' with ha'
  set b' := δ + gdist p₂ v' with hb'

  have hstep1 : gdist p₁ v = gdist p₁ v' + 1 ∨ gdist p₁ v = gdist p₁ v' - 1 :=
    gdist_adj_step (q := p₁) hvv'
  have hstep2 : gdist p₂ v = gdist p₂ v' + 1 ∨ gdist p₂ v = gdist p₂ v' - 1 :=
    gdist_adj_step (q := p₂) hvv'

  have hpar : (a - b) % 2 = 0 := cone_pair_same_parity hδ v
  have hpar' : (a' - b') % 2 = 0 := cone_pair_same_parity hδ v'

  rw [abs_eq (by norm_num : (0:ℤ) ≤ 1)]
  simp only [ha, hb, ha', hb'] at *
  omega

end OrigamiCone
