import OrigamiCone.Parity

namespace OrigamiCone

variable {m n : ℕ}

def cpe (p₁ p₂ : Cell m n) (δ : ℤ) (v : Cell m n) : ℤ :=
  min (gdist p₁ v) (δ + gdist p₂ v)

lemma conePair_apexL_strictMin {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) (hact : -gdist p₁ p₂ < δ) :
    IsStrictLocalMin (cpe p₁ p₂ δ) p₁ := by

  have hD : gdist p₂ p₁ = gdist p₁ p₂ := gdist_comm p₂ p₁
  have hp1 : cpe p₁ p₂ δ p₁ = 0 := by
    unfold cpe
    rw [gdist_self, hD]
    have : (0 : ℤ) ≤ δ + gdist p₁ p₂ := by omega
    omega
  intro u hu

  have hu1 : gdist p₁ u = 1 := hu
  have htri := gdist_triangle p₂ u p₁
  have huone : gdist u p₁ = 1 := by rw [gdist_comm]; exact hu

  have hval : cpe p₁ p₂ δ u = 1 := by
    unfold cpe
    rw [hu1]
    have : (1 : ℤ) ≤ δ + gdist p₂ u := by omega
    omega
  omega

lemma conePair_apexR_strictMin {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) (hact : δ < gdist p₁ p₂) :
    IsStrictLocalMin (cpe p₁ p₂ δ) p₂ := by

  have hp2 : cpe p₁ p₂ δ p₂ = δ := by
    unfold cpe
    rw [gdist_self]
    have : δ + 0 ≤ gdist p₁ p₂ := by omega
    omega
  intro u hu
  have hu2 : gdist p₂ u = 1 := hu
  have htri := gdist_triangle p₁ u p₂
  have huone : gdist u p₂ = 1 := by rw [gdist_comm]; exact hu

  have hval : cpe p₁ p₂ δ u = δ + 1 := by
    unfold cpe
    rw [hu2]
    have : δ + 1 ≤ gdist p₁ u := by omega
    omega
  omega

lemma conePair_strictMin_imp_apex {p₁ p₂ : Cell m n} {δ : ℤ} {w : Cell m n}
    (hw : IsStrictLocalMin (cpe p₁ p₂ δ) w) : w = p₁ ∨ w = p₂ := by
  by_contra hcon
  push_neg at hcon
  obtain ⟨hw1, hw2⟩ := hcon

  rcases (by omega : gdist p₁ w ≤ δ + gdist p₂ w ∨ δ + gdist p₂ w < gdist p₁ w) with hle | hlt
  ·
    have hval : cpe p₁ p₂ δ w = gdist p₁ w := by unfold cpe; omega
    obtain ⟨u, hadj, hstep⟩ := exists_step_toward hw1
    have hcone1 : gdist p₁ u = gdist p₁ w - 1 := by
      rw [gdist_comm p₁ u, hstep, gdist_comm w p₁]
    have hlo : cpe p₁ p₂ δ u ≤ gdist p₁ w - 1 := by
      unfold cpe; rw [← hcone1]; exact min_le_left _ _
    have hstrict := hw u hadj
    rw [hval] at hstrict
    omega
  ·
    have hval : cpe p₁ p₂ δ w = δ + gdist p₂ w := by unfold cpe; omega
    obtain ⟨u, hadj, hstep⟩ := exists_step_toward hw2
    have hcone2 : gdist p₂ u = gdist p₂ w - 1 := by
      rw [gdist_comm p₂ u, hstep, gdist_comm w p₂]
    have hlo : cpe p₁ p₂ δ u ≤ δ + gdist p₂ w - 1 := by
      unfold cpe
      have : δ + gdist p₂ u = δ + gdist p₂ w - 1 := by rw [hcone2]; ring
      rw [← this]; exact min_le_right _ _
    have hstrict := hw u hadj
    rw [hval] at hstrict
    omega

theorem conePair_strictMin_iff {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0)
    (hactL : -gdist p₁ p₂ < δ) (hactR : δ < gdist p₁ p₂) (w : Cell m n) :
    IsStrictLocalMin (cpe p₁ p₂ δ) w ↔ w = p₁ ∨ w = p₂ := by
  constructor
  · exact conePair_strictMin_imp_apex
  · rintro (rfl | rfl)
    · exact conePair_apexL_strictMin hδ hactL
    · exact conePair_apexR_strictMin hδ hactR

end OrigamiCone
