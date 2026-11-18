import OrigamiCone.Cone

namespace OrigamiCone

variable {m n : ℕ}

lemma exists_descent_to_min {h : Cell m n → ℤ} (hh : IsHeight h) (v : Cell m n) :
    ∃ p, IsStrictLocalMin h p ∧ h v = h p + gdist p v := by
  haveI : Nonempty (Cell m n) := ⟨v⟩
  obtain ⟨w0, hw0⟩ := Finite.exists_min h

  suffices key : ∀ (k : ℕ) (v : Cell m n), (h v - h w0).toNat ≤ k →
      ∃ p, IsStrictLocalMin h p ∧ h v = h p + gdist p v by
    exact key (h v - h w0).toNat v le_rfl
  intro k
  induction k with
  | zero =>
    intro v hk
    have hmin0 : h v = h w0 := by have := hw0 v; omega
    have hvmin : IsStrictLocalMin h v := by
      intro u hu
      have hle := hw0 u
      have h1 := hh v u hu
      rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega
    exact ⟨v, hvmin, by simp [gdist_self]⟩
  | succ k ih =>
    intro v hk
    by_cases hvmin : IsStrictLocalMin h v
    · exact ⟨v, hvmin, by simp [gdist_self]⟩
    ·
      simp only [IsStrictLocalMin, not_forall] at hvmin
      obtain ⟨u, hu⟩ := hvmin
      have hadj : adj v u := by tauto
      have hune : h u ≠ h v + 1 := by tauto
      have h1 := hh v u hadj
      have hdown : h u = h v - 1 := by
        rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 h1 with h2 | h2
        · omega
        · exact absurd (by omega : h u = h v + 1) hune
      have hdefu : (h u - h w0).toNat ≤ k := by have := hw0 u; omega
      obtain ⟨p, hpmin, hpu⟩ := ih u hdefu

      have htri := gdist_triangle p u v
      have huv : gdist u v = 1 := by rw [gdist_comm]; exact hadj
      have hlip : |h p - h v| ≤ gdist p v :=
        height_lipschitz hh (gdist p v).toNat p v
          (le_of_eq (Int.toNat_of_nonneg (gdist_nonneg p v)).symm)
      have hge0 := gdist_nonneg p u
      have hpv : gdist p v = gdist p u + 1 := by
        rw [abs_le] at hlip; omega
      exact ⟨p, hpmin, by omega⟩

lemma envelope_le {h : Cell m n → ℤ} (hh : IsHeight h) (p v : Cell m n) :
    h v ≤ h p + gdist p v := by
  have hlip : |h v - h p| ≤ gdist v p :=
    height_lipschitz hh (gdist v p).toNat v p
      (le_of_eq (Int.toNat_of_nonneg (gdist_nonneg v p)).symm)
  rw [gdist_comm v p, abs_le] at hlip
  omega

theorem envelope_min {h : Cell m n → ℤ} (hh : IsHeight h) (v : Cell m n) :
    (∃ p, IsStrictLocalMin h p ∧ h v = h p + gdist p v)
      ∧ (∀ p, IsStrictLocalMin h p → h v ≤ h p + gdist p v) :=
  ⟨exists_descent_to_min hh v, fun p _ => envelope_le hh p v⟩

theorem envelope_max {h : Cell m n → ℤ} (hh : IsHeight h) (v : Cell m n) :
    (∃ q, IsStrictLocalMax h q ∧ h v = h q - gdist q v)
      ∧ (∀ q, IsStrictLocalMax h q → h q - gdist q v ≤ h v) := by
  have hh' : IsHeight (fun w => -h w) := by
    intro a b hab
    have := hh a b hab
    rw [show (-h a) - (-h b) = -(h a - h b) by ring, abs_neg]; exact this
  obtain ⟨⟨p, hpmin, hpeq⟩, hple⟩ := envelope_min hh' v
  constructor
  · refine ⟨p, ?_, ?_⟩
    · intro u hu
      have := hpmin u hu; simp only at this; linarith
    · have := hpeq; simp only at this; linarith
  · intro q hq
    have hq' : IsStrictLocalMin (fun w => -h w) q := by
      intro u hu; have := hq u hu; simp only; linarith
    have := hple q hq'; linarith

end OrigamiCone
