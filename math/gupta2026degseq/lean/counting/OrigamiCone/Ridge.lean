import OrigamiCone.Basic

namespace OrigamiCone

variable {m n : ℕ}

def IsCorner (v : Cell m n) : Prop :=
  (v.1.val = 0 ∨ v.1.val = m - 1) ∧ (v.2.val = 0 ∨ v.2.val = n - 1)

lemma exists_farther_of_interior_row {p v : Cell m n}
    (hi : 0 < v.1.val ∧ v.1.val < m - 1) :
    ∃ u, adj v u ∧ gdist p u = gdist p v + 1 := by
  have h1 := v.1.isLt
  rcases (by omega : p.1.val ≤ v.1.val ∨ v.1.val < p.1.val) with hpv | hpv
  ·
    refine ⟨(⟨v.1.val + 1, by omega⟩, v.2), ?_, ?_⟩
    · unfold adj gdist; dsimp only; omega
    · unfold gdist; dsimp only; omega
  ·
    refine ⟨(⟨v.1.val - 1, by omega⟩, v.2), ?_, ?_⟩
    · unfold adj gdist; dsimp only; omega
    · unfold gdist; dsimp only; omega

lemma exists_farther_of_interior_col {p v : Cell m n}
    (hj : 0 < v.2.val ∧ v.2.val < n - 1) :
    ∃ u, adj v u ∧ gdist p u = gdist p v + 1 := by
  have h2 := v.2.isLt
  rcases (by omega : p.2.val ≤ v.2.val ∨ v.2.val < p.2.val) with hpv | hpv
  · refine ⟨(v.1, ⟨v.2.val + 1, by omega⟩), ?_, ?_⟩
    · unfold adj gdist; dsimp only; omega
    · unfold gdist; dsimp only; omega
  · refine ⟨(v.1, ⟨v.2.val - 1, by omega⟩), ?_, ?_⟩
    · unfold adj gdist; dsimp only; omega
    · unfold gdist; dsimp only; omega

theorem gdist_strictMax_imp_corner {p v : Cell m n}
    (hv : IsStrictLocalMax (gdist p) v) : IsCorner v := by
  by_contra hc
  by_cases hrow : v.1.val = 0 ∨ v.1.val = m - 1
  · by_cases hcol : v.2.val = 0 ∨ v.2.val = n - 1
    · exact hc ⟨hrow, hcol⟩
    · have hji : 0 < v.2.val ∧ v.2.val < n - 1 := by have := v.2.isLt; omega
      obtain ⟨u, hadj, hfar⟩ := exists_farther_of_interior_col (p := p) hji
      have := hv u hadj; omega
  · have hii : 0 < v.1.val ∧ v.1.val < m - 1 := by have := v.1.isLt; omega
    obtain ⟨u, hadj, hfar⟩ := exists_farther_of_interior_row (p := p) hii
    have := hv u hadj; omega

lemma corner_origin_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) (p : Cell m n)
    (v : Cell m n) (hv0 : v.1.val = 0 ∧ v.2.val = 0) :
    IsStrictLocalMax (gdist p) v ↔ 1 ≤ p.1.val ∧ 1 ≤ p.2.val := by
  obtain ⟨hi, hj⟩ := hv0
  constructor
  · intro hmax

    have hu1 : adj v (⟨1, by omega⟩, v.2) := by unfold adj gdist; dsimp only; omega
    have hu2 : adj v (v.1, ⟨1, by omega⟩) := by unfold adj gdist; dsimp only; omega
    have e1 := hmax _ hu1
    have e2 := hmax _ hu2
    unfold gdist at e1 e2
    dsimp only at e1 e2
    omega
  · rintro ⟨hp1, hp2⟩ u hu
    unfold adj gdist at hu
    unfold gdist
    omega

end OrigamiCone
