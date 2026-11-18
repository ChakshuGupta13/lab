import OrigamiCone.ConePair
import OrigamiCone.Ridge

namespace OrigamiCone

variable {m n : ℕ}

def OnRidge (p₁ p₂ : Cell m n) (δ : ℤ) (v : Cell m n) : Prop :=
  gdist p₁ v = δ + gdist p₂ v

lemma cpe_on_ridge {p₁ p₂ : Cell m n} {δ : ℤ} {v : Cell m n}
    (hv : OnRidge p₁ p₂ δ v) : cpe p₁ p₂ δ v = gdist p₁ v := by
  unfold cpe
  unfold OnRidge at hv
  rw [← hv, min_self]

lemma ridge_strictMax_iff_closer {p₁ p₂ : Cell m n} {δ : ℤ} {v : Cell m n}
    (hv : OnRidge p₁ p₂ δ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔
      ∀ u, adj v u → gdist p₁ u < gdist p₁ v ∨ gdist p₂ u < gdist p₂ v := by
  have hcv : cpe p₁ p₂ δ v = gdist p₁ v := cpe_on_ridge hv
  unfold OnRidge at hv
  constructor
  · intro hmax u huv
    have hu := hmax u huv
    rw [hcv] at hu
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    unfold cpe at hu
    rcases le_total (gdist p₁ u) (δ + gdist p₂ u) with hm | hm
    · rw [min_eq_left hm] at hu; omega
    · rw [min_eq_right hm] at hu; omega
  · intro hclose u huv
    have hc := hclose u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    show cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1
    rw [hcv]
    unfold cpe
    rcases le_total (gdist p₁ u) (δ + gdist p₂ u) with hm | hm
    · rw [min_eq_left hm]; omega
    · rw [min_eq_right hm]; omega

def RowAdm (p₁ p₂ v : Cell m n) : Prop :=
  v.1.val = 0 ∨ v.1.val = m - 1 ∨
    (min p₁.1.val p₂.1.val < v.1.val ∧ v.1.val < max p₁.1.val p₂.1.val)

def ColAdm (p₁ p₂ v : Cell m n) : Prop :=
  v.2.val = 0 ∨ v.2.val = n - 1 ∨
    (min p₁.2.val p₂.2.val < v.2.val ∧ v.2.val < max p₁.2.val p₂.2.val)

private lemma adj_split {v u : Cell m n} (h : adj v u) :
    (u.1 = v.1 ∧ (u.2.val = v.2.val + 1 ∨ u.2.val + 1 = v.2.val)) ∨
    (u.2 = v.2 ∧ (u.1.val = v.1.val + 1 ∨ u.1.val + 1 = v.1.val)) := by
  have hn : ((v.1.val : ℤ) - u.1.val).natAbs + ((v.2.val : ℤ) - u.2.val).natAbs = 1 := by
    have h' := h; unfold adj gdist at h'; exact_mod_cast h'
  by_cases hr : u.1.val = v.1.val
  · left; exact ⟨Fin.ext hr, by omega⟩
  · right; exact ⟨Fin.ext (by omega), by omega⟩

theorem ridge_strictMax_iff_admissible {p₁ p₂ : Cell m n} {δ : ℤ}
    (hr : p₁.1.val ≠ p₂.1.val) (hs : p₁.2.val ≠ p₂.2.val)
    {v : Cell m n} (hv : OnRidge p₁ p₂ δ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔ RowAdm p₁ p₂ v ∧ ColAdm p₁ p₂ v := by
  have hp1 := p₁.1.isLt; have hp2 := p₂.1.isLt
  have hq1 := p₁.2.isLt; have hq2 := p₂.2.isLt
  have hvi := v.1.isLt; have hvj := v.2.isLt
  rw [ridge_strictMax_iff_closer hv]
  constructor
  · intro hclose
    refine ⟨?_, ?_⟩
    ·
      rcases (by omega : v.1.val = 0 ∨ v.1.val = m - 1 ∨
          (0 < v.1.val ∧ v.1.val < m - 1)) with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · refine Or.inr (Or.inr ⟨?_, ?_⟩)
        · have hb : v.1.val - 1 < m := by omega
          have hadj : adj v (⟨v.1.val - 1, hb⟩, v.2) := by
            unfold adj gdist; dsimp only; omega
          have hcl := hclose (⟨v.1.val - 1, hb⟩, v.2) hadj
          unfold gdist at hcl; dsimp only at hcl; omega
        · have hb : v.1.val + 1 < m := by omega
          have hadj : adj v (⟨v.1.val + 1, hb⟩, v.2) := by
            unfold adj gdist; dsimp only; omega
          have hcl := hclose (⟨v.1.val + 1, hb⟩, v.2) hadj
          unfold gdist at hcl; dsimp only at hcl; omega
    ·
      rcases (by omega : v.2.val = 0 ∨ v.2.val = n - 1 ∨
          (0 < v.2.val ∧ v.2.val < n - 1)) with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · refine Or.inr (Or.inr ⟨?_, ?_⟩)
        · have hb : v.2.val - 1 < n := by omega
          have hadj : adj v (v.1, ⟨v.2.val - 1, hb⟩) := by
            unfold adj gdist; dsimp only; omega
          have hcl := hclose (v.1, ⟨v.2.val - 1, hb⟩) hadj
          unfold gdist at hcl; dsimp only at hcl; omega
        · have hb : v.2.val + 1 < n := by omega
          have hadj : adj v (v.1, ⟨v.2.val + 1, hb⟩) := by
            unfold adj gdist; dsimp only; omega
          have hcl := hclose (v.1, ⟨v.2.val + 1, hb⟩) hadj
          unfold gdist at hcl; dsimp only at hcl; omega
  · rintro ⟨hrow, hcol⟩ u huv
    rcases adj_split huv with ⟨he, hjj⟩ | ⟨he, hii⟩
    ·
      have hev : u.1.val = v.1.val := by rw [he]
      have hui := u.2.isLt
      unfold RowAdm at hrow; unfold ColAdm at hcol
      unfold gdist; omega
    ·
      have hev : u.2.val = v.2.val := by rw [he]
      have hui := u.1.isLt
      unfold RowAdm at hrow; unfold ColAdm at hcol
      unfold gdist; omega

lemma cpe_strictMax_off_ridge_imp_corner {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) {v : Cell m n}
    (hoff : ¬ OnRidge p₁ p₂ δ v)
    (hmax : IsStrictLocalMax (cpe p₁ p₂ δ) v) : IsCorner v := by
  have hpar := cone_pair_same_parity hδ v
  unfold OnRidge at hoff
  rcases (by omega : gdist p₁ v < δ + gdist p₂ v ∨ gdist p₁ v > δ + gdist p₂ v)
      with hlt | hgt
  ·
    refine gdist_strictMax_imp_corner (p := p₁) ?_
    intro u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    unfold cpe at hmu
    omega
  ·
    refine gdist_strictMax_imp_corner (p := p₂) ?_
    intro u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    unfold cpe at hmu
    omega

theorem cpe_strictMax_imp {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0)
    (hr : p₁.1.val ≠ p₂.1.val) (hs : p₁.2.val ≠ p₂.2.val)
    {v : Cell m n} (hmax : IsStrictLocalMax (cpe p₁ p₂ δ) v) :
    (OnRidge p₁ p₂ δ v ∧ RowAdm p₁ p₂ v ∧ ColAdm p₁ p₂ v) ∨ IsCorner v := by
  by_cases hon : OnRidge p₁ p₂ δ v
  · exact Or.inl ⟨hon, (ridge_strictMax_iff_admissible hr hs hon).mp hmax⟩
  · exact Or.inr (cpe_strictMax_off_ridge_imp_corner hδ hon hmax)

lemma corner_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) {p v : Cell m n}
    (hc : IsCorner v) :
    IsStrictLocalMax (gdist p) v ↔ p.1.val ≠ v.1.val ∧ p.2.val ≠ v.2.val := by
  obtain ⟨hrow, hcol⟩ := hc
  have hp1 := p.1.isLt; have hp2 := p.2.isLt
  have hv1 := v.1.isLt; have hv2 := v.2.isLt
  constructor
  · intro hmax
    refine ⟨?_, ?_⟩
    · rcases hrow with h0 | h1
      · have hadj : adj v (⟨1, hm⟩, v.2) := by unfold adj gdist; dsimp only; omega
        have hcl := hmax _ hadj
        unfold gdist at hcl; dsimp only at hcl; omega
      · have hadj : adj v (⟨m - 2, by omega⟩, v.2) := by unfold adj gdist; dsimp only; omega
        have hcl := hmax _ hadj
        unfold gdist at hcl; dsimp only at hcl; omega
    · rcases hcol with h0 | h1
      · have hadj : adj v (v.1, ⟨1, hn⟩) := by unfold adj gdist; dsimp only; omega
        have hcl := hmax _ hadj
        unfold gdist at hcl; dsimp only at hcl; omega
      · have hadj : adj v (v.1, ⟨n - 2, by omega⟩) := by unfold adj gdist; dsimp only; omega
        have hcl := hmax _ hadj
        unfold gdist at hcl; dsimp only at hcl; omega
  · rintro ⟨hpr, hpc⟩ u huv
    rcases adj_split huv with ⟨he, hjj⟩ | ⟨he, hii⟩
    · have hev : u.1.val = v.1.val := by rw [he]
      have hui := u.2.isLt
      unfold gdist; omega
    · have hev : u.2.val = v.2.val := by rw [he]
      have hui := u.1.isLt
      unfold gdist; omega

lemma gdist_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) {p v : Cell m n} :
    IsStrictLocalMax (gdist p) v ↔
      IsCorner v ∧ p.1.val ≠ v.1.val ∧ p.2.val ≠ v.2.val := by
  constructor
  · intro h
    have hcor := gdist_strictMax_imp_corner h
    exact ⟨hcor, (corner_strictMax_iff hm hn hcor).mp h⟩
  · rintro ⟨hc, hav⟩
    exact (corner_strictMax_iff hm hn hc).mpr hav

lemma cpe_strictMax_iff_cone_left {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) {v : Cell m n}
    (hlt : gdist p₁ v < δ + gdist p₂ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔ IsStrictLocalMax (gdist p₁) v := by
  have hpar := cone_pair_same_parity hδ v
  have hcv : cpe p₁ p₂ δ v = gdist p₁ v := by unfold cpe; exact min_eq_left (le_of_lt hlt)
  constructor
  · intro hmax u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    have hle : gdist p₁ u ≤ δ + gdist p₂ u := by omega
    have hcu : cpe p₁ p₂ δ u = gdist p₁ u := by unfold cpe; exact min_eq_left hle
    rw [hcu, hcv] at hmu; exact hmu
  · intro hmax u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    have hle : gdist p₁ u ≤ δ + gdist p₂ u := by omega
    have hcu : cpe p₁ p₂ δ u = gdist p₁ u := by unfold cpe; exact min_eq_left hle
    show cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1
    rw [hcu, hcv]; exact hmu

lemma cpe_strictMax_iff_cone_right {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) {v : Cell m n}
    (hgt : δ + gdist p₂ v < gdist p₁ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔ IsStrictLocalMax (gdist p₂) v := by
  have hpar := cone_pair_same_parity hδ v
  have hcv : cpe p₁ p₂ δ v = δ + gdist p₂ v := by unfold cpe; exact min_eq_right (le_of_lt hgt)
  constructor
  · intro hmax u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    have hle : δ + gdist p₂ u ≤ gdist p₁ u := by omega
    have hcu : cpe p₁ p₂ δ u = δ + gdist p₂ u := by unfold cpe; exact min_eq_right hle
    rw [hcu, hcv] at hmu; omega
  · intro hmax u huv
    have hmu := hmax u huv
    have s1 := gdist_adj_step (q := p₁) huv
    have s2 := gdist_adj_step (q := p₂) huv
    have hle : δ + gdist p₂ u ≤ gdist p₁ u := by omega
    have hcu : cpe p₁ p₂ δ u = δ + gdist p₂ u := by unfold cpe; exact min_eq_right hle
    show cpe p₁ p₂ δ u = cpe p₁ p₂ δ v - 1
    rw [hcu, hcv]; omega

def QualifyingCorner (p₁ p₂ : Cell m n) (δ : ℤ) (v : Cell m n) : Prop :=
  IsCorner v ∧
    (gdist p₁ v < δ + gdist p₂ v → p₁.1.val ≠ v.1.val ∧ p₁.2.val ≠ v.2.val) ∧
    (δ + gdist p₂ v < gdist p₁ v → p₂.1.val ≠ v.1.val ∧ p₂.2.val ≠ v.2.val)

theorem cpe_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0)
    (hr : p₁.1.val ≠ p₂.1.val) (hs : p₁.2.val ≠ p₂.2.val) {v : Cell m n} :
    IsStrictLocalMax (cpe p₁ p₂ δ) v ↔
      (OnRidge p₁ p₂ δ v ∧ RowAdm p₁ p₂ v ∧ ColAdm p₁ p₂ v) ∨
      (¬ OnRidge p₁ p₂ δ v ∧ QualifyingCorner p₁ p₂ δ v) := by
  by_cases hon : OnRidge p₁ p₂ δ v
  · rw [ridge_strictMax_iff_admissible hr hs hon]
    constructor
    · intro h; exact Or.inl ⟨hon, h⟩
    · rintro (⟨_, h⟩ | ⟨hno, _⟩)
      · exact h
      · exact absurd hon hno
  ·
    have hne : gdist p₁ v ≠ δ + gdist p₂ v := hon
    rcases (by omega : gdist p₁ v < δ + gdist p₂ v ∨ δ + gdist p₂ v < gdist p₁ v)
        with hlt | hgt
    · rw [cpe_strictMax_iff_cone_left hδ hlt, gdist_strictMax_iff hm hn]
      constructor
      · rintro ⟨hcor, ha1, ha2⟩
        exact Or.inr ⟨hon, hcor, fun _ => ⟨ha1, ha2⟩, fun h => absurd h (by omega)⟩
      · rintro (⟨hr', _⟩ | ⟨_, hcor, himpl, _⟩)
        · exact absurd hr' hon
        · obtain ⟨ha1, ha2⟩ := himpl hlt
          exact ⟨hcor, ha1, ha2⟩
    · rw [cpe_strictMax_iff_cone_right hδ hgt, gdist_strictMax_iff hm hn]
      constructor
      · rintro ⟨hcor, ha1, ha2⟩
        exact Or.inr ⟨hon, hcor, fun h => absurd h (by omega), fun _ => ⟨ha1, ha2⟩⟩
      · rintro (⟨hr', _⟩ | ⟨_, hcor, _, himpl⟩)
        · exact absurd hr' hon
        · obtain ⟨ha1, ha2⟩ := himpl hgt
          exact ⟨hcor, ha1, ha2⟩

end OrigamiCone
