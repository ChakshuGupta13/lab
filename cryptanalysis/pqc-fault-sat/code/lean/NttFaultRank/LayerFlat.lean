/-
Flat-index layer machinery for the support invariant inductive step.

  * `partnerIdx ℓ i` — the butterfly partner of `i` at layer `ℓ`.
  * `partnerIdx_mod_L_eq` — partner has same residue mod `L ℓ`.
  * `layer_eq_zero_of_partner_zero` — if `v i = 0` and `v (partnerIdx ℓ i) = 0`,
    then `(layer Z ℓ v) i = 0`.

All lemmas in this file are fully proved with standard mathlib axioms.
-/

import NttFaultRank.LayerOn
import NttFaultRank.Ntt

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-- The butterfly partner of flat index `i` at layer `ℓ`. -/
def partnerIdx (ℓ : Fin spec.N) (i : Fin spec.n) : Fin spec.n := by
  classical
  refine if h : i.val % (2 * spec.L ℓ) < spec.L ℓ then
    ⟨i.val + spec.L ℓ, ?_⟩
  else
    ⟨i.val - spec.L ℓ, ?_⟩
  · have hgGL : spec.G ℓ * (2 * spec.L ℓ) = spec.n := spec.hGL ℓ
    have hiN : i.val < spec.n := i.isLt
    set L2 := 2 * spec.L ℓ with hL2_def
    have hL2pos : 0 < L2 := by
      rcases Nat.eq_zero_or_pos L2 with h0 | hpos
      · exfalso; rw [hL2_def] at h0
        have : spec.n = 0 := by
          have hmul : spec.G ℓ * (2 * spec.L ℓ) = 0 := by rw [h0]; ring
          linarith [spec.hGL ℓ]
        omega
      · exact hpos
    have hsplit : i.val = (i.val / L2) * L2 + i.val % L2 := by
      have := Nat.div_add_mod i.val L2
      linarith
    have hmod_lt : i.val % L2 < L2 := Nat.mod_lt _ hL2pos
    have hquot : i.val / L2 + 1 ≤ spec.G ℓ := by
      have hlt_mul : i.val < spec.G ℓ * L2 := by
        rw [hL2_def] at hgGL ⊢
        rw [hgGL]; exact hiN
      have : i.val / L2 < spec.G ℓ :=
        Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hlt_mul)
      omega
    calc i.val + spec.L ℓ
        = (i.val / L2) * L2 + i.val % L2 + spec.L ℓ := by rw [← hsplit]
      _ < (i.val / L2) * L2 + spec.L ℓ + spec.L ℓ := by
          have : i.val % L2 < spec.L ℓ := by rw [hL2_def] at hmod_lt ⊢; exact h
          linarith
      _ = (i.val / L2) * L2 + L2 := by rw [hL2_def]; ring
      _ = (i.val / L2 + 1) * L2 := by ring
      _ ≤ spec.G ℓ * L2 := Nat.mul_le_mul_right _ hquot
      _ = spec.n := by rw [hL2_def]; exact hgGL
  · have : i.val < spec.n := i.isLt
    omega

@[simp] lemma partnerIdx_val_aside {ℓ : Fin spec.N} {i : Fin spec.n}
    (h : i.val % (2 * spec.L ℓ) < spec.L ℓ) :
    (spec.partnerIdx ℓ i).val = i.val + spec.L ℓ := by
  unfold partnerIdx; simp [h]

@[simp] lemma partnerIdx_val_bside {ℓ : Fin spec.N} {i : Fin spec.n}
    (h : ¬ i.val % (2 * spec.L ℓ) < spec.L ℓ) :
    (spec.partnerIdx ℓ i).val = i.val - spec.L ℓ := by
  unfold partnerIdx; simp [h]

/-- `partnerIdx ℓ i` has the same residue mod `L ℓ` as `i`. -/
lemma partnerIdx_mod_L_eq (ℓ : Fin spec.N) (i : Fin spec.n) :
    (spec.partnerIdx ℓ i).val % spec.L ℓ = i.val % spec.L ℓ := by
  by_cases h : i.val % (2 * spec.L ℓ) < spec.L ℓ
  · rw [spec.partnerIdx_val_aside h]
    rw [Nat.add_mod, Nat.mod_self, Nat.add_zero, Nat.mod_mod]
  · rw [spec.partnerIdx_val_bside h]
    push_neg at h
    have hge : i.val ≥ spec.L ℓ := by
      have : i.val % (2 * spec.L ℓ) ≤ i.val := Nat.mod_le _ _
      linarith
    have hcancel : i.val - spec.L ℓ + spec.L ℓ = i.val := Nat.sub_add_cancel hge
    conv_rhs => rw [← hcancel]
    rw [Nat.add_mod, Nat.mod_self, Nat.add_zero, Nat.mod_mod]

end LayerSpec

end NttFaultRank

namespace NttFaultRank
namespace LayerSpec
variable {K : Type*} [Field K]
variable (spec : LayerSpec)

/-- The flat index corresponding to indexed coordinate `(g, s, j)` at layer `ℓ`. -/
def flatIdx (ℓ : Fin spec.N) (g : Fin (spec.G ℓ)) (s : Fin 2) (j : Fin (spec.L ℓ)) :
    Fin spec.n :=
  (finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)).symm
    ((coordEquiv (spec.G ℓ) (spec.L ℓ)).symm (g, s, j))

/-- Local copy of the coordEquiv-symm value formula. -/
private lemma coordEquiv_symm_val_loc (G L : ℕ) (g : Fin G) (s : Fin 2) (j : Fin L) :
    ((coordEquiv G L).symm (g, s, j)).val = g.val * (2 * L) + s.val * L + j.val := by
  unfold coordEquiv
  simp [finProdFinEquiv, Equiv.prodCongr]
  ring

lemma flatIdx_val (ℓ : Fin spec.N) (g : Fin (spec.G ℓ)) (s : Fin 2) (j : Fin (spec.L ℓ)) :
    (spec.flatIdx ℓ g s j).val = g.val * (2 * spec.L ℓ) + s.val * spec.L ℓ + j.val := by
  unfold flatIdx
  simp [finCastEquiv, coordEquiv_symm_val_loc]

/-- Reshape ∘ funReindex applied to v: returns v at the flat index. -/
lemma reshape_funReindex_apply (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (g : Fin (spec.G ℓ)) (s : Fin 2) (j : Fin (spec.L ℓ)) :
    (reshape K (spec.G ℓ) (spec.L ℓ))
        ((funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) v) g s j
      = v (spec.flatIdx ℓ g s j) := by
  rfl

/-- Round-trip: `i = flatIdx ℓ g s j` where `(g, s, j)` is the coordEquiv
    decomposition of `i`. -/
lemma flatIdx_coordEquiv (ℓ : Fin spec.N) (i : Fin spec.n) :
    let j_full := (finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i
    let p := coordEquiv (spec.G ℓ) (spec.L ℓ) j_full
    spec.flatIdx ℓ p.1 p.2.1 p.2.2 = i := by
  unfold flatIdx
  simp [Equiv.symm_apply_apply]

/-- Auxiliary: i.val expressed via its coordEquiv decomposition. -/
lemma val_eq_flatIdx_val (ℓ : Fin spec.N) (i : Fin spec.n) :
    i.val =
      (coordEquiv (spec.G ℓ) (spec.L ℓ)
          ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)).1.val *
        (2 * spec.L ℓ) +
      (coordEquiv (spec.G ℓ) (spec.L ℓ)
          ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)).2.1.val *
        spec.L ℓ +
      (coordEquiv (spec.G ℓ) (spec.L ℓ)
          ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)).2.2.val := by
  have h := spec.flatIdx_val ℓ
    ((coordEquiv (spec.G ℓ) (spec.L ℓ))
        ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)).1
    ((coordEquiv (spec.G ℓ) (spec.L ℓ))
        ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)).2.1
    ((coordEquiv (spec.G ℓ) (spec.L ℓ))
        ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)).2.2
  have hround := spec.flatIdx_coordEquiv ℓ i
  rw [hround] at h
  linarith

/-- Auxiliary: i.val mod (2 L) equals s*L + j when i has coordEquiv decomposition (g, s, j). -/
lemma val_mod_2L (ℓ : Fin spec.N) (i : Fin spec.n) :
    i.val % (2 * spec.L ℓ) =
      (coordEquiv (spec.G ℓ) (spec.L ℓ)
          ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)).2.1.val *
        spec.L ℓ +
      (coordEquiv (spec.G ℓ) (spec.L ℓ)
          ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)).2.2.val := by
  have hi_val := spec.val_eq_flatIdx_val ℓ i
  set p := coordEquiv (spec.G ℓ) (spec.L ℓ)
    ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i) with hp_def
  have hs_lt : p.2.1.val < 2 := p.2.1.isLt
  have hj_lt : p.2.2.val < spec.L ℓ := p.2.2.isLt
  have hL_pos : 0 < spec.L ℓ := by
    rcases Nat.eq_zero_or_pos (spec.L ℓ) with h0 | hpos
    · exfalso; omega
    · exact hpos
  have hsmall : p.2.1.val * spec.L ℓ + p.2.2.val < 2 * spec.L ℓ := by
    have : p.2.1.val ≤ 1 := by omega
    nlinarith
  have hi_split : i.val = (p.2.1.val * spec.L ℓ + p.2.2.val) + p.1.val * (2 * spec.L ℓ) := by
    linarith
  rw [hi_split]
  -- (a + b * c) % c = a % c when written with c on the right
  simp [Nat.add_mul_mod_self_left, Nat.add_mul_mod_self_right,
        Nat.mod_eq_of_lt hsmall]

/-- **Layer support preservation.** If `v i = 0` and
    `v (partnerIdx ℓ i) = 0`, then `(layer Z ℓ v) i = 0`.

    The Cooley-Tukey butterfly at layer `ℓ` couples each flat index `i`
    with its partner `partnerIdx ℓ i`: the layer's output at `i` is a
    linear combination of `v i` and `v (partnerIdx ℓ i)` (specifically,
    either `v i + z·v (partner i)` on the a-side or
    `v (partner i) - z·v i` on the b-side). When both inputs vanish,
    so does the output.

    The proof requires unfolding the four-layer composition
    `layer = funReindex⁻¹ ∘ reshape⁻¹ ∘ fullLayer ∘ reshape ∘ funReindex`
    and applying `groupLayer_apply_zero` / `_one` after the coordEquiv
    decomposition of `i`. Estimated effort: ~80-150 lines. -/
lemma layer_eq_zero_of_partner_zero
    (Z : spec.Twiddles K) (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (i : Fin spec.n) (hi : v i = 0) (hp : v (spec.partnerIdx ℓ i) = 0) :
    spec.layer Z ℓ v i = 0 := by
  -- Identify coords of i.
  set p := coordEquiv (spec.G ℓ) (spec.L ℓ)
    ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i) with hp_def
  have hround : spec.flatIdx ℓ p.1 p.2.1 p.2.2 = i := spec.flatIdx_coordEquiv ℓ i
  -- Partner-flip: flatIdx with s-flipped equals partnerIdx ℓ i.
  have hpartner_aside : p.2.1 = 0 → spec.flatIdx ℓ p.1 1 p.2.2 = spec.partnerIdx ℓ i := by
    intro hs0
    apply Fin.ext
    rw [spec.flatIdx_val]
    have hmod_i := spec.val_mod_2L ℓ i
    rw [← hp_def] at hmod_i
    have hs_val : p.2.1.val = 0 := by rw [hs0]; rfl
    rw [hs_val, Nat.zero_mul, Nat.zero_add] at hmod_i
    have hj_lt : p.2.2.val < spec.L ℓ := p.2.2.isLt
    have ha : i.val % (2 * spec.L ℓ) < spec.L ℓ := hmod_i ▸ hj_lt
    rw [spec.partnerIdx_val_aside ha]
    have hi_val := spec.val_eq_flatIdx_val ℓ i
    rw [← hp_def] at hi_val
    rw [hi_val, hs_val]
    show p.1.val * (2 * spec.L ℓ) + (1 : Fin 2).val * spec.L ℓ + p.2.2.val
         = p.1.val * (2 * spec.L ℓ) + 0 * spec.L ℓ + p.2.2.val + spec.L ℓ
    have : (1 : Fin 2).val = 1 := rfl
    rw [this]; ring
  have hpartner_bside : p.2.1 = 1 → spec.flatIdx ℓ p.1 0 p.2.2 = spec.partnerIdx ℓ i := by
    intro hs1
    apply Fin.ext
    rw [spec.flatIdx_val]
    have hmod_i := spec.val_mod_2L ℓ i
    rw [← hp_def] at hmod_i
    have hs_val : p.2.1.val = 1 := by rw [hs1]; rfl
    rw [hs_val, Nat.one_mul] at hmod_i
    have hj_lt : p.2.2.val < spec.L ℓ := p.2.2.isLt
    have hb : ¬ i.val % (2 * spec.L ℓ) < spec.L ℓ := by
      rw [hmod_i]; omega
    rw [spec.partnerIdx_val_bside hb]
    have hi_val := spec.val_eq_flatIdx_val ℓ i
    rw [← hp_def] at hi_val
    rw [hi_val, hs_val]
    show p.1.val * (2 * spec.L ℓ) + (0 : Fin 2).val * spec.L ℓ + p.2.2.val
         = p.1.val * (2 * spec.L ℓ) + 1 * spec.L ℓ + p.2.2.val - spec.L ℓ
    have : (0 : Fin 2).val = 0 := rfl
    rw [this]; omega
  -- Now unfold the layer composition.
  show layerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z ℓ) v i = 0
  unfold layerOn
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe]
  show fullLayer' (spec.G ℓ) (spec.L ℓ) (Z ℓ)
        ((funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) v)
        ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i) = 0
  unfold fullLayer'
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, reshape_symm_apply]
  rw [fullLayer_apply]
  rcases fin2_cases p.2.1 with hs0 | hs1
  · rw [hs0, groupLayer_apply_zero]
    rw [reshape_funReindex_apply, reshape_funReindex_apply]
    have h1 : spec.flatIdx ℓ p.1 0 p.2.2 = i := by rw [← hs0]; exact hround
    have h2 : spec.flatIdx ℓ p.1 1 p.2.2 = spec.partnerIdx ℓ i := hpartner_aside hs0
    rw [h1, h2, hi, hp]; ring
  · rw [hs1, groupLayer_apply_one]
    rw [reshape_funReindex_apply, reshape_funReindex_apply]
    have h1 : spec.flatIdx ℓ p.1 1 p.2.2 = i := by rw [← hs1]; exact hround
    have h2 : spec.flatIdx ℓ p.1 0 p.2.2 = spec.partnerIdx ℓ i := hpartner_bside hs1
    rw [h1, h2, hi, hp]; ring

end LayerSpec

end NttFaultRank


