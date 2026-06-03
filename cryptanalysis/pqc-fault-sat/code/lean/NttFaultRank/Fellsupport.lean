/-
Phase F3 — F_ell support formula (paper Lemma `lem:Fellsupport`,
main.tex lines 491–544).

Headline (paper):
  F_ℓ(e_j) = Σ_{h ∈ {0,1}^ℓ} e_{[h | 1 | s]},
for every j = 2^{β_ℓ} + s ∈ I_ℓ, every one-per-layer selection, and
every twiddle assignment.

Mechanisation strategy.

The paper tracks σ_k = (L*_{k-1} ∘ ⋯ ∘ L*_0)(e_j) via a bit-level
support formula. In Lean we replace the bit decomposition by a
single flat-index mod-arithmetic invariant:

    σ_k(i) = 1   if i.val % (2 L_k) = j.val
           = 0   otherwise,

valid for every 0 ≤ k ≤ ℓ.val, every twiddle assignment `Z` (faulted
or not), and every basis index `j` with `j.val < 2 L_ℓ`.

Twiddle independence at the step k → k+1: at every layer-k butterfly
pair (i, partner_k i) intersecting σ_k's support, the b-side input is
zero. With (a, b) = (·, 0), both unfaulted (a, b) ↦ (a + z b, a − z b)
and faulted (a, b) ↦ (a, a) send (1, 0) to (1, 1), independent of z.
Hence the running σ_k is a {0, 1}-indicator at every step.

The arithmetic glue uses the Kyber recurrence `L_{k+1} · 2 = L_k`
(`KyberLike.L_succ`). The single iterated fact needed is the
antitonicity of `spec.L` along `Fin spec.N`, proved below.
-/

import NttFaultRank.Theorem2Subset

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Arithmetic glue: antitonicity of `spec.L` -/

/-- `spec.L` is antitone along `Fin spec.N` for a `KyberLike` spec:
    if `k ≤ k'`, then `spec.L ⟨k', _⟩ ≤ spec.L ⟨k, _⟩`. Proved by
    iterating `KyberLike.L_succ` (`L_{k+1} · 2 = L_k`). -/
lemma L_antitone [KyberLike spec] :
    ∀ {k' : ℕ} (hk' : k' < spec.N) {k : ℕ} (hk : k < spec.N), k ≤ k' →
      spec.L ⟨k', hk'⟩ ≤ spec.L ⟨k, hk⟩ := by
  intro k'
  induction k' with
  | zero =>
    intro hk' k hk hkk'
    have : k = 0 := Nat.le_zero.mp hkk'
    subst this
    exact le_refl _
  | succ k' ih =>
    intro hk' k hk hkk'
    by_cases hkek' : k = k' + 1
    · subst hkek'; exact le_refl _
    · have hk_le_k' : k ≤ k' := by omega
      have hk'_lt : k' < spec.N := Nat.lt_of_succ_lt hk'
      have hLs := KyberLike.L_succ (spec := spec) k' hk'
      have ih' := ih hk'_lt hk hk_le_k'
      omega

/-- For `k < k'` in `Fin spec.N`, `2 * spec.L k' ≤ spec.L k`. Combines
    `L_succ` at index `k' − 1` with antitonicity from `k` to `k' − 1`. -/
lemma two_L_le_L_of_lt [KyberLike spec]
    {k k' : ℕ} (hk : k < spec.N) (hk' : k' < spec.N) (hkk' : k < k') :
    2 * spec.L ⟨k', hk'⟩ ≤ spec.L ⟨k, hk⟩ := by
  match k', hk', hkk' with
  | 0, _, hkk' => exact absurd hkk' (by omega)
  | k'' + 1, hk', hkk' =>
    have hk'' : k'' < spec.N := Nat.lt_of_succ_lt hk'
    have hk_le_k'' : k ≤ k'' := Nat.lt_succ_iff.mp hkk'
    have hLs := KyberLike.L_succ (spec := spec) k'' hk'
    have hanti := spec.L_antitone hk'' hk hk_le_k''
    omega

/-! ### Layer-apply formulas for the twiddle-independence step -/

/-- `layer Z ℓ v i = v i` when `i` is on the a-side of its layer-`ℓ`
    butterfly and the partner's value is zero. Mirrors
    `layer_eq_zero_of_partner_zero` (LayerFlat.lean). -/
lemma layer_apply_aside_of_partner_zero
    (Z : spec.Twiddles K) (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (i : Fin spec.n) (hi : i.val % (2 * spec.L ℓ) < spec.L ℓ)
    (hp : v (spec.partnerIdx ℓ i) = 0) :
    spec.layer Z ℓ v i = v i := by
  set p := coordEquiv (spec.G ℓ) (spec.L ℓ)
    ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i) with hp_def
  have hround : spec.flatIdx ℓ p.1 p.2.1 p.2.2 = i := spec.flatIdx_coordEquiv ℓ i
  have hmod_i := spec.val_mod_2L ℓ i
  rw [← hp_def] at hmod_i
  have hL_pos : 0 < spec.L ℓ := by
    rcases Nat.eq_zero_or_pos (spec.L ℓ) with h0 | hpos
    · rw [h0] at hi; omega
    · exact hpos
  have hs_eq_zero : p.2.1.val = 0 := by
    have hj_lt : p.2.2.val < spec.L ℓ := p.2.2.isLt
    have hsum : p.2.1.val * spec.L ℓ + p.2.2.val < spec.L ℓ := hmod_i ▸ hi
    nlinarith
  have hs0 : p.2.1 = (0 : Fin 2) := by apply Fin.ext; rw [hs_eq_zero]; rfl
  have hpartner : spec.flatIdx ℓ p.1 1 p.2.2 = spec.partnerIdx ℓ i := by
    apply Fin.ext
    rw [spec.flatIdx_val, spec.partnerIdx_val_aside hi]
    have hi_val := spec.val_eq_flatIdx_val ℓ i
    rw [← hp_def] at hi_val
    rw [hi_val, hs_eq_zero]
    show p.1.val * (2 * spec.L ℓ) + (1 : Fin 2).val * spec.L ℓ + p.2.2.val
       = p.1.val * (2 * spec.L ℓ) + 0 * spec.L ℓ + p.2.2.val + spec.L ℓ
    have : (1 : Fin 2).val = 1 := rfl
    rw [this]; ring
  show layerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z ℓ) v i = v i
  unfold layerOn
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe]
  show fullLayer' (spec.G ℓ) (spec.L ℓ) (Z ℓ)
        ((funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) v)
        ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i) = v i
  unfold fullLayer'
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, reshape_symm_apply]
  rw [fullLayer_apply, hs0, groupLayer_apply_zero,
      reshape_funReindex_apply, reshape_funReindex_apply]
  have h1 : spec.flatIdx ℓ p.1 0 p.2.2 = i := by rw [← hs0]; exact hround
  rw [h1, hpartner, hp]; ring

/-- `layer Z ℓ v i = v (partnerIdx ℓ i)` when `i` is on the b-side of
    its layer-`ℓ` butterfly and its own value is zero. -/
lemma layer_apply_bside_of_self_zero
    (Z : spec.Twiddles K) (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (i : Fin spec.n) (hi : ¬ i.val % (2 * spec.L ℓ) < spec.L ℓ)
    (h0 : v i = 0) :
    spec.layer Z ℓ v i = v (spec.partnerIdx ℓ i) := by
  set p := coordEquiv (spec.G ℓ) (spec.L ℓ)
    ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i) with hp_def
  have hround : spec.flatIdx ℓ p.1 p.2.1 p.2.2 = i := spec.flatIdx_coordEquiv ℓ i
  have hmod_i := spec.val_mod_2L ℓ i
  rw [← hp_def] at hmod_i
  have hL_pos : 0 < spec.L ℓ := by
    rcases Nat.eq_zero_or_pos (spec.L ℓ) with h0' | hpos
    · exfalso
      have hgl := spec.hGL ℓ
      have hn_pos : 0 < spec.n := lt_of_le_of_lt (Nat.zero_le _) i.isLt
      rw [h0'] at hgl; simp at hgl; omega
    · exact hpos
  have hs_eq_one : p.2.1.val = 1 := by
    have hj_lt : p.2.2.val < spec.L ℓ := p.2.2.isLt
    have hnot : ¬ p.2.1.val * spec.L ℓ + p.2.2.val < spec.L ℓ := hmod_i ▸ hi
    rcases fin2_cases p.2.1 with hs0' | hs1'
    · exfalso; apply hnot; rw [hs0']; show 0 * spec.L ℓ + p.2.2.val < spec.L ℓ
      simpa using hj_lt
    · rw [hs1']; rfl
  have hs1 : p.2.1 = (1 : Fin 2) := by apply Fin.ext; rw [hs_eq_one]; rfl
  have hpartner : spec.flatIdx ℓ p.1 0 p.2.2 = spec.partnerIdx ℓ i := by
    apply Fin.ext
    rw [spec.flatIdx_val, spec.partnerIdx_val_bside hi]
    have hi_val := spec.val_eq_flatIdx_val ℓ i
    rw [← hp_def] at hi_val
    rw [hi_val, hs_eq_one]
    show p.1.val * (2 * spec.L ℓ) + (0 : Fin 2).val * spec.L ℓ + p.2.2.val
       = p.1.val * (2 * spec.L ℓ) + 1 * spec.L ℓ + p.2.2.val - spec.L ℓ
    have : (0 : Fin 2).val = 0 := rfl
    rw [this]; omega
  show layerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z ℓ) v i = v _
  unfold layerOn
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe]
  show fullLayer' (spec.G ℓ) (spec.L ℓ) (Z ℓ)
        ((funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) v)
        ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i)
      = v (spec.partnerIdx ℓ i)
  unfold fullLayer'
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe, reshape_symm_apply]
  rw [fullLayer_apply, hs1, groupLayer_apply_one,
      reshape_funReindex_apply, reshape_funReindex_apply]
  have h1 : spec.flatIdx ℓ p.1 1 p.2.2 = i := by rw [← hs1]; exact hround
  rw [h1, hpartner, h0]; ring

/-! ### Main value-indicator theorem -/

/-- **F3 main lemma.** For a fixed layer `ℓ`, a basis vector `e_j` with
    `j.val < 2 L_ℓ`, and any twiddle assignment `Z` (faulted or not),
    the `k`-step running image is the {0,1}-indicator of the mod-residue
    class of `j`:

      `(runPrefix Z k (basis j)) i = 1`  iff  `i.val % (2 L_k) = j.val`,

    for every `0 ≤ k ≤ ℓ.val`. Twiddle-independent: same formula for `Z`
    and for `faultedTwiddles Z F`. -/
theorem runPrefix_basis_value_indicator [KyberLike spec]
    (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (j : Fin spec.n) (hj_lt : j.val < 2 * spec.L ℓ)
    (k : ℕ) :
    ∀ (hk : k ≤ ℓ.val) (i : Fin spec.n),
    spec.runPrefix Z k (le_trans hk (Nat.le_of_lt ℓ.isLt))
        (spec.basis (K := K) j) i
      = if i.val % (2 * spec.L ⟨k, lt_of_le_of_lt hk ℓ.isLt⟩) = j.val
        then (1 : K) else 0 := by
  induction k with
  | zero =>
    intro hk i
    have h0lt : (0 : ℕ) < spec.N := lt_of_le_of_lt (Nat.zero_le _) ℓ.isLt
    have hL0 : 2 * spec.L ⟨0, h0lt⟩ = spec.n :=
      KyberLike.L_zero_doubled (spec := spec) h0lt
    have hi_mod : i.val % (2 * spec.L ⟨0, h0lt⟩) = i.val := by
      rw [hL0]; exact Nat.mod_eq_of_lt i.isLt
    have hrun : spec.runPrefix Z 0 (Nat.zero_le _) (spec.basis (K := K) j)
                  = spec.basis (K := K) j := by
      simp [runPrefix]
    rw [hrun, hi_mod]
    unfold basis
    by_cases hij : i = j
    · rw [if_pos hij, if_pos]; rw [hij]
    · rw [if_neg hij, if_neg]
      intro habs
      apply hij; apply Fin.ext; exact habs
  | succ k ih =>
    intro hk i
    have hk1_le : k + 1 ≤ ℓ.val := hk
    have hkN : k < spec.N := lt_of_lt_of_le (Nat.lt_succ_self k)
      (le_trans hk1_le (Nat.le_of_lt ℓ.isLt))
    have hk1N : k + 1 < spec.N := lt_of_le_of_lt hk1_le ℓ.isLt
    have hk_le : k ≤ ℓ.val := Nat.le_of_succ_le hk
    set hkN' : k + 1 ≤ spec.N := le_trans hk1_le (Nat.le_of_lt ℓ.isLt) with hkN'_def
    set ℓ_k : Fin spec.N := ⟨k, hkN'⟩ with hℓk_def
    have hLrec : 2 * spec.L ⟨k + 1, hk1N⟩ = spec.L ℓ_k := by
      have h := KyberLike.L_succ (spec := spec) k hk1N
      simp only [hℓk_def]; omega
    have hk_lt_ℓ : k < ℓ.val := hk1_le
    have hjLk : j.val < spec.L ℓ_k := by
      have hbound := spec.two_L_le_L_of_lt (k := k) (k' := ℓ.val)
        hkN ℓ.isLt hk_lt_ℓ
      have hellEq : (⟨ℓ.val, ℓ.isLt⟩ : Fin spec.N) = ℓ := Fin.eta ℓ ℓ.isLt
      rw [hellEq] at hbound
      show j.val < spec.L ⟨k, hkN'⟩
      omega
    have ih_i := ih hk_le i
    have ih_p := ih hk_le (spec.partnerIdx ℓ_k i)
    -- (use them below)
    rw [spec.runPrefix_succ Z k hkN']
    simp only [LinearMap.comp_apply]
    -- (⟨k, hkN'⟩ = ℓ_k by definition of ℓ_k).
    set v := spec.runPrefix Z k (Nat.le_of_succ_le hkN')
              (spec.basis (K := K) j) with hv_def
    have hℓk_alt : (⟨k, lt_of_le_of_lt hk_le ℓ.isLt⟩ : Fin spec.N) = ℓ_k := by
      apply Fin.ext; rfl
    have ih_i' : v i = if i.val % (2 * spec.L ℓ_k) = j.val then (1 : K) else 0 := by
      rw [hv_def, ← hℓk_alt]; exact ih_i
    have ih_p' : v (spec.partnerIdx ℓ_k i)
                  = if (spec.partnerIdx ℓ_k i).val % (2 * spec.L ℓ_k) = j.val
                    then (1 : K) else 0 := by
      rw [hv_def, ← hℓk_alt]; exact ih_p
    by_cases hi_side : i.val % (2 * spec.L ℓ_k) < spec.L ℓ_k
    · -- a-side branch.
      have h2L_pos : 0 < 2 * spec.L ℓ_k := by omega
      have hL_pos : 0 < spec.L ℓ_k := by omega
      have hpartner_mod : (spec.partnerIdx ℓ_k i).val % (2 * spec.L ℓ_k)
                            = i.val % (2 * spec.L ℓ_k) + spec.L ℓ_k := by
        rw [spec.partnerIdx_val_aside hi_side]
        have hsplit : i.val = (2 * spec.L ℓ_k) * (i.val / (2 * spec.L ℓ_k))
                              + i.val % (2 * spec.L ℓ_k) := (Nat.div_add_mod _ _).symm
        set r := i.val % (2 * spec.L ℓ_k) with hr_def
        set q := i.val / (2 * spec.L ℓ_k) with hq_def
        have hr_sum_lt : r + spec.L ℓ_k < 2 * spec.L ℓ_k := by omega
        calc (i.val + spec.L ℓ_k) % (2 * spec.L ℓ_k)
            = ((2 * spec.L ℓ_k) * q + r + spec.L ℓ_k) % (2 * spec.L ℓ_k) := by
              conv_lhs => rw [hsplit]
          _ = (r + spec.L ℓ_k + (2 * spec.L ℓ_k) * q) % (2 * spec.L ℓ_k) := by
              congr 1; ring
          _ = (r + spec.L ℓ_k) % (2 * spec.L ℓ_k) := by
              rw [Nat.add_mul_mod_self_left]
          _ = r + spec.L ℓ_k := Nat.mod_eq_of_lt hr_sum_lt
      have hp_value : v (spec.partnerIdx ℓ_k i) = 0 := by
        rw [ih_p', if_neg]
        rw [hpartner_mod]; intro habs; omega
      rw [spec.layer_apply_aside_of_partner_zero Z ℓ_k v i hi_side hp_value, ih_i']
      congr 1
      rw [hLrec]
      have hdvd : spec.L ℓ_k ∣ 2 * spec.L ℓ_k := ⟨2, by ring⟩
      have hmm : i.val % spec.L ℓ_k = (i.val % (2 * spec.L ℓ_k)) % spec.L ℓ_k :=
        (Nat.mod_mod_of_dvd _ hdvd).symm
      rw [hmm, Nat.mod_eq_of_lt hi_side]
    · -- b-side branch.
      push_neg at hi_side
      have h2L_pos : 0 < 2 * spec.L ℓ_k := by
        rcases Nat.eq_zero_or_pos (2 * spec.L ℓ_k) with h0 | hpos
        · exfalso; rw [h0] at hi_side; omega
        · exact hpos
      have hL_pos : 0 < spec.L ℓ_k := by omega
      have hi_not : ¬ i.val % (2 * spec.L ℓ_k) < spec.L ℓ_k := by omega
      have hi_value : v i = 0 := by
        rw [ih_i', if_neg]; intro habs; omega
      rw [spec.layer_apply_bside_of_self_zero Z ℓ_k v i hi_not hi_value, ih_p']
      congr 1
      have hpartner_mod : (spec.partnerIdx ℓ_k i).val % (2 * spec.L ℓ_k)
                            = i.val % (2 * spec.L ℓ_k) - spec.L ℓ_k := by
        rw [spec.partnerIdx_val_bside hi_not]
        have hsplit : i.val = (2 * spec.L ℓ_k) * (i.val / (2 * spec.L ℓ_k))
                              + i.val % (2 * spec.L ℓ_k) := (Nat.div_add_mod _ _).symm
        set r := i.val % (2 * spec.L ℓ_k) with hr_def
        set q := i.val / (2 * spec.L ℓ_k) with hq_def
        have hr_lt : r < 2 * spec.L ℓ_k := Nat.mod_lt _ h2L_pos
        have hi_ge : spec.L ℓ_k ≤ i.val := by rw [hsplit]; omega
        have hdiff_lt : r - spec.L ℓ_k < spec.L ℓ_k := by omega
        calc (i.val - spec.L ℓ_k) % (2 * spec.L ℓ_k)
            = ((2 * spec.L ℓ_k) * q + (r - spec.L ℓ_k)) % (2 * spec.L ℓ_k) := by
              congr 1; omega
          _ = ((r - spec.L ℓ_k) + (2 * spec.L ℓ_k) * q) % (2 * spec.L ℓ_k) := by
              congr 1; ring
          _ = (r - spec.L ℓ_k) % (2 * spec.L ℓ_k) := by
              rw [Nat.add_mul_mod_self_left]
          _ = r - spec.L ℓ_k := Nat.mod_eq_of_lt (by omega)
      rw [hpartner_mod, hLrec]
      have hdvd : spec.L ℓ_k ∣ 2 * spec.L ℓ_k := ⟨2, by ring⟩
      have hmm : i.val % spec.L ℓ_k = (i.val % (2 * spec.L ℓ_k)) % spec.L ℓ_k :=
        (Nat.mod_mod_of_dvd _ hdvd).symm
      rw [hmm]
      have hr_ge : spec.L ℓ_k ≤ i.val % (2 * spec.L ℓ_k) := hi_side
      have hr_lt : i.val % (2 * spec.L ℓ_k) < 2 * spec.L ℓ_k := Nat.mod_lt _ h2L_pos
      have : (i.val % (2 * spec.L ℓ_k)) % spec.L ℓ_k
              = i.val % (2 * spec.L ℓ_k) - spec.L ℓ_k := by
        have hgt : i.val % (2 * spec.L ℓ_k) - spec.L ℓ_k < spec.L ℓ_k := by omega
        have hadd : i.val % (2 * spec.L ℓ_k) - spec.L ℓ_k + spec.L ℓ_k
                      = i.val % (2 * spec.L ℓ_k) := by omega
        conv_lhs => rw [← hadd]
        rw [Nat.add_mod, Nat.mod_self, Nat.add_zero, Nat.mod_mod,
            Nat.mod_eq_of_lt hgt]
      rw [this]

/-! ### Named target: F_ℓ support pattern (paper headline) -/

/-- The support pattern of `F_ℓ(e_j)` for `j` with `j.val < 2 L_ℓ`:
    the set of flat indices `i` with `i.val % (2 L_ℓ) = j.val`. -/
def fellPattern (ℓ : Fin spec.N) (j : Fin spec.n) : Finset (Fin spec.n) :=
  Finset.univ.filter (fun i => i.val % (2 * spec.L ℓ) = j.val)

/-- **Headline (paper Lemma `lem:Fellsupport`).** For every layer `ℓ`,
    every basis index `j` with `j.val < 2 L_ℓ`, and every twiddle
    assignment `Z` (faulted or unfaulted), the value of `F_ℓ(e_j) :=
    runPrefix Z ℓ.val (basis j)` at flat index `i` is the {0,1}-indicator
    of `fellPattern ℓ j`.

    Twiddle-independent: applies in particular to `faultedTwiddles Z F`
    for any one-per-layer fault selection `F`. -/
theorem fellPrefix_basis_eq_pattern [KyberLike spec]
    (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (j : Fin spec.n) (hj_lt : j.val < 2 * spec.L ℓ)
    (i : Fin spec.n) :
    spec.runPrefix Z ℓ.val (Nat.le_of_lt ℓ.isLt) (spec.basis (K := K) j) i
      = if i ∈ spec.fellPattern ℓ j then (1 : K) else 0 := by
  have hind := spec.runPrefix_basis_value_indicator Z ℓ j hj_lt ℓ.val le_rfl i
  -- bridge: indicator at k=ℓ.val matches fellPattern.
  have hellEq : (⟨ℓ.val, lt_of_le_of_lt (le_refl ℓ.val) ℓ.isLt⟩ : Fin spec.N) = ℓ :=
    Fin.eta ℓ _
  rw [hellEq] at hind
  rw [hind]
  congr 1
  unfold fellPattern
  simp [Finset.mem_filter]

end LayerSpec
end NttFaultRank
