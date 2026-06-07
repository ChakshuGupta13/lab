/-
Phase F7 — Inverse-NTT image of `teleTerm` lies in `V_ℓ`.

Paper Lemma `lem:Vcontain` (main.tex lines ~403–425). We prove that
`INTT(im T_ℓ) ⊆ V_ℓ` for every `ℓ`.

  Strategy (paper-aligned):
    1. The factorisation `ntt Z = suffix ∘ layer_k ∘ prefix` (F5
       `ntt_factor_at`) gives `nttInv = prefix⁻¹ ∘ layer_k⁻¹ ∘ suffix⁻¹`.
    2. Composing with `teleTerm Z Z' k = suffix ∘ pert ∘ runPrefix Z' k`
       cancels the suffix: `nttInv ∘ teleTerm = prefix⁻¹ ∘ layer_k⁻¹ ∘
       pert ∘ runPrefix Z' k`.
    3. The image of `pert ∘ runPrefix Z' k` is supported only at the
       faulted layer-`k` group `g₀`, with antisymmetric `(z·u, −z·u)`
       pattern.
    4. `layer_k⁻¹` sends `(z·u, −z·u)` to `(0, u)`, so the result is
       supported on b-side of `g₀`, i.e. on indices with `bitOf k i = 1`.
    5. The inverse-prefix layers `L_0⁻¹, …, L_{k-1}⁻¹` couple positions
       differing in higher bits `β_0, …, β_{k-1}` (all > β_k), so they
       preserve the bit-β_k value. Hence the result stays in `V_k`.
-/

import NttFaultRank.Vsubspaces
import NttFaultRank.TeleTermRank

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Step 1 — `ntt`/`runPrefix` as `LinearEquiv` and inverses -/

/-- Composition of `Fin.foldl`-style layer chains is bijective when each
    factor is bijective. -/
lemma fin_foldl_layers_bijective {n : ℕ}
    (f : Fin n → ((Fin spec.n → K) →ₗ[K] (Fin spec.n → K)))
    (hf : ∀ i, Function.Bijective (f i)) :
    Function.Bijective
      (Fin.foldl n (fun acc i => (f i).comp acc) (LinearMap.id) :
        (Fin spec.n → K) →ₗ[K] _) := by
  induction n with
  | zero =>
    show Function.Bijective (Fin.foldl 0 _ LinearMap.id : _ →ₗ[K] _)
    simp [Fin.foldl_zero]
    exact Function.bijective_id
  | succ n ih =>
    rw [Fin.foldl_succ_last]
    -- Goal: Bijective ((f (Fin.last n)).comp (Fin.foldl n (...) id))
    have hih := ih (fun i => f i.castSucc) (fun i => hf _)
    have h_last := hf (Fin.last n)
    exact h_last.comp hih

/-- `runPrefix Z k _` is bijective when twiddles at the first `k` layers
    are nonzero (and `2 ≠ 0`). -/
lemma runPrefix_bijective (Z : spec.Twiddles K) (h2 : (2 : K) ≠ 0)
    (k : ℕ) (hk : k ≤ spec.N)
    (hZ : ∀ i : Fin k, ∀ g, Z (i.castLE hk) g ≠ 0) :
    Function.Bijective (spec.runPrefix Z k hk) := by
  unfold runPrefix
  apply spec.fin_foldl_layers_bijective
  intro i
  exact layerOn_bijective spec.n (spec.G (i.castLE hk)) (spec.L (i.castLE hk))
    (hZ i) h2

/-- `ntt Z` is bijective when all twiddles are nonzero. -/
lemma ntt_bijective {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0) : Function.Bijective (spec.ntt Z) := by
  rw [spec.ntt_eq_runPrefix_last]
  exact spec.runPrefix_bijective Z h2 spec.N le_rfl (fun i g => hZ _ g)

/-- `runPrefix Z k _` as a `LinearEquiv`. -/
noncomputable def runPrefixEquiv (Z : spec.Twiddles K)
    (h2 : (2 : K) ≠ 0) (k : ℕ) (hk : k ≤ spec.N)
    (hZ : ∀ i : Fin k, ∀ g, Z (i.castLE hk) g ≠ 0) :
    (Fin spec.n → K) ≃ₗ[K] (Fin spec.n → K) :=
  LinearEquiv.ofBijective (spec.runPrefix Z k hk)
    (spec.runPrefix_bijective Z h2 k hk hZ)

/-- `ntt Z` as a `LinearEquiv` when all twiddles are nonzero. -/
noncomputable def nttEquiv {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0) : (Fin spec.n → K) ≃ₗ[K] (Fin spec.n → K) :=
  LinearEquiv.ofBijective (spec.ntt Z) (spec.ntt_bijective hZ h2)

/-- Inverse NTT. -/
noncomputable def nttInv {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0) : (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  (spec.nttEquiv hZ h2).symm.toLinearMap

/-! ### Step 2 — Suffix cancellation

  `nttInv ∘ teleTerm Z Z' k = invPrefix ∘ layerInv_k ∘ pert ∘ runPrefix Z' k`

  Proof outline: by `ntt_factor_at` (F5), `ntt Z = suffix ∘ layer_k ∘
  runPrefix Z k`. So `nttEquiv⁻¹` applied to `suffix(v)` gives
  `runPrefix⁻¹(layer⁻¹(v))`, which is exactly `invPrefix ∘ layerInv_k`. -/

/-- Inverse of `layer Z ⟨k,_⟩` as `LinearMap`. -/
noncomputable def layerInv (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (hZℓ : ∀ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  (spec.layerEquiv Z ℓ hZℓ h2).symm.toLinearMap

/-- Inverse of `runPrefix Z k _` as `LinearMap`. -/
noncomputable def runPrefixInv (Z : spec.Twiddles K) (h2 : (2 : K) ≠ 0)
    (k : ℕ) (hk : k ≤ spec.N)
    (hZ : ∀ i : Fin k, ∀ g, Z (i.castLE hk) g ≠ 0) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  (spec.runPrefixEquiv Z h2 k hk hZ).symm.toLinearMap

/-- **Suffix cancellation.** For any `v`,
    `nttInv (suffix v) = runPrefixInv (layerInv v)`. -/
lemma nttInv_suffix_apply
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (k : ℕ) (hk : k < spec.N) (v : Fin spec.n → K) :
    spec.nttInv hZ h2
        ((chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) (by omega)))
          v)
      = spec.runPrefixInv Z h2 k (Nat.le_of_lt hk)
          (fun i g => hZ _ g)
          (spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 v) := by
  -- Apply nttEquiv to both sides and use ntt_factor_at.
  set hk_le : k ≤ spec.N := Nat.le_of_lt hk
  set hZ_pre : ∀ i : Fin k, ∀ g, Z (i.castLE hk_le) g ≠ 0 := fun i g => hZ _ g
  set hZ_layer : ∀ g, Z ⟨k, hk⟩ g ≠ 0 := fun g => hZ ⟨k, hk⟩ g
  -- (nttEquiv).injective : if nttEquiv (LHS) = nttEquiv (RHS), then LHS = RHS.
  apply (spec.nttEquiv hZ h2).injective
  -- LHS: nttEquiv (nttEquiv.symm (suffix v)) = suffix v (apply_symm_apply).
  -- RHS: nttEquiv (runPrefixInv (layerInv v)) = ntt Z (runPrefix⁻¹ (layer⁻¹ v))
  --    = suffix (layer (runPrefix (runPrefix⁻¹ (layer⁻¹ v))))   [ntt_factor_at]
  --    = suffix (layer (layer⁻¹ v))                              [runPrefix apply_symm]
  --    = suffix v.                                                [layer apply_symm]
  show (spec.nttEquiv hZ h2) (spec.nttInv hZ h2 _) =
       (spec.nttEquiv hZ h2) (spec.runPrefixInv Z h2 k hk_le hZ_pre
         (spec.layerInv Z ⟨k, hk⟩ hZ_layer h2 v))
  -- LHS reduces via apply_symm_apply of nttEquiv.
  show (spec.nttEquiv hZ h2)
        ((spec.nttEquiv hZ h2).symm
          ((chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) _)) v)) = _
  rw [LinearEquiv.apply_symm_apply]
  -- RHS unfolds: nttEquiv = ofBijective (ntt Z), so nttEquiv (w) = ntt Z w.
  show _ = spec.ntt Z (spec.runPrefixInv Z h2 k hk_le hZ_pre
            (spec.layerInv Z ⟨k, hk⟩ hZ_layer h2 v))
  rw [spec.ntt_factor_at Z k hk]
  show _ = (chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) _))
            ((spec.layer Z ⟨k, hk⟩) ((spec.runPrefix Z k hk_le)
              ((spec.runPrefixInv Z h2 k hk_le hZ_pre)
                ((spec.layerInv Z ⟨k, hk⟩ hZ_layer h2) v))))
  -- runPrefix Z k (runPrefixInv ...) = id, layer Z ⟨k,_⟩ (layerInv ...) = id.
  have h_pre : (spec.runPrefix Z k hk_le)
                ((spec.runPrefixInv Z h2 k hk_le hZ_pre)
                  ((spec.layerInv Z ⟨k, hk⟩ hZ_layer h2) v))
              = (spec.layerInv Z ⟨k, hk⟩ hZ_layer h2) v := by
    show (spec.runPrefixEquiv Z h2 k hk_le hZ_pre)
          ((spec.runPrefixEquiv Z h2 k hk_le hZ_pre).symm _) = _
    rw [LinearEquiv.apply_symm_apply]
  rw [h_pre]
  have h_layer : (spec.layer Z ⟨k, hk⟩)
                  ((spec.layerInv Z ⟨k, hk⟩ hZ_layer h2) v) = v := by
    show (spec.layerEquiv Z ⟨k, hk⟩ hZ_layer h2)
          ((spec.layerEquiv Z ⟨k, hk⟩ hZ_layer h2).symm v) = _
    rw [LinearEquiv.apply_symm_apply]
  rw [h_layer]

/-! ### Step 3 — Membership characterisation of `V ℓ` via bit-vanishing

  `v ∈ V ℓ` iff `v i = 0` for every `i` with `bitOf ℓ i = 0`. We prove
  only the direction we need (⇐). -/

/-- Any vector whose support is contained in `{i : bitOf ℓ i = 1}` lies
    in `V ℓ`. Used in Steps 4 and 5. -/
lemma mem_V_of_bit_zero_vanish (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (hv : ∀ i, spec.bitOf ℓ i = 0 → v i = 0) :
    v ∈ spec.V (K := K) ℓ := by
  -- Write v as ∑ i, v i • basis i and split by bit value.
  have hsum : v = ∑ i : Fin spec.n, v i • spec.basis (K := K) i := by
    funext j
    rw [Finset.sum_apply]
    rw [Finset.sum_eq_single j]
    · show v j = v j • spec.basis (K := K) j j
      show v j = v j * spec.basis (K := K) j j
      unfold basis
      simp
    · intros i _ hij
      show v i • spec.basis (K := K) i j = 0
      show v i * spec.basis (K := K) i j = 0
      unfold basis
      rw [if_neg (fun h => hij h.symm)]
      ring
    · intro h; exact absurd (Finset.mem_univ _) h
  rw [hsum]
  apply Submodule.sum_mem
  intro i _
  by_cases hb : spec.bitOf ℓ i = 0
  · rw [hv i hb, zero_smul]; exact Submodule.zero_mem _
  · have hb1 : spec.bitOf ℓ i = 1 := by
      have h_lt : spec.bitOf ℓ i < 2 := Nat.mod_lt _ (by norm_num)
      unfold bitOf at hb ⊢
      omega
    apply Submodule.smul_mem
    apply Submodule.subset_span
    exact ⟨i, hb1, rfl⟩

/-! ### Step 4 — `layerInv ∘ pert` lands in `V ⟨k, hk⟩`

  Concrete claim: for any `w`, `(layerInv ∘ pert) w` vanishes at every
  index with `bitOf k = 0`.

  Index `i` with `bitOf k i = 0` lies on the a-side of its layer-`k`
  butterfly. The `pert` map produces an antisymmetric `(z·b, −z·b)`
  pair on the faulted group `g₀` and zero outside. After `layerInv`,
  the a-side value becomes `(a + b) / 2 = (z·b + (−z·b))/2 = 0`. -/

/-- Mirror of F5's `aSide_layer` for the b-side. -/
lemma bSide_layer (Z : spec.Twiddles K) (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.bSide (spec.layer Z ℓ v) ℓ g j
      = spec.aSide v ℓ g j - Z ℓ g * spec.bSide v ℓ g j := by
  unfold aSide bSide layer layerOn fullLayer'
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe,
    LinearEquiv.apply_symm_apply, fullLayer_apply, groupLayer_apply_one]

/-- `aSide (layerInv v) = (aSide v + bSide v) / 2`. Solves the 2×2 butterfly
    system using `aSide_layer` + `bSide_layer` together with `layer ∘ layerInv = id`. -/
lemma aSide_layerInv (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (hZℓ : ∀ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (v : Fin spec.n → K) (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.aSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j
      = (spec.aSide v ℓ g j + spec.bSide v ℓ g j) / 2 := by
  -- layer (layerInv v) = v, so aSide / bSide of v equal those of layer (layerInv v).
  have hcomp : spec.layer Z ℓ ((spec.layerInv Z ℓ hZℓ h2) v) = v := by
    show (spec.layerEquiv Z ℓ hZℓ h2) ((spec.layerEquiv Z ℓ hZℓ h2).symm v) = v
    rw [LinearEquiv.apply_symm_apply]
  -- Eq 1: aSide v = aSide (layerInv v) + Z ℓ g * bSide (layerInv v).
  have heq1 : spec.aSide v ℓ g j
            = spec.aSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j
              + Z ℓ g * spec.bSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j := by
    conv_lhs => rw [← hcomp]; rw [spec.aSide_layer Z ℓ _ g j]
  -- Eq 2: bSide v = aSide (layerInv v) - Z ℓ g * bSide (layerInv v).
  have heq2 : spec.bSide v ℓ g j
            = spec.aSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j
              - Z ℓ g * spec.bSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j := by
    conv_lhs => rw [← hcomp]; rw [spec.bSide_layer Z ℓ _ g j]
  -- Add: aSide v + bSide v = 2 * aSide (layerInv v).
  have hsum : spec.aSide v ℓ g j + spec.bSide v ℓ g j
            = 2 * spec.aSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j := by
    rw [heq1, heq2]; ring
  rw [hsum, mul_comm 2 _, mul_div_assoc, div_self h2, mul_one]

/-- `aSide (pert w) ℓ g j = (Z ℓ g − Z' ℓ g) * bSide w ℓ g j`. -/
lemma aSide_perturbation (Z : spec.Twiddles K) (F : spec.FaultSet)
    (ℓ : Fin spec.N) (w : Fin spec.n → K)
    (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.aSide (spec.perturbation Z F ℓ w) ℓ g j
      = (Z ℓ g - (spec.faultedTwiddles Z F) ℓ g) * spec.bSide w ℓ g j := by
  show spec.aSide (((spec.layer Z ℓ) - (spec.layer (spec.faultedTwiddles Z F) ℓ)) w)
        ℓ g j = _
  show spec.aSide ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddles Z F) ℓ) w)
        ℓ g j = _
  -- aSide is linear.
  have hsub : spec.aSide ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddles Z F) ℓ) w)
                ℓ g j
            = spec.aSide ((spec.layer Z ℓ) w) ℓ g j
              - spec.aSide ((spec.layer (spec.faultedTwiddles Z F) ℓ) w) ℓ g j := by
    show ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddles Z F) ℓ) w)
          (spec.aSideIdx ℓ g j)
        = _
    rw [Pi.sub_apply]; rfl
  rw [hsub, spec.aSide_layer Z ℓ w g j, spec.aSide_layer _ ℓ w g j]
  ring

/-- `bSide (pert w) ℓ g j = (Z' ℓ g − Z ℓ g) * bSide w ℓ g j`. -/
lemma bSide_perturbation (Z : spec.Twiddles K) (F : spec.FaultSet)
    (ℓ : Fin spec.N) (w : Fin spec.n → K)
    (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.bSide (spec.perturbation Z F ℓ w) ℓ g j
      = ((spec.faultedTwiddles Z F) ℓ g - Z ℓ g) * spec.bSide w ℓ g j := by
  show spec.bSide (((spec.layer Z ℓ) - (spec.layer (spec.faultedTwiddles Z F) ℓ)) w)
        ℓ g j = _
  show spec.bSide ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddles Z F) ℓ) w)
        ℓ g j = _
  have hsub : spec.bSide ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddles Z F) ℓ) w)
                ℓ g j
            = spec.bSide ((spec.layer Z ℓ) w) ℓ g j
              - spec.bSide ((spec.layer (spec.faultedTwiddles Z F) ℓ) w) ℓ g j := by
    show ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddles Z F) ℓ) w)
          (spec.bSideIdx ℓ g j)
        = _
    rw [Pi.sub_apply]; rfl
  rw [hsub, spec.bSide_layer Z ℓ w g j, spec.bSide_layer _ ℓ w g j]
  ring

/-- Recover coordinates `(g, j)` from a flat a-side index `i`
    (with `bitOf ℓ i = 0`). -/
lemma exists_aSide_decomp (ℓ : Fin spec.N) (i : Fin spec.n)
    (hbit : spec.bitOf ℓ i = 0) :
    ∃ (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)), i = spec.aSideIdx ℓ g j := by
  -- coordEquiv decomp: i ↔ (g, s, j). bitOf ℓ i = 0 means s = 0, so i = aSideIdx g j.
  set p := coordEquiv (spec.G ℓ) (spec.L ℓ)
    ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i) with hp_def
  refine ⟨p.1, p.2.2, ?_⟩
  -- We need: i = aSideIdx ℓ p.1 p.2.2.
  -- aSideIdx ℓ g j = flatIdx ℓ g 0 j, and flatIdx_coordEquiv: flatIdx ℓ p.1 p.2.1 p.2.2 = i.
  -- So we need p.2.1 = 0.
  have hround : spec.flatIdx ℓ p.1 p.2.1 p.2.2 = i := spec.flatIdx_coordEquiv ℓ i
  have hmod := spec.val_mod_2L ℓ i
  rw [← hp_def] at hmod
  -- bitOf ℓ i = 0 ⇒ i.val / L < 2 and i.val / L % 2 = 0 ⇒ i.val / L = 0... no.
  -- Actually bitOf ℓ i = 0 means (i.val / spec.L ℓ) % 2 = 0; combined with the
  -- decomposition i.val = p.1 * (2L) + p.2.1 * L + p.2.2, we get
  -- (i.val / L) % 2 = p.2.1.
  have hL_pos : 0 < spec.L ℓ := by
    rcases Nat.eq_zero_or_pos (spec.L ℓ) with hzero | hpos
    · exfalso
      have hgl := spec.hGL ℓ
      have hn_pos : 0 < spec.n := lt_of_le_of_lt (Nat.zero_le _) i.isLt
      rw [hzero] at hgl; simp at hgl; omega
    · exact hpos
  have hi_val := spec.val_eq_flatIdx_val ℓ i
  rw [← hp_def] at hi_val
  -- hi_val: i.val = p.1 * (2L) + p.2.1 * L + p.2.2
  have hj_lt : p.2.2.val < spec.L ℓ := p.2.2.isLt
  have hs_lt : p.2.1.val < 2 := p.2.1.isLt
  -- i.val / L = p.1 * 2 + p.2.1 (using p.2.2 < L).
  have hdiv : i.val / spec.L ℓ = p.1.val * 2 + p.2.1.val := by
    rw [hi_val]
    rw [show p.1.val * (2 * spec.L ℓ) + p.2.1.val * spec.L ℓ + p.2.2.val
          = p.2.2.val + (p.2.1.val + p.1.val * 2) * spec.L ℓ from by ring]
    rw [Nat.add_mul_div_right _ _ hL_pos]
    rw [Nat.div_eq_of_lt hj_lt]
    omega
  -- bitOf ℓ i = (i.val / L) % 2 = (p.1 * 2 + p.2.1) % 2 = p.2.1 % 2 = p.2.1 (since p.2.1 < 2).
  have hbitVal : spec.bitOf ℓ i = p.2.1.val := by
    unfold bitOf
    rw [hdiv]
    -- (p.1.val * 2 + p.2.1.val) % 2 = p.2.1.val.
    omega
  -- hbit : bitOf ℓ i = 0, so p.2.1.val = 0.
  have hs_zero : p.2.1 = (0 : Fin 2) := by
    apply Fin.ext
    rw [← hbitVal]; exact hbit
  -- Now: aSideIdx ℓ p.1 p.2.2 = flatIdx ℓ p.1 0 p.2.2 = (by hs_zero) = flatIdx ℓ p.1 p.2.1 p.2.2 = i.
  show i = spec.aSideIdx ℓ p.1 p.2.2
  apply Fin.ext
  rw [spec.aSideIdx_val, hi_val, hs_zero]
  -- Goal: p.1.val * (2L) + 0 * L + p.2.2.val = p.1.val * (2L) + p.2.2.val
  show p.1.val * (2 * spec.L ℓ) + (0 : Fin 2).val * spec.L ℓ + p.2.2.val
      = p.1.val * (2 * spec.L ℓ) + p.2.2.val
  simp

/-- `bSide (layerInv v) = (aSide v − bSide v) / (2 · Z ℓ g)`. Mirror of `aSide_layerInv`. -/
lemma bSide_layerInv (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (hZℓ : ∀ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (v : Fin spec.n → K) (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.bSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j
      = (spec.aSide v ℓ g j - spec.bSide v ℓ g j) / (2 * Z ℓ g) := by
  have hcomp : spec.layer Z ℓ ((spec.layerInv Z ℓ hZℓ h2) v) = v := by
    show (spec.layerEquiv Z ℓ hZℓ h2) ((spec.layerEquiv Z ℓ hZℓ h2).symm v) = v
    rw [LinearEquiv.apply_symm_apply]
  have heq1 : spec.aSide v ℓ g j
            = spec.aSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j
              + Z ℓ g * spec.bSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j := by
    conv_lhs => rw [← hcomp]; rw [spec.aSide_layer Z ℓ _ g j]
  have heq2 : spec.bSide v ℓ g j
            = spec.aSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j
              - Z ℓ g * spec.bSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j := by
    conv_lhs => rw [← hcomp]; rw [spec.bSide_layer Z ℓ _ g j]
  have hsub : spec.aSide v ℓ g j - spec.bSide v ℓ g j
            = (2 * Z ℓ g) * spec.bSide ((spec.layerInv Z ℓ hZℓ h2) v) ℓ g j := by
    rw [heq1, heq2]; ring
  have h2z : (2 * Z ℓ g) ≠ 0 := mul_ne_zero h2 (hZℓ g)
  rw [hsub, mul_comm (2 * Z ℓ g) _, mul_div_assoc, div_self h2z, mul_one]

/-! ### Step 5b — Divisibility `2·L_k ∣ L_ℓ'` for `ℓ' < k` -/

/-- Auxiliary form: parameterised on `n = k − ℓ' − 1`. -/
lemma two_L_dvd_L_aux [CooleyTukeyLike spec] :
    ∀ (n ℓ' : ℕ) (hℓ' : ℓ' < spec.N) (hk : ℓ' + n + 1 < spec.N),
      (2 * spec.L ⟨ℓ' + n + 1, hk⟩) ∣ spec.L ⟨ℓ', hℓ'⟩
  | 0, ℓ', hℓ', hk => by
      have hk1 : ℓ' + 1 < spec.N := by omega
      have h := CooleyTukeyLike.L_succ (spec := spec) ℓ' hk1
      have hidx : (⟨ℓ' + 0 + 1, hk⟩ : Fin spec.N) = ⟨ℓ' + 1, hk1⟩ := by
        apply Fin.ext
        show ℓ' + 0 + 1 = ℓ' + 1
        ring
      rw [hidx]
      refine ⟨1, ?_⟩
      have hidx' : (⟨ℓ', Nat.lt_of_succ_lt hk1⟩ : Fin spec.N) = ⟨ℓ', hℓ'⟩ := rfl
      rw [← hidx', ← h]; ring
  | n+1, ℓ', hℓ', hk => by
      have hℓ'1 : ℓ' + 1 < spec.N := by omega
      have h1 := two_L_dvd_L_aux n (ℓ' + 1) hℓ'1 (by omega)
      have hidx : (⟨(ℓ'+1) + n + 1, by omega⟩ : Fin spec.N)
                = ⟨ℓ' + (n+1) + 1, hk⟩ := by
        apply Fin.ext
        show (ℓ'+1) + n + 1 = ℓ' + (n+1) + 1
        ring
      rw [hidx] at h1
      have hLs := CooleyTukeyLike.L_succ (spec := spec) ℓ' hℓ'1
      have hidx' : (⟨ℓ', Nat.lt_of_succ_lt hℓ'1⟩ : Fin spec.N) = ⟨ℓ', hℓ'⟩ := rfl
      have hLeq : spec.L ⟨ℓ', hℓ'⟩ = 2 * spec.L ⟨ℓ' + 1, hℓ'1⟩ := by
        rw [← hidx', ← hLs]; ring
      rw [hLeq]
      exact Dvd.dvd.mul_left h1 2

/-- For `ℓ' < k`, `2 · L_k ∣ L_ℓ'`. -/
lemma two_L_dvd_L [CooleyTukeyLike spec] {ℓ' k : ℕ} (hℓ' : ℓ' < spec.N)
    (hk : k < spec.N) (hℓ'k : ℓ' < k) :
    (2 * spec.L ⟨k, hk⟩) ∣ spec.L ⟨ℓ', hℓ'⟩ := by
  obtain ⟨n, hn⟩ : ∃ n, k = ℓ' + n + 1 := ⟨k - ℓ' - 1, by omega⟩
  subst hn
  exact spec.two_L_dvd_L_aux n ℓ' hℓ' hk

/-! ### Step 5c — `bitOf k` is preserved by `partnerIdx ⟨ℓ', _⟩` when `ℓ' < k` -/

/-- The layer-`ℓ'` partner has the same bit-`β_k` value, when `ℓ' < k`. -/
lemma bitOf_partnerIdx_eq_of_lt [CooleyTukeyLike spec]
    {ℓ' k : ℕ} (hℓ' : ℓ' < spec.N) (hk : k < spec.N) (hℓ'k : ℓ' < k)
    (i : Fin spec.n) :
    spec.bitOf ⟨k, hk⟩ (spec.partnerIdx ⟨ℓ', hℓ'⟩ i) = spec.bitOf ⟨k, hk⟩ i := by
  -- 0 < L_k from spec.n > 0.
  have hL_pos : 0 < spec.L ⟨k, hk⟩ := by
    rcases Nat.eq_zero_or_pos (spec.L ⟨k, hk⟩) with hz | hp
    · exfalso
      have hgl := spec.hGL ⟨k, hk⟩
      have hn_pos : 0 < spec.n := lt_of_le_of_lt (Nat.zero_le _) i.isLt
      rw [hz] at hgl; simp at hgl; omega
    · exact hp
  -- Get the factorisation 2L_k * m = L_ℓ'.
  obtain ⟨m, hm⟩ := spec.two_L_dvd_L hℓ' hk hℓ'k
  -- Rewrite L_ℓ' = L_k * (2m); used by both branches.
  have hLeq : spec.L ⟨ℓ', hℓ'⟩ = spec.L ⟨k, hk⟩ * (2 * m) := by
    rw [hm]; ring
  by_cases hside : i.val % (2 * spec.L ⟨ℓ', hℓ'⟩) < spec.L ⟨ℓ', hℓ'⟩
  · -- a-side: partnerIdx.val = i.val + L_ℓ'.
    unfold bitOf
    rw [spec.partnerIdx_val_aside hside, hLeq]
    rw [show i.val + spec.L ⟨k, hk⟩ * (2 * m)
          = i.val + (2 * m) * spec.L ⟨k, hk⟩ from by ring]
    rw [Nat.add_mul_div_right _ _ hL_pos]
    omega
  · -- b-side: partnerIdx.val = i.val - L_ℓ'; need L_ℓ' ≤ i.val.
    push_neg at hside
    have hLi : spec.L ⟨ℓ', hℓ'⟩ ≤ i.val := by
      have h_mod_le : i.val % (2 * spec.L ⟨ℓ', hℓ'⟩) ≤ i.val := Nat.mod_le _ _
      linarith
    unfold bitOf
    rw [spec.partnerIdx_val_bside (not_lt.mpr hside)]
    have habs : i.val = (i.val - spec.L ⟨ℓ', hℓ'⟩) + (2 * m) * spec.L ⟨k, hk⟩ := by
      have : spec.L ⟨ℓ', hℓ'⟩ = (2 * m) * spec.L ⟨k, hk⟩ := by rw [hLeq]; ring
      omega
    conv_rhs => rw [habs]
    rw [Nat.add_mul_div_right _ _ hL_pos]
    omega

/-! ### Step 5d — `exists_bSide_decomp` (b-side counterpart of `exists_aSide_decomp`) -/

/-- Recover coordinates `(g, j)` from a flat b-side index `i`
    (with `bitOf ℓ i = 1`). -/
lemma exists_bSide_decomp (ℓ : Fin spec.N) (i : Fin spec.n)
    (hbit : spec.bitOf ℓ i = 1) :
    ∃ (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)), i = spec.bSideIdx ℓ g j := by
  set p := coordEquiv (spec.G ℓ) (spec.L ℓ)
    ((finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) i) with hp_def
  refine ⟨p.1, p.2.2, ?_⟩
  have hround : spec.flatIdx ℓ p.1 p.2.1 p.2.2 = i := spec.flatIdx_coordEquiv ℓ i
  have hL_pos : 0 < spec.L ℓ := by
    rcases Nat.eq_zero_or_pos (spec.L ℓ) with hzero | hpos
    · exfalso
      have hgl := spec.hGL ℓ
      have hn_pos : 0 < spec.n := lt_of_le_of_lt (Nat.zero_le _) i.isLt
      rw [hzero] at hgl; simp at hgl; omega
    · exact hpos
  have hi_val := spec.val_eq_flatIdx_val ℓ i
  rw [← hp_def] at hi_val
  have hj_lt : p.2.2.val < spec.L ℓ := p.2.2.isLt
  have hs_lt : p.2.1.val < 2 := p.2.1.isLt
  have hdiv : i.val / spec.L ℓ = p.1.val * 2 + p.2.1.val := by
    rw [hi_val]
    rw [show p.1.val * (2 * spec.L ℓ) + p.2.1.val * spec.L ℓ + p.2.2.val
          = p.2.2.val + (p.2.1.val + p.1.val * 2) * spec.L ℓ from by ring]
    rw [Nat.add_mul_div_right _ _ hL_pos]
    rw [Nat.div_eq_of_lt hj_lt]; omega
  have hbitVal : spec.bitOf ℓ i = p.2.1.val := by
    unfold bitOf; rw [hdiv]; omega
  have hs_one : p.2.1 = (1 : Fin 2) := by
    apply Fin.ext; rw [← hbitVal]; exact hbit
  show i = spec.bSideIdx ℓ p.1 p.2.2
  apply Fin.ext
  rw [spec.bSideIdx_val, hi_val, hs_one]
  show p.1.val * (2 * spec.L ℓ) + (1 : Fin 2).val * spec.L ℓ + p.2.2.val
      = p.1.val * (2 * spec.L ℓ) + spec.L ℓ + p.2.2.val
  have : (1 : Fin 2).val = 1 := rfl
  rw [this]; ring

/-- **Step 4 headline.** `(layerInv ∘ pert) w ∈ V ⟨k, hk⟩` for every `w`. -/
lemma layerInv_pert_mem_V
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (F : spec.FaultSet) (k : ℕ) (hk : k < spec.N) (w : Fin spec.n → K) :
    (spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
        (spec.perturbation Z F ⟨k, hk⟩ w) ∈ spec.V (K := K) ⟨k, hk⟩ := by
  apply spec.mem_V_of_bit_zero_vanish
  intro i hbit
  -- Decompose i as aSideIdx g j.
  obtain ⟨g, j, rfl⟩ := spec.exists_aSide_decomp ⟨k, hk⟩ i hbit
  -- aSide (layerInv (pert w)) g j = (aSide (pert w) g j + bSide (pert w) g j) / 2
  --   = ((Z g - Z' g) * bSide w g j + (Z' g - Z g) * bSide w g j) / 2 = 0.
  show spec.aSide ((spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
        (spec.perturbation Z F ⟨k, hk⟩ w)) ⟨k, hk⟩ g j = 0
  rw [spec.aSide_layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 _ g j,
      spec.aSide_perturbation Z F ⟨k, hk⟩ w g j,
      spec.bSide_perturbation Z F ⟨k, hk⟩ w g j]
  -- ((Z g - Z' g) * b + (Z' g - Z g) * b) / 2 = 0.
  have : (Z ⟨k, hk⟩ g - (spec.faultedTwiddles Z F) ⟨k, hk⟩ g)
            * spec.bSide w ⟨k, hk⟩ g j
        + ((spec.faultedTwiddles Z F) ⟨k, hk⟩ g - Z ⟨k, hk⟩ g)
            * spec.bSide w ⟨k, hk⟩ g j
        = 0 := by ring
  rw [this, zero_div]

/-! ### Step 5e — `layerInv ⟨ℓ',⟩` preserves `V ⟨k,⟩` when `ℓ' < k` -/

/-- Converse of `mem_V_of_bit_zero_vanish`: a member of `V ℓ` vanishes at
    every index with `bitOf ℓ` equal to zero. -/
lemma bit_zero_vanish_of_mem_V (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (hv : v ∈ spec.V (K := K) ℓ) (i : Fin spec.n) (hbit : spec.bitOf ℓ i = 0) :
    v i = 0 := by
  refine Submodule.span_induction (p := fun w _ => w i = 0) ?_ ?_ ?_ ?_ hv
  · -- mem case: w = basis i₀ with bitOf ℓ i₀ = 1.
    rintro w ⟨i₀, hb1, rfl⟩
    unfold basis
    by_cases hii : i = i₀
    · exfalso; rw [hii] at hbit; rw [hb1] at hbit; exact absurd hbit (by decide)
    · show (if i = i₀ then (1 : K) else 0) = 0
      rw [if_neg hii]
  · -- zero
    simp
  · -- add
    intros u w _ _ hu hw
    show (u + w) i = 0
    rw [Pi.add_apply, hu, hw, add_zero]
  · -- smul
    intros c w _ hw
    show (c • w) i = 0
    rw [Pi.smul_apply, hw, smul_zero]

/-- `partnerIdx ⟨ℓ⟩` swaps `aSideIdx` ↔ `bSideIdx`. -/
lemma partnerIdx_aSideIdx (ℓ : Fin spec.N) (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.partnerIdx ℓ (spec.aSideIdx ℓ g j) = spec.bSideIdx ℓ g j := by
  apply Fin.ext
  have hmod : (spec.aSideIdx ℓ g j).val % (2 * spec.L ℓ) < spec.L ℓ := by
    rw [spec.aSideIdx_val]
    rw [show g.val * (2 * spec.L ℓ) + j.val = j.val + g.val * (2 * spec.L ℓ) from by ring]
    rw [Nat.add_mul_mod_self_right]
    have hj := j.isLt
    have : j.val % (2 * spec.L ℓ) ≤ j.val := Nat.mod_le _ _
    omega
  rw [spec.partnerIdx_val_aside hmod, spec.aSideIdx_val, spec.bSideIdx_val]
  ring

/-- `partnerIdx ⟨ℓ⟩` swaps `bSideIdx` → `aSideIdx`. -/
lemma partnerIdx_bSideIdx (ℓ : Fin spec.N) (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.partnerIdx ℓ (spec.bSideIdx ℓ g j) = spec.aSideIdx ℓ g j := by
  apply Fin.ext
  have hmod : ¬ (spec.bSideIdx ℓ g j).val % (2 * spec.L ℓ) < spec.L ℓ := by
    rw [spec.bSideIdx_val]
    rw [show g.val * (2 * spec.L ℓ) + spec.L ℓ + j.val
          = (spec.L ℓ + j.val) + g.val * (2 * spec.L ℓ) from by ring]
    rw [Nat.add_mul_mod_self_right]
    have hj := j.isLt
    -- (spec.L ℓ + j.val) % (2 * spec.L ℓ) = spec.L ℓ + j.val since spec.L ℓ + j.val < 2 * spec.L ℓ
    have hbnd : spec.L ℓ + j.val < 2 * spec.L ℓ := by omega
    rw [Nat.mod_eq_of_lt hbnd]
    omega
  rw [spec.partnerIdx_val_bside hmod, spec.aSideIdx_val, spec.bSideIdx_val]
  -- bSideIdx.val - L = (g*(2L) + L + j) - L = g*(2L) + j = aSideIdx.val
  omega

/-- **Step 5e.** For `ℓ' < k`, `layerInv ⟨ℓ',⟩` preserves `V ⟨k,⟩`. -/
lemma layerInv_preserves_V_of_lt [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {ℓ' k : ℕ} (hℓ' : ℓ' < spec.N) (hk : k < spec.N) (hℓ'k : ℓ' < k)
    {v : Fin spec.n → K} (hv : v ∈ spec.V (K := K) ⟨k, hk⟩) :
    (spec.layerInv Z ⟨ℓ', hℓ'⟩ (fun g => hZ ⟨ℓ', hℓ'⟩ g) h2) v
      ∈ spec.V (K := K) ⟨k, hk⟩ := by
  apply spec.mem_V_of_bit_zero_vanish
  intro i hbit
  -- Case-split on bitOf ⟨ℓ',⟩ i.
  by_cases hbℓ' : spec.bitOf ⟨ℓ', hℓ'⟩ i = 0
  · -- a-side case: i = aSideIdx ⟨ℓ',⟩ g j.
    obtain ⟨g, j, rfl⟩ := spec.exists_aSide_decomp ⟨ℓ', hℓ'⟩ i hbℓ'
    show spec.aSide ((spec.layerInv Z ⟨ℓ', hℓ'⟩ (fun g => hZ ⟨ℓ', hℓ'⟩ g) h2) v)
            ⟨ℓ', hℓ'⟩ g j = 0
    rw [spec.aSide_layerInv Z ⟨ℓ', hℓ'⟩ _ h2 v g j]
    -- (aSide v + bSide v) / 2 = 0.
    have h_aSide : spec.aSide v ⟨ℓ', hℓ'⟩ g j = 0 := by
      rw [spec.aSide_eq_apply]
      exact spec.bit_zero_vanish_of_mem_V ⟨k, hk⟩ v hv _ hbit
    have h_partner :
        spec.bitOf ⟨k, hk⟩ (spec.bSideIdx ⟨ℓ', hℓ'⟩ g j) = 0 := by
      have := spec.bitOf_partnerIdx_eq_of_lt hℓ' hk hℓ'k (spec.aSideIdx ⟨ℓ', hℓ'⟩ g j)
      rw [spec.partnerIdx_aSideIdx] at this
      rw [this]; exact hbit
    have h_bSide : spec.bSide v ⟨ℓ', hℓ'⟩ g j = 0 := by
      rw [spec.bSide_eq_apply]
      exact spec.bit_zero_vanish_of_mem_V ⟨k, hk⟩ v hv _ h_partner
    rw [h_aSide, h_bSide, add_zero, zero_div]
  · -- b-side case: i = bSideIdx ⟨ℓ',⟩ g j.
    have hbℓ'1 : spec.bitOf ⟨ℓ', hℓ'⟩ i = 1 := by
      have h_lt : spec.bitOf ⟨ℓ', hℓ'⟩ i < 2 := Nat.mod_lt _ (by norm_num)
      unfold bitOf at hbℓ' ⊢; omega
    obtain ⟨g, j, rfl⟩ := spec.exists_bSide_decomp ⟨ℓ', hℓ'⟩ i hbℓ'1
    -- (layerInv v) (bSideIdx g j) = bSide (layerInv v) ⟨ℓ',⟩ g j (by bSide_eq_apply)
    show spec.bSide ((spec.layerInv Z ⟨ℓ', hℓ'⟩ (fun g => hZ ⟨ℓ', hℓ'⟩ g) h2) v)
            ⟨ℓ', hℓ'⟩ g j = 0
    rw [spec.bSide_layerInv Z ⟨ℓ', hℓ'⟩ _ h2 v g j]
    -- (aSide v - bSide v) / (2 * Z ℓ' g) = 0.
    have h_partner :
        spec.bitOf ⟨k, hk⟩ (spec.aSideIdx ⟨ℓ', hℓ'⟩ g j) = 0 := by
      have := spec.bitOf_partnerIdx_eq_of_lt hℓ' hk hℓ'k (spec.bSideIdx ⟨ℓ', hℓ'⟩ g j)
      rw [spec.partnerIdx_bSideIdx] at this
      rw [this]; exact hbit
    have h_aSide : spec.aSide v ⟨ℓ', hℓ'⟩ g j = 0 := by
      rw [spec.aSide_eq_apply]
      exact spec.bit_zero_vanish_of_mem_V ⟨k, hk⟩ v hv _ h_partner
    have h_bSide : spec.bSide v ⟨ℓ', hℓ'⟩ g j = 0 := by
      rw [spec.bSide_eq_apply]
      exact spec.bit_zero_vanish_of_mem_V ⟨k, hk⟩ v hv _ hbit
    rw [h_aSide, h_bSide, sub_zero, zero_div]

/-! ### Step 5f — `runPrefixInv m` preserves `V ⟨k,⟩` when `m ≤ k` -/

/-- Unfolding lemma for `runPrefixInv`: peels off the last layer. -/
lemma runPrefixInv_succ {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0) (k : ℕ) (hk : k + 1 ≤ spec.N) (v : Fin spec.n → K) :
    spec.runPrefixInv Z h2 (k+1) hk (fun i g => hZ _ g) v
      = spec.runPrefixInv Z h2 k (Nat.le_of_succ_le hk) (fun i g => hZ _ g)
          (spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 v) := by
  apply (spec.runPrefixEquiv Z h2 (k+1) hk (fun i g => hZ _ g)).injective
  show (spec.runPrefixEquiv Z h2 (k+1) hk (fun i g => hZ _ g))
        ((spec.runPrefixEquiv Z h2 (k+1) hk (fun i g => hZ _ g)).symm v)
      = (spec.runPrefixEquiv Z h2 (k+1) hk (fun i g => hZ _ g))
          (spec.runPrefixInv Z h2 k (Nat.le_of_succ_le hk) (fun i g => hZ _ g)
            (spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 v))
  rw [LinearEquiv.apply_symm_apply]
  show v = spec.runPrefix Z (k+1) hk
            (spec.runPrefixInv Z h2 k (Nat.le_of_succ_le hk) (fun i g => hZ _ g)
              (spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 v))
  rw [spec.runPrefix_succ Z k hk]
  show v = (spec.layer Z ⟨k, hk⟩) ((spec.runPrefix Z k (Nat.le_of_succ_le hk))
            (spec.runPrefixInv Z h2 k (Nat.le_of_succ_le hk) (fun i g => hZ _ g)
              (spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 v)))
  have h_pre : (spec.runPrefix Z k (Nat.le_of_succ_le hk))
                ((spec.runPrefixInv Z h2 k (Nat.le_of_succ_le hk)
                    (fun i g => hZ _ g))
                  (spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 v))
              = spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 v := by
    show (spec.runPrefixEquiv Z h2 k (Nat.le_of_succ_le hk) (fun i g => hZ _ g))
          ((spec.runPrefixEquiv Z h2 k (Nat.le_of_succ_le hk)
              (fun i g => hZ _ g)).symm _) = _
    rw [LinearEquiv.apply_symm_apply]
  rw [h_pre]
  show v = (spec.layerEquiv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
            ((spec.layerEquiv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2).symm v)
  rw [LinearEquiv.apply_symm_apply]

/-- **Step 5f.** `runPrefixInv m` preserves `V ⟨k,⟩` whenever `m ≤ k`. -/
lemma runPrefixInv_preserves_V [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (k : ℕ) (hk : k < spec.N) :
    ∀ (m : ℕ) (hm : m ≤ k) {v : Fin spec.n → K},
      v ∈ spec.V (K := K) ⟨k, hk⟩ →
      (spec.runPrefixInv Z h2 m (le_trans hm (Nat.le_of_lt hk))
          (fun i g => hZ _ g)) v ∈ spec.V (K := K) ⟨k, hk⟩
  | 0, _, v, hv => by
      -- runPrefixInv 0 = id.
      show (spec.runPrefixEquiv Z h2 0 (Nat.zero_le _) (fun i g => hZ _ g)).symm v
            ∈ spec.V (K := K) ⟨k, hk⟩
      -- runPrefixEquiv at length 0 is identity (foldl over empty = id).
      have h_id : (spec.runPrefixEquiv Z h2 0 (Nat.zero_le _) (fun i g => hZ _ g)).symm v
                = v := by
        apply (spec.runPrefixEquiv Z h2 0 (Nat.zero_le _) (fun i g => hZ _ g)).injective
        rw [LinearEquiv.apply_symm_apply]
        -- runPrefixEquiv 0 v = runPrefix Z 0 v = id v = v.
        show v = spec.runPrefix Z 0 (Nat.zero_le _) v
        unfold runPrefix
        simp
      rw [h_id]; exact hv
  | m+1, hm, v, hv => by
      have hm_le : m ≤ k := Nat.le_of_succ_le hm
      have hm_lt : m < k := hm
      have hmN : m < spec.N := lt_of_lt_of_le hm_lt (Nat.le_of_lt hk)
      -- Apply step lemma: runPrefixInv (m+1) v = runPrefixInv m (layerInv ⟨m,⟩ v).
      have hstep := spec.runPrefixInv_succ hZ h2 m
                      (le_trans hm (Nat.le_of_lt hk)) v
      rw [hstep]
      -- Apply IH to layerInv v, which is in V_k by 5e (m < k).
      have hlay := spec.layerInv_preserves_V_of_lt hZ h2 hmN hk hm_lt hv
      exact runPrefixInv_preserves_V hZ h2 k hk m hm_le hlay

/-! ### Step 6 — Headline -/

/-- **F7 main result.** `INTT(im T_ℓ) ⊆ V_ℓ` (paper Lemma `lem:Vcontain`). -/
theorem teleTerm_intt_image_subset_V
    [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) :
    LinearMap.range
        ((spec.nttInv hZ h2).comp
            (spec.teleTerm Z (spec.faultedTwiddles Z F) k))
      ≤ spec.V (K := K) ⟨k, hk⟩ := by
  rintro x ⟨w, hw⟩
  rw [← hw]
  show spec.nttInv hZ h2 (spec.teleTerm Z (spec.faultedTwiddles Z F) k w)
        ∈ spec.V (K := K) ⟨k, hk⟩
  rw [spec.teleTerm_factor Z (spec.faultedTwiddles Z F) k hk,
      spec.layer_diff_eq_perturbation Z F k hk]
  rw [LinearMap.comp_apply, LinearMap.comp_apply]
  rw [spec.nttInv_suffix_apply hZ h2 k hk]
  apply spec.runPrefixInv_preserves_V hZ h2 k hk k le_rfl
  exact spec.layerInv_pert_mem_V hZ h2 F k hk _

/-! ### Generalized headlines for arbitrary replacement twiddles -/

/-- `aSide (pertGen w) ℓ g j = (Z ℓ g − Z' ℓ g) * bSide w ℓ g j`
    (gen version using `perturbationGen`). -/
lemma aSide_perturbationGen (Z Z_repl : spec.Twiddles K) (F : spec.FaultSet)
    (ℓ : Fin spec.N) (w : Fin spec.n → K)
    (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.aSide (spec.perturbationGen Z Z_repl F ℓ w) ℓ g j
      = (Z ℓ g - (spec.faultedTwiddlesGen Z Z_repl F) ℓ g) * spec.bSide w ℓ g j := by
  show spec.aSide ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ℓ) w)
        ℓ g j = _
  have hsub : spec.aSide ((spec.layer Z ℓ) w -
      (spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ℓ) w) ℓ g j
    = spec.aSide ((spec.layer Z ℓ) w) ℓ g j
      - spec.aSide ((spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ℓ) w) ℓ g j := by
    show ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ℓ) w)
          (spec.aSideIdx ℓ g j) = _
    rw [Pi.sub_apply]; rfl
  rw [hsub, spec.aSide_layer Z ℓ w g j, spec.aSide_layer _ ℓ w g j]; ring

/-- `bSide (pertGen w) ℓ g j = (Z' ℓ g − Z ℓ g) * bSide w ℓ g j`
    (gen version). -/
lemma bSide_perturbationGen (Z Z_repl : spec.Twiddles K) (F : spec.FaultSet)
    (ℓ : Fin spec.N) (w : Fin spec.n → K)
    (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.bSide (spec.perturbationGen Z Z_repl F ℓ w) ℓ g j
      = ((spec.faultedTwiddlesGen Z Z_repl F) ℓ g - Z ℓ g) * spec.bSide w ℓ g j := by
  show spec.bSide ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ℓ) w)
        ℓ g j = _
  have hsub : spec.bSide ((spec.layer Z ℓ) w -
      (spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ℓ) w) ℓ g j
    = spec.bSide ((spec.layer Z ℓ) w) ℓ g j
      - spec.bSide ((spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ℓ) w) ℓ g j := by
    show ((spec.layer Z ℓ) w - (spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ℓ) w)
          (spec.bSideIdx ℓ g j) = _
    rw [Pi.sub_apply]; rfl
  rw [hsub, spec.bSide_layer Z ℓ w g j, spec.bSide_layer _ ℓ w g j]; ring

/-- `layerInv(pertGen w) ∈ V_k` (gen version). -/
lemma layerInv_pertGen_mem_V
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (Z_repl : spec.Twiddles K) (F : spec.FaultSet)
    (k : ℕ) (hk : k < spec.N) (w : Fin spec.n → K) :
    (spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
        (spec.perturbationGen Z Z_repl F ⟨k, hk⟩ w) ∈ spec.V (K := K) ⟨k, hk⟩ := by
  apply spec.mem_V_of_bit_zero_vanish
  intro i hbit
  obtain ⟨g, j, rfl⟩ := spec.exists_aSide_decomp ⟨k, hk⟩ i hbit
  show spec.aSide ((spec.layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2)
        (spec.perturbationGen Z Z_repl F ⟨k, hk⟩ w)) ⟨k, hk⟩ g j = 0
  rw [spec.aSide_layerInv Z ⟨k, hk⟩ (fun g => hZ ⟨k, hk⟩ g) h2 _ g j,
      spec.aSide_perturbationGen Z Z_repl F ⟨k, hk⟩ w g j,
      spec.bSide_perturbationGen Z Z_repl F ⟨k, hk⟩ w g j]
  have : (Z ⟨k, hk⟩ g - (spec.faultedTwiddlesGen Z Z_repl F) ⟨k, hk⟩ g)
            * spec.bSide w ⟨k, hk⟩ g j
        + ((spec.faultedTwiddlesGen Z Z_repl F) ⟨k, hk⟩ g - Z ⟨k, hk⟩ g)
            * spec.bSide w ⟨k, hk⟩ g j
        = 0 := by ring
  rw [this, zero_div]

/-- **F7-gen headline.** `INTT(teleTerm_gen image) ⊆ V_k` for arbitrary replacement. -/
theorem teleTerm_intt_image_subset_V_gen
    [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) :
    LinearMap.range
        ((spec.nttInv hZ h2).comp
            (spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k))
      ≤ spec.V (K := K) ⟨k, hk⟩ := by
  rintro x ⟨w, hw⟩
  rw [← hw]
  show spec.nttInv hZ h2 (spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k w)
        ∈ spec.V (K := K) ⟨k, hk⟩
  rw [spec.teleTerm_factor Z (spec.faultedTwiddlesGen Z Z_repl F) k hk,
      spec.layer_diff_eq_perturbationGen Z Z_repl F k hk]
  rw [LinearMap.comp_apply, LinearMap.comp_apply]
  rw [spec.nttInv_suffix_apply hZ h2 k hk]
  apply spec.runPrefixInv_preserves_V hZ h2 k hk k le_rfl
  exact spec.layerInv_pertGen_mem_V hZ h2 Z_repl F k hk _

end LayerSpec
end NttFaultRank
