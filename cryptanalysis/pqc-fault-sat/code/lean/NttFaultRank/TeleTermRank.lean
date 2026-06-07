/-
Phase F5 — Per-term rank equality.

  `finrank (range (teleTerm Z (faultedTwiddles Z F) k)) = spec.L ⟨k, hk⟩`

  for every layer `k < spec.N`, every twiddle assignment `Z` with all
  twiddles nonzero, and every one-per-layer fault selection `F`.

Strategy (suffixList / chainList bridge, avoiding `Fin.foldl_add`):
  1. `suffixList k m h` : structural recursion on `m`, returns
     `[⟨k,_⟩, ⟨k+1,_⟩, ..., ⟨k+m-1,_⟩]`.
  2. `chainList_suffix_runPrefix` : `chainList Z (suffixList k m h) ∘
     runPrefix Z k _ = runPrefix Z (k+m) _` (induction on `m` via
     `runPrefix_succ`).
  3. `runPrefix_congr_twiddles` : if `Z` and `Z'` agree on layers
     `< k`, then `runPrefix Z k = runPrefix Z' k`.
  4. `chainList_congr_twiddles` : if `Z` and `Z'` agree on every
     `ℓ ∈ ls`, then `chainList Z ls = chainList Z' ls`.
  5. `hybrid_eq_ntt_hybridTwiddles` : `hybrid Z Z' k = ntt (W k)`
     where `W k ℓ g := if ℓ.val < k then Z' ℓ g else Z ℓ g`.
  6. `teleTerm_factor` : `teleTerm Z Z' k = suffix ∘ (layer Z ⟨k,_⟩
     - layer Z' ⟨k,_⟩) ∘ runPrefix Z' k _` where `suffix = chainList
     Z (suffixList (k+1) (N-k-1) _)`.
  7. Specialize Z' = faultedTwiddles Z F; middle = `perturbation Z F`.
     Upper bound via F4 + `range_comp_le_range` + LinearEquiv on suffix.
  8. Lower bound: inputs `e_{L_k + s}` for `s : Fin L_k`. Compute
     `(runPrefix Z' k (basis (L_k + s)))` at faulted-group's
     b-side using F3; identity submatrix; then linear-independence of
     `{teleTerm _ _ _ (basis (L_k + s)) : s}` via injectivity of
     suffix and the explicit a-side coefficient extraction.
-/

import NttFaultRank.Invertibility
import NttFaultRank.Fellsupport
import NttFaultRank.PerturbationRank

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Step 1 — `suffixList` and its bridge to `runPrefix` -/

/-- `suffixList k m h = [⟨k,_⟩, ⟨k+1,_⟩, …, ⟨k+m-1,_⟩]`, structural
    recursion on `m`. Caller provides the bound `k + m ≤ spec.N`. -/
def suffixList (k m : ℕ) (h : k + m ≤ spec.N) : List (Fin spec.N) :=
  match m with
  | 0 => []
  | m + 1 =>
    ⟨k, by omega⟩ :: suffixList (k + 1) m (by omega)

@[simp] lemma suffixList_zero (k : ℕ) (h : k + 0 ≤ spec.N) :
    spec.suffixList k 0 h = [] := rfl

@[simp] lemma suffixList_succ (k m : ℕ) (h : k + (m + 1) ≤ spec.N) :
    spec.suffixList k (m + 1) h =
      ⟨k, by omega⟩ :: spec.suffixList (k + 1) m (by omega) := rfl

/-- **Key bridge.** Composing the suffix chain after the prefix gives the
    extended prefix. Proved by induction on `m` using `runPrefix_succ`. -/
lemma chainList_suffix_comp_runPrefix
    (Z : spec.Twiddles K) (k m : ℕ) (h : k + m ≤ spec.N) :
    (chainList spec Z (spec.suffixList k m h)).comp
        (spec.runPrefix Z k (le_trans (Nat.le_add_right k m) h))
      = spec.runPrefix Z (k + m) h := by
  induction m generalizing k with
  | zero =>
    -- suffixList k 0 _ = []; chainList _ [] = id; id ∘ runPrefix = runPrefix.
    simp [suffixList_zero, chainList_nil, LinearMap.id_comp]
  | succ m ih =>
    -- suffixList k (m+1) = ⟨k,_⟩ :: suffixList (k+1) m
    -- chainList Z (⟨k,_⟩ :: ...) = chainList Z (suffix (k+1) m) ∘ layer Z ⟨k,_⟩
    -- ∘ runPrefix Z k = chainList ∘ (layer Z ⟨k,_⟩ ∘ runPrefix Z k)
    -- = chainList ∘ runPrefix Z (k+1) [by runPrefix_succ]
    -- = runPrefix Z ((k+1) + m) [by ih]
    -- = runPrefix Z (k + (m+1)).
    have hsucc : k + 1 + m ≤ spec.N := by omega
    have hk1 : k + 1 ≤ spec.N := by omega
    have ih' := ih (k := k + 1) hsucc
    rw [suffixList_succ, chainList_cons]
    have hpre_succ : (spec.layer Z ⟨k, by omega⟩).comp
                        (spec.runPrefix Z k (by omega))
                      = spec.runPrefix Z (k + 1) hk1 := by
      have := spec.runPrefix_succ Z k hk1
      rw [this]
    -- combine
    calc ((chainList spec Z (spec.suffixList (k + 1) m _)).comp
            (spec.layer Z ⟨k, by omega⟩)).comp
              (spec.runPrefix Z k _)
        = (chainList spec Z (spec.suffixList (k + 1) m _)).comp
            ((spec.layer Z ⟨k, by omega⟩).comp
              (spec.runPrefix Z k _)) := by rw [LinearMap.comp_assoc]
      _ = (chainList spec Z (spec.suffixList (k + 1) m _)).comp
            (spec.runPrefix Z (k + 1) hk1) := by rw [hpre_succ]
      _ = spec.runPrefix Z (k + 1 + m) (by omega) := ih'
      _ = spec.runPrefix Z (k + (m + 1)) h := by
        congr 1; omega

/-- Bound-rewrite helper for `runPrefix`. -/
lemma runPrefix_nat_congr (Z : spec.Twiddles K) {k k' : ℕ}
    (heq : k = k') (h : k ≤ spec.N) (h' : k' ≤ spec.N) :
    spec.runPrefix Z k h = spec.runPrefix Z k' h' := by
  subst heq; rfl

/-- Membership in `suffixList k m h` forces index value `≥ k`. -/
lemma mem_suffixList_imp_ge (k m : ℕ) (h : k + m ≤ spec.N)
    (ℓ : Fin spec.N) (hℓ : ℓ ∈ spec.suffixList k m h) : k ≤ ℓ.val := by
  induction m generalizing k with
  | zero => simp [suffixList_zero] at hℓ
  | succ m ih =>
    rw [suffixList_succ] at hℓ
    rcases List.mem_cons.mp hℓ with hℓ_eq | hℓ_tail
    · subst hℓ_eq; rfl
    · have := ih (k := k + 1) (by omega) hℓ_tail
      omega

/-! ### Step 2 — twiddle-congruence and the `hybrid` factorization -/

/-- If `Z` and `Z'` agree on every layer index `ℓ` with `ℓ.val < k`, then
    `runPrefix Z k = runPrefix Z' k`. Proved by induction on `k` using
    `runPrefix_succ`. -/
lemma runPrefix_congr_twiddles
    (Z Z' : spec.Twiddles K) (k : ℕ) (hk : k ≤ spec.N)
    (hagree : ∀ ℓ : Fin spec.N, ℓ.val < k → Z ℓ = Z' ℓ) :
    spec.runPrefix Z k hk = spec.runPrefix Z' k hk := by
  induction k with
  | zero =>
    -- Both sides reduce to `LinearMap.id` (Fin.foldl 0 = id).
    show spec.runPrefix Z 0 hk = spec.runPrefix Z' 0 hk
    simp [runPrefix]
  | succ k ih =>
    have hkN : k + 1 ≤ spec.N := hk
    have hk_le : k ≤ spec.N := Nat.le_of_succ_le hkN
    have hagree' : ∀ ℓ : Fin spec.N, ℓ.val < k → Z ℓ = Z' ℓ :=
      fun ℓ hℓ => hagree ℓ (Nat.lt_succ_of_lt hℓ)
    rw [spec.runPrefix_succ Z k hkN, spec.runPrefix_succ Z' k hkN]
    rw [ih hk_le hagree']
    -- layer Z ⟨k,_⟩ = layer Z' ⟨k,_⟩ because their twiddle at ⟨k,_⟩ agrees.
    have hk_lt_N : k < spec.N := hkN
    have hag := hagree ⟨k, hk_lt_N⟩ (Nat.lt_succ_self k)
    show (spec.layer Z ⟨k, hk_lt_N⟩).comp _
       = (spec.layer Z' ⟨k, hk_lt_N⟩).comp _
    have : spec.layer Z ⟨k, hk_lt_N⟩ = spec.layer Z' ⟨k, hk_lt_N⟩ := by
      unfold layer; rw [hag]
    rw [this]

/-- If `Z` and `Z'` agree on every layer in the list, then their
    chain-lists are equal. -/
lemma chainList_congr_twiddles
    (Z Z' : spec.Twiddles K) (ls : List (Fin spec.N))
    (hagree : ∀ ℓ ∈ ls, Z ℓ = Z' ℓ) :
    chainList spec Z ls = chainList spec Z' ls := by
  induction ls with
  | nil => rfl
  | cons ℓ ls ih =>
    rw [chainList_cons, chainList_cons]
    have hag : Z ℓ = Z' ℓ := hagree ℓ List.mem_cons_self
    have hag_tail : ∀ ℓ' ∈ ls, Z ℓ' = Z' ℓ' :=
      fun ℓ' hℓ' => hagree ℓ' (List.mem_cons_of_mem _ hℓ')
    rw [ih hag_tail]
    have : spec.layer Z ℓ = spec.layer Z' ℓ := by unfold layer; rw [hag]
    rw [this]

/-- The hybrid twiddle set at threshold `k`: picks `Z'` on layers `< k`,
    `Z` on layers `≥ k`. -/
def hybridTwiddles (Z Z' : spec.Twiddles K) (k : ℕ) : spec.Twiddles K :=
  fun ℓ g => if ℓ.val < k then Z' ℓ g else Z ℓ g

/-- Layers `< k`: `hybridTwiddles = Z'`. -/
lemma hybridTwiddles_eq_left (Z Z' : spec.Twiddles K) (k : ℕ)
    (ℓ : Fin spec.N) (hℓ : ℓ.val < k) :
    spec.hybridTwiddles Z Z' k ℓ = Z' ℓ := by
  unfold hybridTwiddles; funext g; simp [hℓ]

/-- Layers `≥ k`: `hybridTwiddles = Z`. -/
lemma hybridTwiddles_eq_right (Z Z' : spec.Twiddles K) (k : ℕ)
    (ℓ : Fin spec.N) (hℓ : ¬ ℓ.val < k) :
    spec.hybridTwiddles Z Z' k ℓ = Z ℓ := by
  unfold hybridTwiddles; funext g; simp [hℓ]

/-- `hybrid Z Z' k = ntt (hybridTwiddles Z Z' k)`: the hybrid map is the
    NTT under the hybrid twiddle set. -/
lemma hybrid_eq_ntt_hybridTwiddles (Z Z' : spec.Twiddles K) (k : ℕ) :
    spec.hybrid Z Z' k = spec.ntt (spec.hybridTwiddles Z Z' k) := by
  unfold hybrid ntt
  congr 1 with acc ℓ
  -- step functions: `layer (if c then Z' else Z) ℓ` vs `layer (hybridTwiddles) ℓ`.
  -- Both reduce by funext on the twiddle and apply_ite.
  by_cases h : ℓ.val < k
  · simp [h, spec.hybridTwiddles_eq_left Z Z' k ℓ h]
    have : spec.layer Z' ℓ = spec.layer (spec.hybridTwiddles Z Z' k) ℓ := by
      unfold layer; rw [spec.hybridTwiddles_eq_left Z Z' k ℓ h]
    rw [this]
  · simp [h, spec.hybridTwiddles_eq_right Z Z' k ℓ h]
    have : spec.layer Z ℓ = spec.layer (spec.hybridTwiddles Z Z' k) ℓ := by
      unfold layer; rw [spec.hybridTwiddles_eq_right Z Z' k ℓ h]
    rw [this]

/-- **Generic `ntt` factorization at index `j`.** For any twiddle set `W`
    and any `j < spec.N`, `ntt W` factors as
    `chainList W (suffix (j+1)) ∘ layer W ⟨j,_⟩ ∘ runPrefix W j _`. -/
theorem ntt_factor_at (W : spec.Twiddles K) (j : ℕ) (hj : j < spec.N) :
    spec.ntt W =
      ((chainList spec W (spec.suffixList (j + 1) (spec.N - j - 1) (by omega))).comp
        ((spec.layer W ⟨j, hj⟩).comp
          (spec.runPrefix W j (Nat.le_of_lt hj)))) := by
  have hk1 : j + 1 ≤ spec.N := hj
  have hbridge := spec.chainList_suffix_comp_runPrefix W (j + 1) (spec.N - j - 1)
    (by omega)
  have hsumN : j + 1 + (spec.N - j - 1) = spec.N := by omega
  rw [spec.runPrefix_nat_congr W hsumN (h := by omega) (h' := le_rfl)] at hbridge
  rw [← spec.ntt_eq_runPrefix_last W] at hbridge
  rw [← hbridge, spec.runPrefix_succ W j hk1]

/-- **Hybrid factorization at index `k`.** For any `k < spec.N`, the
    hybrid map factors as
    `chainList Z (suffixList (k+1) (N-k-1) _) ∘ layer Z ⟨k, hk⟩ ∘ runPrefix Z' k _`.
    Derived from `ntt_factor_at` applied to `hybridTwiddles Z Z' k`. -/
theorem hybrid_factor_at (Z Z' : spec.Twiddles K) (k : ℕ) (hk : k < spec.N) :
    spec.hybrid Z Z' k =
      ((chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) (by omega))).comp
        ((spec.layer Z ⟨k, hk⟩).comp
          (spec.runPrefix Z' k (Nat.le_of_lt hk)))) := by
  rw [spec.hybrid_eq_ntt_hybridTwiddles Z Z' k]
  set W := spec.hybridTwiddles Z Z' k with hW_def
  rw [spec.ntt_factor_at W k hk]
  -- Now bridge W → {Z, Z'} on suffix / middle / prefix.
  have hsuf_bd : k + 1 + (spec.N - k - 1) ≤ spec.N := by omega
  have hsuffix_eq : chainList spec W (spec.suffixList (k + 1) (spec.N - k - 1) hsuf_bd)
                  = chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) hsuf_bd) := by
    apply spec.chainList_congr_twiddles
    intro ℓ hℓ
    have hℓ_ge := spec.mem_suffixList_imp_ge (k + 1) (spec.N - k - 1) hsuf_bd ℓ hℓ
    have : W ℓ = Z ℓ := by
      apply spec.hybridTwiddles_eq_right; omega
    exact this
  rw [hsuffix_eq]
  have hlayer_eq : spec.layer W ⟨k, hk⟩ = spec.layer Z ⟨k, hk⟩ := by
    unfold layer
    have : W ⟨k, hk⟩ = Z ⟨k, hk⟩ := by
      apply spec.hybridTwiddles_eq_right; simp
    rw [this]
  rw [hlayer_eq]
  have hprefix_eq : spec.runPrefix W k (Nat.le_of_lt hk)
                  = spec.runPrefix Z' k (Nat.le_of_lt hk) := by
    apply spec.runPrefix_congr_twiddles
    intro ℓ hℓ
    apply spec.hybridTwiddles_eq_left; exact hℓ
  rw [hprefix_eq]

/-- **Hybrid factorization at index `k`, threshold `k+1`.** The hybrid
    at threshold `k+1` factors with the SAME suffix and prefix as the
    threshold-`k` factorization, but with `Z'` (not `Z`) at the middle
    layer `⟨k,_⟩`. -/
theorem hybrid_factor_at_succ (Z Z' : spec.Twiddles K) (k : ℕ) (hk : k < spec.N) :
    spec.hybrid Z Z' (k + 1) =
      ((chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) (by omega))).comp
        ((spec.layer Z' ⟨k, hk⟩).comp
          (spec.runPrefix Z' k (Nat.le_of_lt hk)))) := by
  rw [spec.hybrid_eq_ntt_hybridTwiddles Z Z' (k + 1)]
  set W := spec.hybridTwiddles Z Z' (k + 1) with hW_def
  rw [spec.ntt_factor_at W k hk]
  have hsuf_bd : k + 1 + (spec.N - k - 1) ≤ spec.N := by omega
  -- suffix: layers ≥ k+1, ℓ.val < k+1 false, so W = Z.
  have hsuffix_eq : chainList spec W (spec.suffixList (k + 1) (spec.N - k - 1) hsuf_bd)
                  = chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) hsuf_bd) := by
    apply spec.chainList_congr_twiddles
    intro ℓ hℓ
    have hℓ_ge := spec.mem_suffixList_imp_ge (k + 1) (spec.N - k - 1) hsuf_bd ℓ hℓ
    have : W ℓ = Z ℓ := by
      apply spec.hybridTwiddles_eq_right; omega
    exact this
  rw [hsuffix_eq]
  -- middle: at ℓ = k, ℓ.val < k+1 true, so W = Z'.
  have hlayer_eq : spec.layer W ⟨k, hk⟩ = spec.layer Z' ⟨k, hk⟩ := by
    unfold layer
    have : W ⟨k, hk⟩ = Z' ⟨k, hk⟩ := by
      apply spec.hybridTwiddles_eq_left; exact Nat.lt_succ_self k
    rw [this]
  rw [hlayer_eq]
  -- prefix: layers < k, hence < k+1, so W = Z'.
  have hprefix_eq : spec.runPrefix W k (Nat.le_of_lt hk)
                  = spec.runPrefix Z' k (Nat.le_of_lt hk) := by
    apply spec.runPrefix_congr_twiddles
    intro ℓ hℓ
    apply spec.hybridTwiddles_eq_left; exact Nat.lt_succ_of_lt hℓ
  rw [hprefix_eq]

/-! ### Step 3 — teleTerm factorization -/

/-- **teleTerm factorization.** For any `k < spec.N`,
    `teleTerm Z Z' k = suffix ∘ (layer Z ⟨k,_⟩ - layer Z' ⟨k,_⟩) ∘ runPrefix Z' k _`,
    where `suffix = chainList Z (suffixList (k+1) (N-k-1) _)`. -/
theorem teleTerm_factor (Z Z' : spec.Twiddles K) (k : ℕ) (hk : k < spec.N) :
    spec.teleTerm Z Z' k =
      ((chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) (by omega))).comp
        (((spec.layer Z ⟨k, hk⟩) - (spec.layer Z' ⟨k, hk⟩)).comp
          (spec.runPrefix Z' k (Nat.le_of_lt hk)))) := by
  unfold teleTerm
  rw [spec.hybrid_factor_at Z Z' k hk, spec.hybrid_factor_at_succ Z Z' k hk]
  ext v
  simp [LinearMap.sub_apply, LinearMap.comp_apply, map_sub]

/-! ### Step 4 — Upper bound on the per-term rank

  `finrank (range (teleTerm Z (faultedTwiddles Z F) k)) ≤ spec.L ⟨k, hk⟩`.

  Suffix is a `LinearEquiv` (all twiddles nonzero in suffix). Middle is
  `perturbation Z F ⟨k, hk⟩`, whose range has finrank `L_k` (F4). -/

/-- The suffix-chain as a `LinearEquiv` when all twiddles are nonzero. -/
noncomputable def suffixEquiv
    (Z : spec.Twiddles K) (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (k : ℕ) (hk : k < spec.N) :
    (Fin spec.n → K) ≃ₗ[K] (Fin spec.n → K) :=
  chainListEquiv spec Z h2 (spec.suffixList (k + 1) (spec.N - k - 1) (by omega))
    (fun ℓ _ _ => hZ ℓ _)

lemma suffixEquiv_toLinearMap
    (Z : spec.Twiddles K) (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    (k : ℕ) (hk : k < spec.N) :
    (spec.suffixEquiv Z hZ h2 k hk).toLinearMap =
      chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) (by omega)) := by
  unfold suffixEquiv
  exact chainListEquiv_toLinearMap spec Z h2 _ _

/-- When `Z' = faultedTwiddles Z F`, the layer-`k` difference equals the
    F4 perturbation. -/
lemma layer_diff_eq_perturbation
    (Z : spec.Twiddles K) (F : spec.FaultSet) (k : ℕ) (hk : k < spec.N) :
    (spec.layer Z ⟨k, hk⟩) - (spec.layer (spec.faultedTwiddles Z F) ⟨k, hk⟩)
      = spec.perturbation Z F ⟨k, hk⟩ := rfl

/-- **Upper bound (Step 4).** -/
theorem teleTerm_rank_le
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) :
    Module.finrank K
        (LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) k))
      ≤ spec.L ⟨k, hk⟩ := by
  set Z' := spec.faultedTwiddles Z F with hZ'_def
  set pre := spec.runPrefix Z' k (Nat.le_of_lt hk) with hpre_def
  set mid := spec.perturbation Z F ⟨k, hk⟩ with hmid_def
  -- Factor teleTerm.
  rw [spec.teleTerm_factor Z Z' k hk]
  rw [spec.layer_diff_eq_perturbation Z F k hk]
  -- Now goal: finrank (range (suf.comp (mid.comp pre))) ≤ L_k.
  -- Step A: range (suf.comp (mid.comp pre)) = (range (mid.comp pre)).map suf.
  rw [LinearMap.range_comp]
  -- Step B: (range (mid.comp pre)).map suf — replace suf by sufEquiv.toLinearMap.
  rw [← spec.suffixEquiv_toLinearMap Z hZ h2 k hk]
  rw [LinearEquiv.finrank_map_eq]
  -- Step C: range (mid.comp pre) = (range pre).map mid; bound by finrank (range mid) = L_k.
  rw [LinearMap.range_comp]
  have hmono : (LinearMap.range pre).map mid ≤ LinearMap.range mid :=
    LinearMap.map_le_range
  refine le_trans (Submodule.finrank_mono hmono) ?_
  exact le_of_eq (spec.perturbation_rank_eq hZ hF ⟨k, hk⟩)

/-! ### Step 5 — Lower bound: identity-submatrix linear independence

  Inputs `v_s := basis ⟨L_k + s.val, _⟩` for `s : Fin L_k`. By F3:
    `pre(v_s)` is `1` at `bSideIdx ⟨k,_⟩ g₀ s` (where `s' = s`) and `0`
    elsewhere within group `g₀`. Then `mid(pre(v_s))` evaluated at
    `aSideIdx ⟨k,_⟩ g₀ s'` gives the diagonal `Z_k g₀ · δ(s, s')`
    via the layer-butterfly formula at the a-side. `Z_k g₀ ≠ 0` then
    forces linear independence. Suffix is a `LinearEquiv` so preserves
    linear independence; the resulting `L_k` vectors live in `range
    teleTerm`. -/

/-- The "a-side" projection at layer `ℓ`, group `g`, position `j`.
    Analogue of `bSide` (defined in `Theorem2Subset.lean`). -/
def aSide (v : Fin spec.n → K) (ℓ : Fin spec.N) (g : Fin (spec.G ℓ))
    (j : Fin (spec.L ℓ)) : K :=
  reshape K (spec.G ℓ) (spec.L ℓ)
    ((funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) v) g 0 j

/-- Flat index for the a-side: `g * (2 L) + j`. -/
def aSideIdx (ℓ : Fin spec.N) (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    Fin spec.n :=
  (finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)).symm
    ((coordEquiv (spec.G ℓ) (spec.L ℓ)).symm (g, 0, j))

lemma aSideIdx_val (ℓ : Fin spec.N) (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    (spec.aSideIdx ℓ g j).val = g.val * (2 * spec.L ℓ) + j.val := by
  unfold aSideIdx
  simp [finCastEquiv, coordEquiv_symm_val]

lemma aSide_eq_apply (v : Fin spec.n → K) (ℓ : Fin spec.N)
    (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.aSide v ℓ g j = v (spec.aSideIdx ℓ g j) := rfl

/-- Butterfly formula at the a-side. -/
lemma aSide_layer (Z : spec.Twiddles K) (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.aSide (spec.layer Z ℓ v) ℓ g j
      = spec.aSide v ℓ g j + Z ℓ g * spec.bSide v ℓ g j := by
  unfold aSide bSide layer layerOn fullLayer'
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe,
    LinearEquiv.apply_symm_apply, fullLayer_apply, groupLayer_apply_zero]

/-- For the relevant inputs `j = L_k + s` (with `s.val < L_k`), the
    flat index `L_k + s.val` lies in `Fin spec.n` (since `2 L_k ≤
    spec.n` for any CooleyTukeyLike spec at `k < spec.N`). -/
lemma two_L_le_n [CooleyTukeyLike spec] (k : ℕ) (hk : k < spec.N) :
    2 * spec.L ⟨k, hk⟩ ≤ spec.n := by
  have h0 : 0 < spec.N := lt_of_le_of_lt (Nat.zero_le _) hk
  have hL0 : 2 * spec.L ⟨0, h0⟩ = spec.n := CooleyTukeyLike.L_zero_doubled h0
  have hanti : spec.L ⟨k, hk⟩ ≤ spec.L ⟨0, h0⟩ :=
    spec.L_antitone hk h0 (Nat.zero_le _)
  omega

/-- `pre(v_s) = runPrefix Z' k (basis ⟨L_k + s, _⟩)` at flat index `i`:
    indicator of `i.val % (2 L_k) = L_k + s.val`. Direct corollary of
    F3 (`runPrefix_basis_value_indicator`). -/
lemma runPrefix_at_basis_Lks
    [CooleyTukeyLike spec] (Z : spec.Twiddles K) (k : ℕ) (hk : k < spec.N)
    (s : Fin (spec.L ⟨k, hk⟩)) (i : Fin spec.n) :
    spec.runPrefix Z k (Nat.le_of_lt hk)
        (spec.basis (K := K)
          ⟨spec.L ⟨k, hk⟩ + s.val,
            lt_of_lt_of_le (by have := s.isLt; omega) (spec.two_L_le_n k hk)⟩) i
      = if i.val % (2 * spec.L ⟨k, hk⟩) = spec.L ⟨k, hk⟩ + s.val
        then (1 : K) else 0 := by
  have hk_le : k ≤ (⟨k, hk⟩ : Fin spec.N).val := le_refl _
  have hj_lt : (spec.L ⟨k, hk⟩ + s.val) < 2 * spec.L ⟨k, hk⟩ := by
    have := s.isLt; omega
  have hjN : spec.L ⟨k, hk⟩ + s.val < spec.n :=
    lt_of_lt_of_le hj_lt (spec.two_L_le_n k hk)
  have h := spec.runPrefix_basis_value_indicator Z ⟨k, hk⟩
    (j := ⟨spec.L ⟨k, hk⟩ + s.val, hjN⟩) hj_lt k hk_le i
  convert h using 3 <;> apply Fin.ext <;> rfl

/-! ### Step 6 — Linear independence at the a-side of the faulted group -/

/-- The key diagonal: `aSide (mid(pre(v_s))) ⟨k,hk⟩ g₀ s'`, where
    `g₀` is the unique fault index at layer `k` and `mid` is the
    layer-`k` perturbation. Equals `Z ⟨k,hk⟩ g₀` if `s' = s`, else `0`. -/
lemma aSide_mid_pre_basis
    [CooleyTukeyLike spec]
    (Z : spec.Twiddles K)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N)
    (s s' : Fin (spec.L ⟨k, hk⟩)) :
    let Z' := spec.faultedTwiddles Z F
    let pre := spec.runPrefix Z' k (Nat.le_of_lt hk)
    let mid := spec.perturbation Z F ⟨k, hk⟩
    let g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose
    let v_s := spec.basis (K := K)
      ⟨spec.L ⟨k, hk⟩ + s.val,
        lt_of_lt_of_le (by have := s.isLt; omega) (spec.two_L_le_n k hk)⟩
    spec.aSide (mid (pre v_s)) ⟨k, hk⟩ g₀ s'
      = if s' = s then Z ⟨k, hk⟩ g₀ else 0 := by
  set Z' := spec.faultedTwiddles Z F with hZ'_def
  set pre := spec.runPrefix Z' k (Nat.le_of_lt hk)
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  set v_s := spec.basis (K := K)
    ⟨spec.L ⟨k, hk⟩ + s.val,
      lt_of_lt_of_le (by have := s.isLt; omega) (spec.two_L_le_n k hk)⟩
  -- mid = layer Z ⟨k,_⟩ - layer Z' ⟨k,_⟩.
  show spec.aSide (spec.perturbation Z F ⟨k, hk⟩ (pre v_s)) ⟨k, hk⟩ g₀ s' = _
  have hpert : spec.perturbation Z F ⟨k, hk⟩ (pre v_s)
             = (spec.layer Z ⟨k, hk⟩) (pre v_s)
               - (spec.layer Z' ⟨k, hk⟩) (pre v_s) := by
    show ((spec.layer Z ⟨k, hk⟩) - spec.layer Z' ⟨k, hk⟩) (pre v_s) =
         (spec.layer Z ⟨k, hk⟩) (pre v_s) - (spec.layer Z' ⟨k, hk⟩) (pre v_s)
    rw [LinearMap.sub_apply]
  rw [hpert]
  -- aSide is linear in its v-argument.
  have hsub : spec.aSide ((spec.layer Z ⟨k, hk⟩) (pre v_s) -
                          (spec.layer Z' ⟨k, hk⟩) (pre v_s)) ⟨k, hk⟩ g₀ s'
            = spec.aSide ((spec.layer Z ⟨k, hk⟩) (pre v_s)) ⟨k, hk⟩ g₀ s'
            - spec.aSide ((spec.layer Z' ⟨k, hk⟩) (pre v_s)) ⟨k, hk⟩ g₀ s' := by
    simp [aSide, map_sub, Pi.sub_apply, reshape, funReindex]
  rw [hsub]
  rw [spec.aSide_layer Z ⟨k, hk⟩ (pre v_s) g₀ s']
  rw [spec.aSide_layer Z' ⟨k, hk⟩ (pre v_s) g₀ s']
  -- Z' ⟨k,_⟩ g₀ = 0 (faulted at g₀); difference simplifies.
  have hZ'_zero : Z' ⟨k, hk⟩ g₀ = 0 := by
    have h_eq : F ⟨k, hk⟩ = {g₀} :=
      (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose_spec
    show spec.faultedTwiddles Z F ⟨k, hk⟩ g₀ = 0
    unfold faultedTwiddles
    simp [h_eq]
  rw [hZ'_zero]
  ring_nf
  -- Now: Z ⟨k,_⟩ g₀ * bSide (pre v_s) ⟨k,hk⟩ g₀ s' = if s'=s then Z ⟨k,_⟩ g₀ else 0.
  -- bSide pre v_s ⟨k, hk⟩ g₀ s' = pre v_s (bSideIdx ⟨k,_⟩ g₀ s')
  --                              = δ(s, s') by runPrefix_at_basis_Lks.
  rw [spec.bSide_eq_apply]
  have hb := spec.runPrefix_at_basis_Lks Z' k hk s (spec.bSideIdx ⟨k, hk⟩ g₀ s')
  -- bSideIdx val: g₀ * (2 L_k) + L_k + s'.val.
  have hb_val : (spec.bSideIdx ⟨k, hk⟩ g₀ s').val % (2 * spec.L ⟨k, hk⟩)
              = spec.L ⟨k, hk⟩ + s'.val := by
    rw [spec.bSideIdx_val]
    have hL_pos : 0 < spec.L ⟨k, hk⟩ := by
      have := s.isLt; omega
    have hsum_lt : spec.L ⟨k, hk⟩ + s'.val < 2 * spec.L ⟨k, hk⟩ := by
      have := s'.isLt; omega
    have h2L_pos : 0 < 2 * spec.L ⟨k, hk⟩ := by omega
    -- (g₀ * 2L + L + s'.val) % 2L = (L + s'.val) % 2L = L + s'.val.
    rw [show g₀.val * (2 * spec.L ⟨k, hk⟩) + spec.L ⟨k, hk⟩ + s'.val
          = (spec.L ⟨k, hk⟩ + s'.val) + (2 * spec.L ⟨k, hk⟩) * g₀.val from by ring]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hsum_lt]
  rw [hb]
  rw [hb_val]
  -- Indicator: `L_k + s'.val = L_k + s.val` iff `s' = s`.
  by_cases hss : s' = s
  · subst hss; simp
  · have hss_val : s'.val ≠ s.val := fun heq => hss (Fin.ext heq)
    have hne : ¬ (spec.L ⟨k, hk⟩ + s'.val = spec.L ⟨k, hk⟩ + s.val) := by omega
    rw [if_neg hne, if_neg hss]; ring

/-- **Lower bound (Step 6 — headline LI).** -/
theorem teleTerm_rank_ge
    [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) :
    spec.L ⟨k, hk⟩ ≤ Module.finrank K
      (LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) k)) := by
  set Z' := spec.faultedTwiddles Z F with hZ'_def
  set pre := spec.runPrefix Z' k (Nat.le_of_lt hk) with hpre_def
  set mid := spec.perturbation Z F ⟨k, hk⟩ with hmid_def
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  -- Step A: Define the v_s inputs and the resulting mid(pre(v_s)) vectors.
  let v : Fin (spec.L ⟨k, hk⟩) → (Fin spec.n → K) := fun s =>
    spec.basis (K := K)
      ⟨spec.L ⟨k, hk⟩ + s.val,
        lt_of_lt_of_le (by have := s.isLt; omega) (spec.two_L_le_n k hk)⟩
  let mp : Fin (spec.L ⟨k, hk⟩) → (Fin spec.n → K) := fun s => mid (pre (v s))
  -- Step B: LI of mp via Fintype.linearIndependent_iff.
  have hZk_ne : Z ⟨k, hk⟩ g₀ ≠ 0 := hZ ⟨k, hk⟩ g₀
  have hmp_LI : LinearIndependent K mp := by
    rw [Fintype.linearIndependent_iff]
    intro c hsum s'
    -- Evaluate hsum at aSideIdx ⟨k, hk⟩ g₀ s'.
    have h_at := congrArg (fun w => spec.aSide w ⟨k, hk⟩ g₀ s') hsum
    simp only at h_at
    -- (∑ s, c s • mp s) at aSide = ∑ s, c s * aSide (mp s).
    have hsum_aSide :
        spec.aSide (∑ s, c s • mp s) ⟨k, hk⟩ g₀ s'
          = ∑ s, c s * spec.aSide (mp s) ⟨k, hk⟩ g₀ s' := by
      show (∑ s, c s • mp s) (spec.aSideIdx ⟨k, hk⟩ g₀ s')
         = ∑ s, c s * spec.aSide (mp s) ⟨k, hk⟩ g₀ s'
      rw [Finset.sum_apply]
      apply Finset.sum_congr rfl
      intros s _
      simp [Pi.smul_apply, smul_eq_mul, aSide_eq_apply]
    rw [hsum_aSide] at h_at
    -- aSide (mp s) ⟨k,hk⟩ g₀ s' = if s' = s then Z ⟨k,hk⟩ g₀ else 0.
    have hcalc : ∀ s : Fin (spec.L ⟨k, hk⟩),
        spec.aSide (mp s) ⟨k, hk⟩ g₀ s' = if s' = s then Z ⟨k, hk⟩ g₀ else 0 := by
      intro s
      exact spec.aSide_mid_pre_basis Z hF k hk s s'
    have h_at' : ∑ s, c s * (if s' = s then Z ⟨k, hk⟩ g₀ else 0)
               = (spec.aSide (0 : Fin spec.n → K) ⟨k, hk⟩ g₀ s') := by
      rw [← h_at]; apply Finset.sum_congr rfl; intros s _; rw [hcalc s]
    -- aSide 0 = 0.
    have hzero : spec.aSide (0 : Fin spec.n → K) ⟨k, hk⟩ g₀ s' = 0 := by
      rw [aSide_eq_apply]; rfl
    rw [hzero] at h_at'
    -- Σ collapses to c s' * Z ⟨k,hk⟩ g₀.
    have hsingle : ∑ s, c s * (if s' = s then Z ⟨k, hk⟩ g₀ else 0)
                = c s' * Z ⟨k, hk⟩ g₀ := by
      rw [Finset.sum_eq_single s']
      · simp
      · intros s _ hne; simp [Ne.symm hne]
      · intro h; exact absurd (Finset.mem_univ _) h
    rw [hsingle] at h_at'
    have : c s' * Z ⟨k, hk⟩ g₀ = 0 := h_at'
    exact (mul_eq_zero.mp this).resolve_right hZk_ne
  -- Step C: Apply suffix (LinearEquiv → ker = ⊥) to lift LI to teleTerm v_s.
  set suf := spec.suffixEquiv Z hZ h2 k hk
  have hsuf_inj : LinearMap.ker (suf.toLinearMap) = ⊥ := suf.ker
  have h_tele_LI : LinearIndependent K (suf.toLinearMap ∘ mp) :=
    hmp_LI.map' suf.toLinearMap hsuf_inj
  -- (suf ∘ mp) s = suf (mid (pre (v s))) = teleTerm v_s (by teleTerm_factor).
  have h_eq : (suf.toLinearMap ∘ mp) = (fun s => spec.teleTerm Z Z' k (v s)) := by
    funext s
    show suf.toLinearMap (mid (pre (v s))) = spec.teleTerm Z Z' k (v s)
    rw [spec.suffixEquiv_toLinearMap Z hZ h2 k hk]
    show (chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) _)) _ = _
    rw [spec.teleTerm_factor Z Z' k hk, spec.layer_diff_eq_perturbation Z F k hk]
    rfl
  rw [h_eq] at h_tele_LI
  -- Step D: Lift to LI in the range submodule.
  let R := LinearMap.range (spec.teleTerm Z Z' k)
  have h_mem : ∀ s, spec.teleTerm Z Z' k (v s) ∈ R :=
    fun s => LinearMap.mem_range_self _ _
  let w : Fin (spec.L ⟨k, hk⟩) → R := fun s => ⟨spec.teleTerm Z Z' k (v s), h_mem s⟩
  have hw_LI : LinearIndependent K w := by
    have hcomp : R.subtype ∘ w = fun s => spec.teleTerm Z Z' k (v s) := rfl
    exact LinearIndependent.of_comp R.subtype (by rw [hcomp]; exact h_tele_LI)
  -- Step E: fintype_card_le_finrank → L_k ≤ finrank R.
  have := hw_LI.fintype_card_le_finrank
  simpa using this

/-! ### Step 7 — Headline theorem -/

/-- **Headline F5: per-term rank equality.** -/
theorem teleTerm_rank_eq
    [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) :
    Module.finrank K
        (LinearMap.range (spec.teleTerm Z (spec.faultedTwiddles Z F) k))
      = spec.L ⟨k, hk⟩ :=
  le_antisymm
    (spec.teleTerm_rank_le hZ h2 hF k hk)
    (spec.teleTerm_rank_ge hZ h2 hF k hk)
/-! ### Step 8 — Generalized per-term rank (arbitrary replacement twiddle)

  The proofs above hardcode `Z' = faultedTwiddles Z F` (zeroing).
  Below, we generalize to `Z' = faultedTwiddlesGen Z Z_repl F` where
  `Z_repl` is an arbitrary replacement-twiddle assignment satisfying
  `hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g`.

  The diagonal coefficient changes from `Z g₀` to `Z g₀ - Z_repl g₀`.
  All other arguments (suffix equiv, Fellsupport, bSide indicator)
  are twiddle-independent and carry over verbatim. -/

/-- Bridge: layer Z ⟨k,hk⟩ - layer (faultedTwiddlesGen Z Z_repl F) ⟨k,hk⟩
    equals perturbationGen. -/
lemma layer_diff_eq_perturbationGen
    (Z Z_repl : spec.Twiddles K) (F : spec.FaultSet) (k : ℕ) (hk : k < spec.N) :
    (spec.layer Z ⟨k, hk⟩) - (spec.layer (spec.faultedTwiddlesGen Z Z_repl F) ⟨k, hk⟩)
      = spec.perturbationGen Z Z_repl F ⟨k, hk⟩ := rfl

/-- **Upper bound (gen).** -/
theorem teleTerm_rank_le_gen
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) :
    Module.finrank K
        (LinearMap.range (spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k))
      ≤ spec.L ⟨k, hk⟩ := by
  set Z' := spec.faultedTwiddlesGen Z Z_repl F with hZ'_def
  set pre := spec.runPrefix Z' k (Nat.le_of_lt hk)
  set mid := spec.perturbationGen Z Z_repl F ⟨k, hk⟩
  rw [spec.teleTerm_factor Z Z' k hk]
  rw [spec.layer_diff_eq_perturbationGen Z Z_repl F k hk]
  rw [LinearMap.range_comp, ← spec.suffixEquiv_toLinearMap Z hZ h2 k hk,
      LinearEquiv.finrank_map_eq, LinearMap.range_comp]
  refine le_trans (Submodule.finrank_mono LinearMap.map_le_range) ?_
  exact le_of_eq (spec.perturbation_rank_eq_gen hδ hF ⟨k, hk⟩)

/-- Generalized diagonal formula: `aSide (mid(pre(v_s))) g₀ s'`
    equals `Z g₀ - Z_repl g₀` if `s' = s`, else `0`. -/
lemma aSide_mid_pre_basis_gen
    [CooleyTukeyLike spec]
    (Z : spec.Twiddles K)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N)
    (s s' : Fin (spec.L ⟨k, hk⟩)) :
    let Z' := spec.faultedTwiddlesGen Z Z_repl F
    let pre := spec.runPrefix Z' k (Nat.le_of_lt hk)
    let mid := spec.perturbationGen Z Z_repl F ⟨k, hk⟩
    let g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose
    let v_s := spec.basis (K := K)
      ⟨spec.L ⟨k, hk⟩ + s.val,
        lt_of_lt_of_le (by have := s.isLt; omega) (spec.two_L_le_n k hk)⟩
    spec.aSide (mid (pre v_s)) ⟨k, hk⟩ g₀ s'
      = if s' = s then Z ⟨k, hk⟩ g₀ - Z_repl ⟨k, hk⟩ g₀ else 0 := by
  set Z' := spec.faultedTwiddlesGen Z Z_repl F with hZ'_def
  set pre := spec.runPrefix Z' k (Nat.le_of_lt hk)
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  set v_s := spec.basis (K := K)
    ⟨spec.L ⟨k, hk⟩ + s.val,
      lt_of_lt_of_le (by have := s.isLt; omega) (spec.two_L_le_n k hk)⟩
  show spec.aSide (spec.perturbationGen Z Z_repl F ⟨k, hk⟩ (pre v_s)) ⟨k, hk⟩ g₀ s' = _
  have hpert : spec.perturbationGen Z Z_repl F ⟨k, hk⟩ (pre v_s)
             = (spec.layer Z ⟨k, hk⟩) (pre v_s)
               - (spec.layer Z' ⟨k, hk⟩) (pre v_s) := by
    show ((spec.layer Z ⟨k, hk⟩) - spec.layer Z' ⟨k, hk⟩) (pre v_s) =
         (spec.layer Z ⟨k, hk⟩) (pre v_s) - (spec.layer Z' ⟨k, hk⟩) (pre v_s)
    rw [LinearMap.sub_apply]
  rw [hpert]
  have hsub : spec.aSide ((spec.layer Z ⟨k, hk⟩) (pre v_s) -
                          (spec.layer Z' ⟨k, hk⟩) (pre v_s)) ⟨k, hk⟩ g₀ s'
            = spec.aSide ((spec.layer Z ⟨k, hk⟩) (pre v_s)) ⟨k, hk⟩ g₀ s'
            - spec.aSide ((spec.layer Z' ⟨k, hk⟩) (pre v_s)) ⟨k, hk⟩ g₀ s' := by
    simp [aSide, map_sub, Pi.sub_apply, reshape, funReindex]
  rw [hsub,
      spec.aSide_layer Z ⟨k, hk⟩ (pre v_s) g₀ s',
      spec.aSide_layer Z' ⟨k, hk⟩ (pre v_s) g₀ s']
  -- Z' ⟨k,hk⟩ g₀ = Z_repl ⟨k,hk⟩ g₀ (faulted at g₀).
  have hZ'_eq : Z' ⟨k, hk⟩ g₀ = Z_repl ⟨k, hk⟩ g₀ := by
    have h_eq : F ⟨k, hk⟩ = {g₀} :=
      (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose_spec
    show spec.faultedTwiddlesGen Z Z_repl F ⟨k, hk⟩ g₀ = Z_repl ⟨k, hk⟩ g₀
    unfold faultedTwiddlesGen
    simp [h_eq]
  rw [hZ'_eq]
  ring_nf
  -- Now: (Z g₀ - Z_repl g₀) * bSide (pre v_s) ⟨k,hk⟩ g₀ s'
  --    = if s' = s then Z g₀ - Z_repl g₀ else 0.
  rw [spec.bSide_eq_apply]
  have hb := spec.runPrefix_at_basis_Lks Z' k hk s (spec.bSideIdx ⟨k, hk⟩ g₀ s')
  have hb_val : (spec.bSideIdx ⟨k, hk⟩ g₀ s').val % (2 * spec.L ⟨k, hk⟩)
              = spec.L ⟨k, hk⟩ + s'.val := by
    rw [spec.bSideIdx_val]
    have hsum_lt : spec.L ⟨k, hk⟩ + s'.val < 2 * spec.L ⟨k, hk⟩ := by
      have := s'.isLt; omega
    rw [show g₀.val * (2 * spec.L ⟨k, hk⟩) + spec.L ⟨k, hk⟩ + s'.val
          = (spec.L ⟨k, hk⟩ + s'.val) + (2 * spec.L ⟨k, hk⟩) * g₀.val from by ring]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hsum_lt]
  rw [hb, hb_val]
  by_cases hss : s' = s
  · subst hss; simp
  · have hne : ¬ (spec.L ⟨k, hk⟩ + s'.val = spec.L ⟨k, hk⟩ + s.val) := by
      intro h; exact hss (Fin.ext (by omega))
    rw [if_neg hne, if_neg hss]; ring

/-- **Lower bound (gen).** -/
theorem teleTerm_rank_ge_gen
    [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) :
    spec.L ⟨k, hk⟩ ≤ Module.finrank K
      (LinearMap.range (spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k)) := by
  set Z' := spec.faultedTwiddlesGen Z Z_repl F with hZ'_def
  set pre := spec.runPrefix Z' k (Nat.le_of_lt hk)
  set mid := spec.perturbationGen Z Z_repl F ⟨k, hk⟩
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  let v : Fin (spec.L ⟨k, hk⟩) → (Fin spec.n → K) := fun s =>
    spec.basis (K := K)
      ⟨spec.L ⟨k, hk⟩ + s.val,
        lt_of_lt_of_le (by have := s.isLt; omega) (spec.two_L_le_n k hk)⟩
  let mp : Fin (spec.L ⟨k, hk⟩) → (Fin spec.n → K) := fun s => mid (pre (v s))
  -- Diagonal is Z g₀ - Z_repl g₀, which is nonzero by hδ.
  have hg₀_mem : g₀ ∈ F ⟨k, hk⟩ := by
    rw [(Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose_spec]
    exact Finset.mem_singleton_self g₀
  have hδk : Z ⟨k, hk⟩ g₀ - Z_repl ⟨k, hk⟩ g₀ ≠ 0 :=
    sub_ne_zero.mpr (hδ ⟨k, hk⟩ g₀ hg₀_mem)
  have hmp_LI : LinearIndependent K mp := by
    rw [Fintype.linearIndependent_iff]
    intro c hsum s'
    have h_at := congrArg (fun w => spec.aSide w ⟨k, hk⟩ g₀ s') hsum
    simp only at h_at
    have hsum_aSide :
        spec.aSide (∑ s, c s • mp s) ⟨k, hk⟩ g₀ s'
          = ∑ s, c s * spec.aSide (mp s) ⟨k, hk⟩ g₀ s' := by
      show (∑ s, c s • mp s) (spec.aSideIdx ⟨k, hk⟩ g₀ s')
         = ∑ s, c s * spec.aSide (mp s) ⟨k, hk⟩ g₀ s'
      rw [Finset.sum_apply]
      apply Finset.sum_congr rfl
      intros s _
      simp [Pi.smul_apply, smul_eq_mul, aSide_eq_apply]
    rw [hsum_aSide] at h_at
    have hcalc : ∀ s : Fin (spec.L ⟨k, hk⟩),
        spec.aSide (mp s) ⟨k, hk⟩ g₀ s'
          = if s' = s then Z ⟨k, hk⟩ g₀ - Z_repl ⟨k, hk⟩ g₀ else 0 := by
      intro s
      exact spec.aSide_mid_pre_basis_gen Z hF k hk s s'
    have h_at' : ∑ s, c s * (if s' = s then Z ⟨k, hk⟩ g₀ - Z_repl ⟨k, hk⟩ g₀ else 0)
               = (spec.aSide (0 : Fin spec.n → K) ⟨k, hk⟩ g₀ s') := by
      rw [← h_at]; apply Finset.sum_congr rfl; intros s _; rw [hcalc s]
    have hzero : spec.aSide (0 : Fin spec.n → K) ⟨k, hk⟩ g₀ s' = 0 := by
      rw [aSide_eq_apply]; rfl
    rw [hzero] at h_at'
    have hsingle : ∑ s, c s * (if s' = s then Z ⟨k, hk⟩ g₀ - Z_repl ⟨k, hk⟩ g₀ else 0)
                = c s' * (Z ⟨k, hk⟩ g₀ - Z_repl ⟨k, hk⟩ g₀) := by
      rw [Finset.sum_eq_single s']
      · simp
      · intros s _ hne; simp [Ne.symm hne]
      · intro h; exact absurd (Finset.mem_univ _) h
    rw [hsingle] at h_at'
    exact (mul_eq_zero.mp h_at').resolve_right hδk
  set suf := spec.suffixEquiv Z hZ h2 k hk
  have h_tele_LI : LinearIndependent K (suf.toLinearMap ∘ mp) :=
    hmp_LI.map' suf.toLinearMap suf.ker
  have h_eq : (suf.toLinearMap ∘ mp) = (fun s => spec.teleTerm Z Z' k (v s)) := by
    funext s
    show suf.toLinearMap (mid (pre (v s))) = spec.teleTerm Z Z' k (v s)
    rw [spec.suffixEquiv_toLinearMap Z hZ h2 k hk]
    show (chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) _)) _ = _
    rw [spec.teleTerm_factor Z Z' k hk, spec.layer_diff_eq_perturbationGen Z Z_repl F k hk]
    rfl
  rw [h_eq] at h_tele_LI
  let R := LinearMap.range (spec.teleTerm Z Z' k)
  have h_mem : ∀ s, spec.teleTerm Z Z' k (v s) ∈ R :=
    fun s => LinearMap.mem_range_self _ _
  let w : Fin (spec.L ⟨k, hk⟩) → R := fun s => ⟨spec.teleTerm Z Z' k (v s), h_mem s⟩
  have hw_LI : LinearIndependent K w := by
    have hcomp : R.subtype ∘ w = fun s => spec.teleTerm Z Z' k (v s) := rfl
    exact LinearIndependent.of_comp R.subtype (by rw [hcomp]; exact h_tele_LI)
  have := hw_LI.fintype_card_le_finrank
  simpa using this

/-- **Headline F5-gen: per-term rank equality (generalized).** -/
theorem teleTerm_rank_eq_gen
    [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : spec.OneFaultPerLayer F)
    (k : ℕ) (hk : k < spec.N) :
    Module.finrank K
        (LinearMap.range (spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k))
      = spec.L ⟨k, hk⟩ :=
  le_antisymm
    (spec.teleTerm_rank_le_gen hZ h2 hδ hF k hk)
    (spec.teleTerm_rank_ge_gen hZ h2 hδ hF k hk)
end LayerSpec
end NttFaultRank
