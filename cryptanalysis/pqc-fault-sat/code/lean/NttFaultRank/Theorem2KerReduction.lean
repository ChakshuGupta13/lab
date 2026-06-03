/-
F10 reduction lemma — kernel of `faultDiff` ⟺ per-layer `bSide` vanishing.

This file reduces the (⊆) direction of Theorem 2 to a precise per-layer
combinatorial claim, using F8's asymmetric Disjoint as the key reduction
step (NOT requiring F9's full lower bound).

Headline:
  `ker(faultDiff Z F) = { v | ∀ k < N, ∀ s, bSide (runPrefix Z' k v) ⟨k,_⟩ g₀^k s = 0 }`

The remaining proof obligation to close Theorem 2 (⊆) is the descent
argument: a vector satisfying these per-layer constraints lies in
`span {basis 0, basis 1}`. That descent is Path A (open).
-/

import NttFaultRank.PiInjective

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Helper: descending peel of a sum of disjoint range elements -/

/-- **Asymmetric-disjoint descending peel.** If `∑ k ∈ range N, T_k v = 0` and each
    range(T_k) is disjoint from `⨆_{j < k} range(T_j)`, then every `T_k v = 0`.

    Proof by strong descending induction: at top index `K = N - 1`,
    rewrite `T_K v = -∑_{k < K} T_k v`; LHS ∈ range(T_K), RHS ∈ ⨆_{k < K} range(T_k),
    so by Disjoint both sides = 0. Recurse on `K - 1`. -/
lemma each_zero_of_sum_zero_disjoint_lower
    (T : ℕ → ((Fin spec.n → K) →ₗ[K] (Fin spec.n → K)))
    (N : ℕ)
    (hdisj : ∀ a : Fin N,
      Disjoint
        (LinearMap.range (T a.val))
        (⨆ b : Fin N, ⨆ _ : b.val < a.val, LinearMap.range (T b.val)))
    (v : Fin spec.n → K)
    (hsum : ∑ k ∈ Finset.range N, T k v = 0) :
    ∀ k : Fin N, T k.val v = 0 := by
  -- Induction on m = N - 1 - k.val, i.e. peel from the top.
  -- Equivalently: strong induction on m ≤ N saying "all k with N - m ≤ k.val < N have T k v = 0
  --                                                AND ∑_{k < N - m} T k v = 0".
  suffices h_peeled : ∀ m : ℕ, m ≤ N →
      (∀ k : Fin N, N - m ≤ k.val → T k.val v = 0) ∧
      (∑ k ∈ Finset.range (N - m), T k v = 0) by
    intro k
    have ⟨h1, _⟩ := h_peeled N le_rfl
    exact h1 k (by have := k.isLt; omega)
  intro m hmN
  induction m with
  | zero =>
    refine ⟨?_, ?_⟩
    · intro k hk
      have := k.isLt; omega
    · simpa [Nat.sub_zero] using hsum
  | succ m ih =>
    have hmN' : m ≤ N := Nat.le_of_succ_le hmN
    obtain ⟨ih_zero, ih_sum⟩ := ih hmN'
    -- Step 1: peel the top index K := N - m - 1 from the sum.
    have hKlt : N - m - 1 < N := by omega
    set K : Fin N := ⟨N - m - 1, hKlt⟩ with hK_def
    have hK_succ : N - m = K.val + 1 := by show N - m = (N - m - 1) + 1; omega
    have hsum_split :
        ∑ k ∈ Finset.range (N - m), T k v
          = (∑ k ∈ Finset.range K.val, T k v) + T K.val v := by
      rw [hK_succ]; rw [Finset.sum_range_succ]
    have hT_K : T K.val v = -(∑ k ∈ Finset.range K.val, T k v) := by
      have hcombine : (∑ k ∈ Finset.range K.val, T k v) + T K.val v = 0 := by
        rw [← hsum_split]; exact ih_sum
      have : T K.val v = 0 - (∑ k ∈ Finset.range K.val, T k v) := by
        rw [eq_sub_iff_add_eq]; rw [add_comm]; exact hcombine
      rw [this]; ring
    -- Step 2: LHS ∈ range(T K). RHS = -∑ ∈ ⨆_{b < K} range(T_b).
    have h_lhs_in : T K.val v ∈ LinearMap.range (T K.val) :=
      LinearMap.mem_range_self _ _
    have h_rhs_in : -(∑ k ∈ Finset.range K.val, T k v)
                  ∈ ⨆ b : Fin N, ⨆ _ : b.val < K.val, LinearMap.range (T b.val) := by
      apply Submodule.neg_mem
      apply Submodule.sum_mem
      intro k hk
      have hkK : k < K.val := Finset.mem_range.mp hk
      have hkN : k < N := by have := K.isLt; omega
      refine Submodule.mem_iSup_of_mem ⟨k, hkN⟩ ?_
      refine Submodule.mem_iSup_of_mem hkK ?_
      exact LinearMap.mem_range_self _ _
    -- Step 3: T K v = 0 by disjointness.
    have h_in_inter :
        T K.val v ∈ (LinearMap.range (T K.val)) ⊓
          (⨆ b : Fin N, ⨆ _ : b.val < K.val, LinearMap.range (T b.val)) := by
      refine ⟨h_lhs_in, ?_⟩
      rw [hT_K]; exact h_rhs_in
    have hT_K_zero : T K.val v = 0 := by
      have hd := hdisj K
      rw [Submodule.disjoint_def] at hd
      exact hd _ h_in_inter.1 h_in_inter.2
    -- Step 4: assemble the new ih.
    refine ⟨?_, ?_⟩
    · intro k hk_ge
      by_cases hcase : k.val = K.val
      · rw [hcase]; exact hT_K_zero
      · apply ih_zero
        have : k.val ≠ N - m - 1 := hcase
        have hk_ge_old : N - (m + 1) ≤ k.val := hk_ge
        omega
    · -- New partial sum at length K.val: from ih_sum (length K.val + 1) drop the top term.
      have hsum_split' :
          ∑ k ∈ Finset.range (N - m), T k v
            = (∑ k ∈ Finset.range K.val, T k v) + T K.val v := hsum_split
      have : (∑ k ∈ Finset.range K.val, T k v) + T K.val v = 0 := by
        rw [← hsum_split']; exact ih_sum
      rw [hT_K_zero, add_zero] at this
      have hKeq : N - (m + 1) = K.val := by show N - (m + 1) = N - m - 1; omega
      rw [hKeq]; exact this

/-! ### F10 reduction theorem -/

/-- **F10 reduction.** `v ∈ ker(faultDiff Z F)` iff every per-layer faulted
    b-side vanishes:
    `∀ k < N, ∀ s, bSide (runPrefix (faultedTwiddles Z F) k v) ⟨k,_⟩ g₀^k s = 0`.

    Combined with F8's `pert_vanish_of_bSide_zero` and `teleTerm_eq_zero_of_pert_zero`,
    plus this file's `each_zero_of_sum_zero_disjoint_lower`, this characterises
    the kernel without reference to rank. The (⊆) direction of Theorem 2 thus
    reduces to: a `v` satisfying ALL per-layer b-side vanishing must lie in
    `span {basis 0, basis 1}`. -/
theorem mem_ker_faultDiff_iff_bSide_vanish
    [KyberLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (v : Fin spec.n → K) :
    v ∈ LinearMap.ker (spec.faultDiff Z F) ↔
      ∀ k : Fin spec.N, ∀ s : Fin (spec.L k),
        spec.bSide (spec.runPrefix (spec.faultedTwiddles Z F) k.val
          (Nat.le_of_lt k.isLt) v) k
          (Finset.card_eq_one.mp (hF k)).choose s = 0 := by
  constructor
  · -- Forward: v ∈ ker ⇒ ∀ k s, bSide vanishes.
    intro hv k s
    rw [LinearMap.mem_ker] at hv
    -- faultDiff v = 0; by diff_ntt_telescope, ∑ teleTerm k v = 0.
    have h_sum : ∑ k ∈ Finset.range spec.N,
                    spec.teleTerm Z (spec.faultedTwiddles Z F) k v = 0 := by
      have h := spec.diff_ntt_telescope Z F
      have h_apply := congrArg (fun L => L v) h
      simp only [LinearMap.sub_apply, Finset.sum_apply] at h_apply
      -- h_apply : (ntt Z) v - (nttFault Z F) v = ∑ ... v (after coeFn_sum unfold).
      -- hv : (ntt Z - nttFault Z F) v = 0; this unfolds to (ntt Z) v - (nttFault Z F) v = 0.
      have h_lhs : (spec.ntt Z) v - (spec.nttFault Z F) v = 0 := by
        have := hv
        unfold faultDiff at this
        rw [LinearMap.sub_apply] at this; exact this
      rw [h_lhs] at h_apply
      have h_rhs :
          (∑ k ∈ Finset.range spec.N,
              spec.teleTerm Z (spec.faultedTwiddles Z F) k) v
            = ∑ k ∈ Finset.range spec.N,
                spec.teleTerm Z (spec.faultedTwiddles Z F) k v := by
        rw [LinearMap.coeFn_sum, Finset.sum_apply]
      rw [h_rhs] at h_apply
      exact h_apply.symm
    -- Apply each_zero_of_sum_zero_disjoint_lower at index k.
    have h_each : ∀ a : Fin spec.N,
        spec.teleTerm Z (spec.faultedTwiddles Z F) a.val v = 0 :=
      spec.each_zero_of_sum_zero_disjoint_lower
        (spec.teleTerm Z (spec.faultedTwiddles Z F))
        spec.N
        (fun a => spec.teleTerm_image_disjoint_lower hZ h2 hF a) v h_sum
    have hT_k : spec.teleTerm Z (spec.faultedTwiddles Z F) k.val v = 0 := h_each k
    -- teleTerm k v = 0 ⇒ pert k (runPrefix Z' k v) = 0 ⇒ bSide vanishes.
    -- Use teleTerm_factor + suffix LinearEquiv injectivity.
    have h_pert : spec.perturbation Z F k
                    (spec.runPrefix (spec.faultedTwiddles Z F) k.val
                      (Nat.le_of_lt k.isLt) v) = 0 := by
      -- teleTerm Z Z' k v = suffix (pert (runPrefix Z' k v)). suffix injective ⇒ pert (runPrefix Z' k v) = 0.
      have hfact := spec.teleTerm_factor Z (spec.faultedTwiddles Z F) k.val k.isLt
      have h_apply := congrArg (fun L => L v) hfact
      simp only at h_apply
      rw [hT_k] at h_apply
      -- h_apply : 0 = suffix(layer_diff(runPrefix v))
      rw [spec.layer_diff_eq_perturbation Z F k.val k.isLt] at h_apply
      simp only [LinearMap.comp_apply] at h_apply
      -- suffix is the chainList; it's injective (LinearEquiv via chainListEquiv).
      have hsuf_inj : Function.Injective
          (chainList spec Z (spec.suffixList (k.val + 1) (spec.N - k.val - 1) (by omega))) := by
        rw [← spec.suffixEquiv_toLinearMap Z hZ h2 k.val k.isLt]
        exact (spec.suffixEquiv Z hZ h2 k.val k.isLt).injective
      -- Apply injectivity: suffix (pert (...)) = suffix 0 ⇒ pert (...) = 0.
      have hzero : (chainList spec Z (spec.suffixList (k.val + 1) (spec.N - k.val - 1) (by omega)))
                    (0 : Fin spec.n → K) = 0 := by
        show _ = _; exact map_zero _
      apply hsuf_inj
      rw [hzero, ← h_apply]
    -- Now apply bSide of pert = 0.
    -- pert k w = 0 implies bSide w g₀ s = 0? We need the CONVERSE direction.
    -- That direction is straightforward: bSide (pert w) g₀ s involves bSide w
    -- (the input to pert), via aSide_perturbation / bSide_perturbation in F8.
    -- Actually: pert k w = layer Z k w - layer Z' k w. At g₀, Z' = 0, so
    -- bSide(pert w) g₀ s = -Z g₀ * bSide w g₀ s. So bSide w g₀ s = 0 follows from
    -- bSide(pert w) g₀ s = 0 + Z g₀ ≠ 0.
    set g₀ := (Finset.card_eq_one.mp (hF k)).choose with hg₀_def
    have h_bSide_pert : spec.bSide (spec.perturbation Z F k
        (spec.runPrefix (spec.faultedTwiddles Z F) k.val
          (Nat.le_of_lt k.isLt) v)) k g₀ s = 0 := by
      rw [h_pert, spec.bSide_eq_apply]; rfl
    -- bSide_perturbation: bSide(pert w) g j = (Z' g - Z g) * bSide w g j.
    rw [spec.bSide_perturbation Z F k _ g₀ s] at h_bSide_pert
    -- (Z' g₀ - Z g₀) * bSide w g₀ s = 0; Z' g₀ = 0 so this is -Z g₀ * bSide w g₀ s.
    have hZ'_zero : (spec.faultedTwiddles Z F) k g₀ = 0 := by
      have hg₀_spec : F k = {g₀} :=
        (Finset.card_eq_one.mp (hF k)).choose_spec
      show (if g₀ ∈ F k then 0 else Z k g₀) = 0
      rw [hg₀_spec, if_pos (Finset.mem_singleton_self _)]
    rw [hZ'_zero, zero_sub] at h_bSide_pert
    -- -Z k g₀ * bSide ... = 0. Z k g₀ ≠ 0 ⇒ bSide = 0.
    have hZk : Z k g₀ ≠ 0 := hZ k g₀
    have hneg : -Z k g₀ ≠ 0 := neg_ne_zero.mpr hZk
    exact (mul_eq_zero.mp h_bSide_pert).resolve_left hneg
  · -- Backward: ∀ k s bSide vanishes ⇒ v ∈ ker.
    intro h_van
    rw [LinearMap.mem_ker]
    -- faultDiff v = ∑ teleTerm k v. Each teleTerm k v = 0 by F8's chain.
    have h_each : ∀ k : Fin spec.N,
        spec.teleTerm Z (spec.faultedTwiddles Z F) k.val v = 0 := by
      intro k
      have h_van_k := h_van k
      -- pert k (runPrefix Z' k v) = 0 via F8's pert_vanish_of_bSide_zero.
      have h_pert := spec.pert_vanish_of_bSide_zero Z hF k.val k.isLt
        (spec.runPrefix (spec.faultedTwiddles Z F) k.val (Nat.le_of_lt k.isLt) v)
        h_van_k
      exact spec.teleTerm_eq_zero_of_pert_zero F k.val k.isLt v h_pert
    -- ∑ teleTerm k v = 0.
    have h_sum : ∑ k ∈ Finset.range spec.N,
                  spec.teleTerm Z (spec.faultedTwiddles Z F) k v = 0 := by
      apply Finset.sum_eq_zero
      intro k hk
      have hkN : k < spec.N := Finset.mem_range.mp hk
      exact h_each ⟨k, hkN⟩
    -- faultDiff v = (ntt Z - nttFault Z F) v = ∑ teleTerm k v = 0 by F1.
    have h := spec.diff_ntt_telescope Z F
    have h_apply := congrArg (fun L => L v) h
    show (spec.faultDiff Z F) v = 0
    unfold faultDiff
    rw [LinearMap.sub_apply]
    simp only [LinearMap.sub_apply, Finset.sum_apply] at h_apply
    rw [h_apply]
    -- RHS: ∑ teleTerm k v applied.
    rw [LinearMap.coeFn_sum, Finset.sum_apply]
    exact h_sum

end LayerSpec
end NttFaultRank
