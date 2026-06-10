/-
Phase F — Assembly: closing the (⊆) direction of Theorem 2.

Paper §"Assembly" of the Theorem 2 proof. Given:
  - per-term rank `teleTerm_rank_eq` (Lemma 4 / TeleTermRank.lean),
  - image disjointness `teleTerm_image_disjoint_lower` (Lemma 2+3
    composed / PiInjective.lean),
  - the support-pattern lemma `fellPrefix_basis_eq_pattern`
    (Lemma 4 / Fellsupport.lean),
this file proves `rank(faultDiff) ≥ n - L_{N-1}`, which combined with
the matching upper bound and the kernel inclusion closes the kernel
equality `ker D_K = span {e_0, e_1}`.

Key lemma:
  `teleTerm_basis_freshIdx_lower_eq_zero`
    `T_{k'} (basis (freshIdx hk s)) = 0` whenever `k' < k`.
  This is the paper's observation "for j ∈ I_l with l' < l, every
  support position of F_{l'}(e_j) has bit β_{l'} = 0, hence is an
  a-input; the b-input to P_{l'} is zero and T_{l'}(e_j) = 0".

Headline:
  `faultDiff_rank_ge_sum_L` (used in `Theorem2Final` to close
  the zeroing case; the generalised analogue
  `faultDiffGen_rank_ge_sum_L` closes the arbitrary-δ case).
-/

import NttFaultRank.Theorem2KerReduction
import NttFaultRank.Fellsupport
import NttFaultRank.Theorem2Subset

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Step 1 — Off-layer vanishing of `T_{k'}(basis (freshIdx hk s))`. -/

/-- **Step 1 (paper "Assembly" block, first sentence).** For `k' < k`, the
    telescope term `T_{k'}` annihilates the basis vector `e_{freshIdx hk s}`.

    Proof sketch: by `fellPrefix_basis_eq_pattern`, the support of
    `runPrefix Z' k' (basis j)` for `j = L_k + s` (with `j.val < 2 L_k ≤
    L_{k'} < 2 L_{k'}`) consists of indices `i` with
    `i.val % (2 L_{k'}) = j.val = L_k + s`. The b-side at layer-`k'`
    group `g₀` position `j_b` has flat index
    `bSideIdx_{k'} g₀ j_b = g₀ · 2 L_{k'} + L_{k'} + j_b`, whose mod is
    `L_{k'} + j_b ≥ L_{k'} ≥ 2 L_k > L_k + s = j.val`, contradicting
    membership in the support. So every b-side value is zero, the
    perturbation vanishes, and `T_{k'}(basis j) = 0`. -/
theorem teleTerm_basis_freshIdx_lower_eq_zero [CooleyTukeyLike spec]
    (Z : spec.Twiddles K)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    {k : ℕ} (hk : k < spec.N) (s : Fin (spec.L ⟨k, hk⟩))
    {k' : ℕ} (hk' : k' < spec.N) (hk'k : k' < k) :
    spec.teleTerm Z (spec.faultedTwiddles Z F) k'
        (spec.basis (K := K) (spec.freshIdx hk s)) = 0 := by
  -- Reduce to: perturbation at layer k' of runPrefix Z' k' (basis j) = 0.
  apply spec.teleTerm_eq_zero_of_pert_zero F k' hk'
  -- Reduce further to: bSide at faulted group g₀ vanishes for every j_b.
  apply spec.pert_vanish_of_bSide_zero Z hF k' hk'
  intro j_b
  -- Set j := freshIdx hk s, the basis index. j.val = L_k + s.val < 2 L_k.
  set j : Fin spec.n := spec.freshIdx hk s with hj_def
  -- Need: j.val < 2 * L_{k'}.
  have hLk_pos : 0 < spec.L ⟨k, hk⟩ := by
    have := s.isLt; omega
  have hn_pos : 0 < spec.n := by
    have := (spec.freshIdx hk s).isLt; omega
  have hLk'_pos : 0 < spec.L ⟨k', hk'⟩ := spec.L_pos k' hk' hn_pos
  have h_two_L_k_le : 2 * spec.L ⟨k, hk⟩ ≤ spec.L ⟨k', hk'⟩ :=
    spec.two_L_le_L_of_lt hk' hk hk'k
  have hj_val : j.val = spec.L ⟨k, hk⟩ + s.val := rfl
  have hs : s.val < spec.L ⟨k, hk⟩ := s.isLt
  have hj_lt : j.val < 2 * spec.L ⟨k', hk'⟩ := by omega
  -- Set g₀ := faulted group at layer k'.
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k', hk'⟩)).choose with hg₀_def
  -- Goal: bSide (runPrefix Z' k' (basis j)) ⟨k', hk'⟩ g₀ j_b = 0.
  change spec.bSide _ ⟨k', hk'⟩ g₀ j_b = 0
  rw [spec.bSide_eq_apply]
  -- Compute via Fellsupport at layer ⟨k', hk'⟩.
  rw [spec.fellPrefix_basis_eq_pattern (spec.faultedTwiddles Z F)
        ⟨k', hk'⟩ j hj_lt]
  -- Reduce to: bSideIdx ⟨k',_⟩ g₀ j_b ∉ fellPattern ⟨k',_⟩ j.
  rw [if_neg]
  intro hmem
  unfold fellPattern at hmem
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hmem
  -- hmem : (bSideIdx ⟨k', hk'⟩ g₀ j_b).val % (2 * L_{k'}) = j.val.
  rw [spec.bSideIdx_val] at hmem
  -- hmem : (g₀.val * (2 * L_{k'}) + L_{k'} + j_b.val) % (2 * L_{k'}) = j.val.
  have hj_b : j_b.val < spec.L ⟨k', hk'⟩ := j_b.isLt
  -- Reduce the mod: (m * (2L) + r) % (2L) = r when r < 2L.
  have h_inner_lt : spec.L ⟨k', hk'⟩ + j_b.val < 2 * spec.L ⟨k', hk'⟩ := by omega
  have h_mod : (g₀.val * (2 * spec.L ⟨k', hk'⟩) + spec.L ⟨k', hk'⟩ + j_b.val)
                  % (2 * spec.L ⟨k', hk'⟩) = spec.L ⟨k', hk'⟩ + j_b.val := by
    have : g₀.val * (2 * spec.L ⟨k', hk'⟩) + spec.L ⟨k', hk'⟩ + j_b.val
            = spec.L ⟨k', hk'⟩ + j_b.val + g₀.val * (2 * spec.L ⟨k', hk'⟩) := by
      ring
    rw [this, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h_inner_lt]
  rw [h_mod] at hmem
  -- hmem : L_{k'} + j_b.val = j.val = L_k + s.val. But L_{k'} ≥ 2 L_k > L_k + s.
  omega

/-- Gen version: `T_{k'}(basis(freshIdx_k s)) = 0` for `k' < k`, arbitrary Z'. -/
theorem teleTerm_basis_freshIdx_lower_eq_zero_gen [CooleyTukeyLike spec]
    (Z Z_repl : spec.Twiddles K)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    {k : ℕ} (hk : k < spec.N) (s : Fin (spec.L ⟨k, hk⟩))
    {k' : ℕ} (hk' : k' < spec.N) (hk'k : k' < k) :
    spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k'
        (spec.basis (K := K) (spec.freshIdx hk s)) = 0 := by
  apply spec.teleTerm_eq_zero_of_pertGen_zero Z_repl F k' hk'
  apply spec.pertGen_vanish_of_bSide_zero Z Z_repl hF k' hk'
  intro j_b
  set j : Fin spec.n := spec.freshIdx hk s
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k', hk'⟩)).choose
  change spec.bSide _ ⟨k', hk'⟩ g₀ j_b = 0
  rw [spec.bSide_eq_apply]
  have hj_lt : j.val < 2 * spec.L ⟨k', hk'⟩ := by
    have h_two_L_k_le := spec.two_L_le_L_of_lt hk' hk hk'k
    have hs := s.isLt; have : j.val = spec.L ⟨k, hk⟩ + s.val := rfl; omega
  rw [spec.fellPrefix_basis_eq_pattern (spec.faultedTwiddlesGen Z Z_repl F)
        ⟨k', hk'⟩ j hj_lt]
  rw [if_neg]; intro hmem
  unfold fellPattern at hmem
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hmem
  rw [spec.bSideIdx_val] at hmem
  have hj_b : j_b.val < spec.L ⟨k', hk'⟩ := j_b.isLt
  have h_inner_lt : spec.L ⟨k', hk'⟩ + j_b.val < 2 * spec.L ⟨k', hk'⟩ := by omega
  have h_mod : (g₀.val * (2 * spec.L ⟨k', hk'⟩) + spec.L ⟨k', hk'⟩ + j_b.val)
                  % (2 * spec.L ⟨k', hk'⟩) = spec.L ⟨k', hk'⟩ + j_b.val := by
    rw [show g₀.val * (2 * spec.L ⟨k', hk'⟩) + spec.L ⟨k', hk'⟩ + j_b.val
          = spec.L ⟨k', hk'⟩ + j_b.val + g₀.val * (2 * spec.L ⟨k', hk'⟩) from by ring,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h_inner_lt]
  rw [h_mod] at hmem
  have h_two_L_k_le := spec.two_L_le_L_of_lt hk' hk hk'k
  have hs := s.isLt; have : j.val = spec.L ⟨k, hk⟩ + s.val := rfl; omega

/-! ### Step 2 — Linear independence of `{D_K (basis (freshIdx hk s))}`. -/

/-- A combined input from the per-layer fresh-index basis vectors.

    For coefficient family `c k s : K` (defined for `k < N`, `s : Fin (L k)`),
    `combinedFreshBasis c = ∑_{k, s} c k s · basis (freshIdx hk s)`.
    This is the "test vector" used to probe linear independence in the
    Assembly argument. -/
noncomputable def combinedFreshBasis [CooleyTukeyLike spec]
    (c : ∀ (k : Fin spec.N), Fin (spec.L k) → K) : Fin spec.n → K :=
  ∑ k : Fin spec.N, ∑ s : Fin (spec.L k),
    c k s • spec.basis (K := K) (spec.freshIdx k.isLt s)

/-- `T_{k'}` applied to a `combinedFreshBasis` vector keeps only the
    `k ≥ k'` terms (the `k < k'` terms vanish by Step 1; equivalently,
    via index-swap, only the `k ≤ k'` block remains in the standard
    block-triangular ordering of the Assembly argument). -/
lemma teleTerm_combinedFreshBasis [CooleyTukeyLike spec]
    (Z : spec.Twiddles K)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (c : ∀ (k : Fin spec.N), Fin (spec.L k) → K)
    {k' : ℕ} (hk' : k' < spec.N) :
    spec.teleTerm Z (spec.faultedTwiddles Z F) k'
        (spec.combinedFreshBasis c) =
      ∑ k : Fin spec.N, ∑ s : Fin (spec.L k),
        if k.val ≤ k' then
          c k s • spec.teleTerm Z (spec.faultedTwiddles Z F) k'
            (spec.basis (K := K) (spec.freshIdx k.isLt s))
        else 0 := by
  unfold combinedFreshBasis
  rw [map_sum]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [map_sum]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [map_smul]
  by_cases hk_le : k.val ≤ k'
  · simp [hk_le]
  · -- k' < k.val: by Step 1, T_{k'}(basis (freshIdx hk s)) = 0.
    push_neg at hk_le
    rw [spec.teleTerm_basis_freshIdx_lower_eq_zero Z hF k.isLt s hk' hk_le,
        smul_zero, if_neg (Nat.not_le_of_lt hk_le)]


/-! ### Step 2 — Per-term linear independence on `{basis (freshIdx _ s)}`. -/

/-- **Per-term LI on the fresh-index basis.** The L_k vectors
    `{T_k (basis (freshIdx hk s)) : s : Fin (L_k)}` are linearly
    independent. Extracted from the inner LI step of
    `TeleTermRank.teleTerm_rank_ge` for direct use in the assembly. -/
theorem teleTerm_basis_freshIdx_linearIndependent
    [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    {k : ℕ} (hk : k < spec.N) :
    LinearIndependent K
      (fun s : Fin (spec.L ⟨k, hk⟩) =>
        spec.teleTerm Z (spec.faultedTwiddles Z F) k
          (spec.basis (K := K) (spec.freshIdx hk s))) := by
  set Z' := spec.faultedTwiddles Z F with hZ'_def
  set pre := spec.runPrefix Z' k (Nat.le_of_lt hk) with hpre_def
  set mid := spec.perturbation Z F ⟨k, hk⟩ with hmid_def
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  let v : Fin (spec.L ⟨k, hk⟩) → (Fin spec.n → K) := fun s =>
    spec.basis (K := K) (spec.freshIdx hk s)
  let mp : Fin (spec.L ⟨k, hk⟩) → (Fin spec.n → K) := fun s => mid (pre (v s))
  -- LI of mp (mirrors TeleTermRank.teleTerm_rank_ge proof).
  have hZk_ne : Z ⟨k, hk⟩ g₀ ≠ 0 := hZ ⟨k, hk⟩ g₀
  have hmp_LI : LinearIndependent K mp := by
    rw [Fintype.linearIndependent_iff]
    intro c hsum s'
    have h_at := congrArg (fun w => spec.aSide w ⟨k, hk⟩ g₀ s') hsum
    simp only at h_at
    have hsum_aSide :
        spec.aSide (∑ s, c s • mp s) ⟨k, hk⟩ g₀ s'
          = ∑ s, c s * spec.aSide (mp s) ⟨k, hk⟩ g₀ s' := by
      change (∑ s, c s • mp s) (spec.aSideIdx ⟨k, hk⟩ g₀ s')
           = ∑ s, c s * spec.aSide (mp s) ⟨k, hk⟩ g₀ s'
      rw [Finset.sum_apply]
      apply Finset.sum_congr rfl
      intros s _
      simp [Pi.smul_apply, smul_eq_mul, aSide_eq_apply]
    rw [hsum_aSide] at h_at
    have hcalc : ∀ s : Fin (spec.L ⟨k, hk⟩),
        spec.aSide (mp s) ⟨k, hk⟩ g₀ s' = if s' = s then Z ⟨k, hk⟩ g₀ else 0 := by
      intro s
      exact spec.aSide_mid_pre_basis Z hF k hk s s'
    have h_at' : ∑ s, c s * (if s' = s then Z ⟨k, hk⟩ g₀ else 0)
               = (spec.aSide (0 : Fin spec.n → K) ⟨k, hk⟩ g₀ s') := by
      rw [← h_at]; apply Finset.sum_congr rfl; intros s _; rw [hcalc s]
    have hzero : spec.aSide (0 : Fin spec.n → K) ⟨k, hk⟩ g₀ s' = 0 := by
      rw [aSide_eq_apply]; rfl
    rw [hzero] at h_at'
    have hsingle : ∑ s, c s * (if s' = s then Z ⟨k, hk⟩ g₀ else 0)
                = c s' * Z ⟨k, hk⟩ g₀ := by
      rw [Finset.sum_eq_single s']
      · simp
      · intros s _ hne; simp [Ne.symm hne]
      · intro h; exact absurd (Finset.mem_univ _) h
    rw [hsingle] at h_at'
    have : c s' * Z ⟨k, hk⟩ g₀ = 0 := h_at'
    exact (mul_eq_zero.mp this).resolve_right hZk_ne
  -- Lift via suffix LinearEquiv.
  set suf := spec.suffixEquiv Z hZ h2 k hk
  have hsuf_inj : LinearMap.ker (suf.toLinearMap) = ⊥ := suf.ker
  have h_tele_LI : LinearIndependent K (suf.toLinearMap ∘ mp) :=
    hmp_LI.map' suf.toLinearMap hsuf_inj
  have h_eq : (suf.toLinearMap ∘ mp) =
      (fun s => spec.teleTerm Z Z' k (v s)) := by
    funext s
    change suf.toLinearMap (mid (pre (v s))) = spec.teleTerm Z Z' k (v s)
    rw [spec.suffixEquiv_toLinearMap Z hZ h2 k hk]
    change (chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) _)) _ = _
    rw [spec.teleTerm_factor Z Z' k hk, spec.layer_diff_eq_perturbation Z F k hk]
    rfl
  rw [h_eq] at h_tele_LI
  exact h_tele_LI

end LayerSpec
end NttFaultRank

-- Re-open namespace for Step 3.
namespace NttFaultRank
variable {K : Type*} [Field K]
namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Step 3 — LI of `{D_K (basis (freshIdx _ s))}_{(k,s)}` and rank bound. -/

/-- **Combined-input linearity.** `D_K (combinedFreshBasis c) =
    ∑_{k, s} c k s • D_K (basis (freshIdx _ s))`. -/
lemma faultDiff_combinedFreshBasis [CooleyTukeyLike spec]
    (Z : spec.Twiddles K) (F : spec.FaultSet)
    (c : ∀ (k : Fin spec.N), Fin (spec.L k) → K) :
    spec.faultDiff Z F (spec.combinedFreshBasis c) =
      ∑ k : Fin spec.N, ∑ s : Fin (spec.L k),
        c k s • spec.faultDiff Z F
          (spec.basis (K := K) (spec.freshIdx k.isLt s)) := by
  unfold combinedFreshBasis
  rw [map_sum]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [map_sum]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [map_smul]

/-- **Step 3a — coefficient annihilation.** If `D_K (combinedFreshBasis c) = 0`,
    then `c = 0`. By each-zero-of-sum-zero on the telescope plus strong
    induction on `ℓ.val` (Step 1 kills upper-triangular, per-term LI kills
    diagonal). -/
theorem combinedFreshBasis_injective [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F)
    (c : ∀ (k : Fin spec.N), Fin (spec.L k) → K)
    (hker : spec.faultDiff Z F (spec.combinedFreshBasis c) = 0) :
    ∀ (k : Fin spec.N), c k = 0 := by
  set Z' := spec.faultedTwiddles Z F with hZ'_def
  -- Step A: D_K V = 0 ⇒ ∑_ℓ T_ℓ V = 0 ⇒ ∀ ℓ, T_ℓ V = 0.
  have h_sum : ∑ k ∈ Finset.range spec.N,
                  spec.teleTerm Z Z' k (spec.combinedFreshBasis c) = 0 := by
    have h := spec.diff_ntt_telescope Z F
    have h_apply := congrArg (fun L => L (spec.combinedFreshBasis c)) h
    simp only at h_apply
    have h_lhs : (spec.ntt Z - spec.nttFault Z F)
                    (spec.combinedFreshBasis c) = 0 := by
      have := hker; change (spec.faultDiff Z F) _ = 0 at this; exact this
    rw [h_lhs] at h_apply
    have h_rhs :
        (∑ k ∈ Finset.range spec.N, spec.teleTerm Z Z' k)
          (spec.combinedFreshBasis c)
          = ∑ k ∈ Finset.range spec.N,
              spec.teleTerm Z Z' k (spec.combinedFreshBasis c) := by
      rw [LinearMap.coe_sum, Finset.sum_apply]
    rw [h_rhs] at h_apply
    exact h_apply.symm
  have hT_each : ∀ ℓ : Fin spec.N,
      spec.teleTerm Z Z' ℓ.val (spec.combinedFreshBasis c) = 0 :=
    spec.each_zero_of_sum_zero_disjoint_lower
      (spec.teleTerm Z Z') spec.N
      (fun a => spec.teleTerm_image_disjoint_lower hZ h2 hF a)
      (spec.combinedFreshBasis c) h_sum
  -- Step B: strong induction on ℓ.val.
  suffices h_strong : ∀ m : ℕ, m ≤ spec.N →
      ∀ ℓ : Fin spec.N, ℓ.val < m → c ℓ = 0 by
    intro k
    exact h_strong spec.N le_rfl k k.isLt
  intro m hm
  induction m with
  | zero => intro ℓ hℓ; exact absurd hℓ (Nat.not_lt_zero _)
  | succ m ih =>
    have hm' : m ≤ spec.N := Nat.le_of_succ_le hm
    intro ℓ hℓ
    by_cases hℓm : ℓ.val < m
    · exact ih hm' ℓ hℓm
    · have hℓ_eq : ℓ.val = m := by omega
      have hm_lt : m < spec.N := hm
      have hT_m_val : spec.teleTerm Z Z' m (spec.combinedFreshBasis c) = 0 :=
        hT_each ⟨m, hm_lt⟩
      rw [spec.teleTerm_combinedFreshBasis Z hF c hm_lt] at hT_m_val
      have h_collapse :
          (∑ k : Fin spec.N, ∑ s : Fin (spec.L k),
              if k.val ≤ m then
                c k s • spec.teleTerm Z Z' m
                  (spec.basis (K := K) (spec.freshIdx k.isLt s))
              else 0)
          = ∑ s : Fin (spec.L ⟨m, hm_lt⟩),
              c ⟨m, hm_lt⟩ s • spec.teleTerm Z Z' m
                (spec.basis (K := K) (spec.freshIdx hm_lt s)) := by
        rw [Finset.sum_eq_single (⟨m, hm_lt⟩ : Fin spec.N)]
        · simp
        · intros k _ hk_ne
          apply Finset.sum_eq_zero
          intros s _
          by_cases hk_le : k.val ≤ m
          · have hk_lt : k.val < m := by
              rcases Nat.lt_or_eq_of_le hk_le with h | h
              · exact h
              · exfalso; apply hk_ne; exact Fin.ext h
            rw [if_pos hk_le, ih hm' k hk_lt]; simp
          · simp [hk_le]
        · intro h; exact absurd (Finset.mem_univ _) h
      rw [h_collapse] at hT_m_val
      have hLI := spec.teleTerm_basis_freshIdx_linearIndependent hZ h2 hF hm_lt
      have h_c_zero : ∀ s : Fin (spec.L ⟨m, hm_lt⟩), c ⟨m, hm_lt⟩ s = 0 :=
        fun s => (Fintype.linearIndependent_iff.mp hLI) (c ⟨m, hm_lt⟩) hT_m_val s
      have hℓ_def : ℓ = ⟨m, hm_lt⟩ := Fin.ext hℓ_eq
      subst hℓ_def
      funext s; exact h_c_zero s

end LayerSpec
end NttFaultRank

-- Re-open for rank bound + headline.
namespace NttFaultRank
variable {K : Type*} [Field K]
namespace LayerSpec
variable (spec : LayerSpec)

/-! ### Step 3b — rank lower bound `rank(D_K) ≥ ∑_ℓ L_ℓ`. -/

/-- **LI of the fresh-basis images under D_K.** The family
    `{D_K (basis (freshIdx k.isLt s)) : (k, s) ∈ Σ Fin N, Fin (L k)}`
    is linearly independent. -/
theorem faultDiff_basis_freshIdx_linearIndependent [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) :
    LinearIndependent K
      (fun (p : Σ k : Fin spec.N, Fin (spec.L k)) =>
        spec.faultDiff Z F
          (spec.basis (K := K) (spec.freshIdx p.1.isLt p.2))) := by
  rw [Fintype.linearIndependent_iff]
  intro g hsum
  -- Convert the Sigma-sum to a nested sum and recognise it as
  -- D_K (combinedFreshBasis c) with c := fun k s => g ⟨k, s⟩.
  let c : ∀ (k : Fin spec.N), Fin (spec.L k) → K := fun k s => g ⟨k, s⟩
  have h_DK : spec.faultDiff Z F (spec.combinedFreshBasis c) = 0 := by
    rw [spec.faultDiff_combinedFreshBasis Z F c]
    rw [show (Finset.univ : Finset (Σ k : Fin spec.N, Fin (spec.L k)))
            = (Finset.univ : Finset (Fin spec.N)).sigma
                (fun k => (Finset.univ : Finset (Fin (spec.L k)))) from rfl,
        Finset.sum_sigma] at hsum
    exact hsum
  have h_c : ∀ k : Fin spec.N, c k = 0 :=
    spec.combinedFreshBasis_injective hZ h2 hF c h_DK
  intro p
  have := congrFun (h_c p.1) p.2
  exact this

/-- The total count of fresh-basis indices equals `∑_ℓ L_ℓ`. -/
lemma card_sigma_fresh [CooleyTukeyLike spec] :
    Fintype.card (Σ k : Fin spec.N, Fin (spec.L k))
      = ∑ k : Fin spec.N, spec.L ⟨k.val, k.isLt⟩ := by
  rw [Fintype.card_sigma]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  simp

/-- **Step 3b headline: rank lower bound.** `rank(D_K) ≥ ∑_ℓ L_ℓ`. -/
theorem faultDiff_rank_ge_sum_L [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : spec.FaultSet} (hF : spec.OneFaultPerLayer F) :
    (∑ k : Fin spec.N, spec.L ⟨k.val, k.isLt⟩)
      ≤ Module.finrank K ↑(LinearMap.range (spec.faultDiff Z F)) := by
  -- Use the LI family in range(D_K) and `LinearIndependent.fintype_card_le_finrank`.
  let R := LinearMap.range (spec.faultDiff Z F)
  let φ : (Σ k : Fin spec.N, Fin (spec.L k)) → (Fin spec.n → K) :=
    fun p => spec.faultDiff Z F
      (spec.basis (K := K) (spec.freshIdx p.1.isLt p.2))
  have hφ_LI : LinearIndependent K φ :=
    spec.faultDiff_basis_freshIdx_linearIndependent hZ h2 hF
  have h_mem : ∀ p, φ p ∈ R := fun p => LinearMap.mem_range_self _ _
  let ψ : (Σ k : Fin spec.N, Fin (spec.L k)) → R := fun p => ⟨φ p, h_mem p⟩
  have hψ_LI : LinearIndependent K ψ := by
    have hcomp : R.subtype ∘ ψ = φ := rfl
    exact LinearIndependent.of_comp R.subtype (by rw [hcomp]; exact hφ_LI)
  have h_card := hψ_LI.fintype_card_le_finrank
  rw [spec.card_sigma_fresh] at h_card
  exact h_card

/-! ### Generalized rank lower bound for arbitrary replacement twiddles -/

/-- **Gen: per-term LI at freshIdx basis.** -/
theorem teleTerm_basis_freshIdx_linearIndependent_gen
    [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : spec.OneFaultPerLayer F)
    {k : ℕ} (hk : k < spec.N) :
    LinearIndependent K
      (fun s : Fin (spec.L ⟨k, hk⟩) =>
        spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k
          (spec.basis (K := K) (spec.freshIdx hk s))) := by
  set Z' := spec.faultedTwiddlesGen Z Z_repl F with hZ'_def
  set pre := spec.runPrefix Z' k (Nat.le_of_lt hk)
  set mid := spec.perturbationGen Z Z_repl F ⟨k, hk⟩
  set g₀ := (Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose with hg₀_def
  let v : Fin (spec.L ⟨k, hk⟩) → (Fin spec.n → K) := fun s =>
    spec.basis (K := K) (spec.freshIdx hk s)
  let mp : Fin (spec.L ⟨k, hk⟩) → (Fin spec.n → K) := fun s => mid (pre (v s))
  have hg₀_mem : g₀ ∈ F ⟨k, hk⟩ := by
    rw [(Finset.card_eq_one.mp (hF ⟨k, hk⟩)).choose_spec]; exact Finset.mem_singleton_self g₀
  have hδk : Z ⟨k, hk⟩ g₀ - Z_repl ⟨k, hk⟩ g₀ ≠ 0 := sub_ne_zero.mpr (hδ ⟨k, hk⟩ g₀ hg₀_mem)
  have hmp_LI : LinearIndependent K mp := by
    rw [Fintype.linearIndependent_iff]
    intro c hsum s'
    have h_at := congrArg (fun w => spec.aSide w ⟨k, hk⟩ g₀ s') hsum
    simp only at h_at
    have hsum_aSide :
        spec.aSide (∑ s, c s • mp s) ⟨k, hk⟩ g₀ s'
          = ∑ s, c s * spec.aSide (mp s) ⟨k, hk⟩ g₀ s' := by
      change (∑ s, c s • mp s) (spec.aSideIdx ⟨k, hk⟩ g₀ s')
           = ∑ s, c s * spec.aSide (mp s) ⟨k, hk⟩ g₀ s'
      rw [Finset.sum_apply]
      apply Finset.sum_congr rfl; intros s _
      simp [Pi.smul_apply, smul_eq_mul, aSide_eq_apply]
    rw [hsum_aSide] at h_at
    have hcalc : ∀ s,
        spec.aSide (mp s) ⟨k, hk⟩ g₀ s'
          = if s' = s then Z ⟨k, hk⟩ g₀ - Z_repl ⟨k, hk⟩ g₀ else 0 :=
      fun s => spec.aSide_mid_pre_basis_gen Z hF k hk s s'
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
  have h_eq : (suf.toLinearMap ∘ mp) =
      (fun s => spec.teleTerm Z Z' k (v s)) := by
    funext s
    show suf.toLinearMap (mid (pre (v s))) = spec.teleTerm Z Z' k (v s)
    rw [spec.suffixEquiv_toLinearMap Z hZ h2 k hk]
    show (chainList spec Z (spec.suffixList (k + 1) (spec.N - k - 1) _)) _ = _
    rw [spec.teleTerm_factor Z Z' k hk, spec.layer_diff_eq_perturbationGen Z Z_repl F k hk]; rfl
  rw [h_eq] at h_tele_LI
  exact h_tele_LI

/-! ### Gen: combinedFreshBasis injective + rank lower bound -/

/-- Gen: `T_{k'}` applied to a `combinedFreshBasis` with gen twiddles. -/
lemma teleTerm_combinedFreshBasis_gen [CooleyTukeyLike spec]
    (Z : spec.Twiddles K)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hF : spec.OneFaultPerLayer F)
    (c : ∀ (k : Fin spec.N), Fin (spec.L k) → K)
    {k' : ℕ} (hk' : k' < spec.N) :
    spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k'
        (spec.combinedFreshBasis c) =
      ∑ k : Fin spec.N, ∑ s : Fin (spec.L k),
        if k.val ≤ k' then
          c k s • spec.teleTerm Z (spec.faultedTwiddlesGen Z Z_repl F) k'
            (spec.basis (K := K) (spec.freshIdx k.isLt s))
        else 0 := by
  unfold combinedFreshBasis
  rw [map_sum]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [map_sum]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [map_smul]
  by_cases hk_le : k.val ≤ k'
  · simp [hk_le]
  · push_neg at hk_le
    rw [spec.teleTerm_basis_freshIdx_lower_eq_zero_gen Z Z_repl hF k.isLt s hk' hk_le,
        smul_zero, if_neg (Nat.not_le_of_lt hk_le)]

/-- Gen: `faultDiffGen (combinedFreshBasis c) = ∑ c k s • faultDiffGen (basis ...)`. -/
lemma faultDiffGen_combinedFreshBasis [CooleyTukeyLike spec]
    (Z Z_repl : spec.Twiddles K) (F : spec.FaultSet)
    (c : ∀ (k : Fin spec.N), Fin (spec.L k) → K) :
    spec.faultDiffGen Z Z_repl F (spec.combinedFreshBasis c) =
      ∑ k : Fin spec.N, ∑ s : Fin (spec.L k),
        c k s • spec.faultDiffGen Z Z_repl F
          (spec.basis (K := K) (spec.freshIdx k.isLt s)) := by
  unfold combinedFreshBasis; rw [map_sum]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [map_sum]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [map_smul]

/-- Gen: coefficient annihilation. -/
theorem combinedFreshBasis_injective_gen [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : spec.OneFaultPerLayer F)
    (c : ∀ (k : Fin spec.N), Fin (spec.L k) → K)
    (hker : spec.faultDiffGen Z Z_repl F (spec.combinedFreshBasis c) = 0) :
    ∀ (k : Fin spec.N), c k = 0 := by
  set Z' := spec.faultedTwiddlesGen Z Z_repl F with hZ'_def
  have h_sum : ∑ k ∈ Finset.range spec.N,
                  spec.teleTerm Z Z' k (spec.combinedFreshBasis c) = 0 := by
    have h := spec.diff_ntt_telescope_gen Z Z_repl F
    have h_apply := congrArg (fun L => L (spec.combinedFreshBasis c)) h
    simp only at h_apply
    -- h_apply: (ntt Z - nttFaultGen)(V) = (∑ teleTerm)(V)
    -- hker: faultDiffGen(V) = 0, which is definitionally (ntt Z - nttFaultGen)(V) = 0.
    have h_lhs : (spec.ntt Z - spec.nttFaultGen Z Z_repl F)
                    (spec.combinedFreshBasis c) = 0 := hker
    rw [h_lhs] at h_apply
    rw [LinearMap.coe_sum, Finset.sum_apply] at h_apply
    exact h_apply.symm
  have hT_each : ∀ ℓ : Fin spec.N,
      spec.teleTerm Z Z' ℓ.val (spec.combinedFreshBasis c) = 0 :=
    spec.each_zero_of_sum_zero_disjoint_lower
      (spec.teleTerm Z Z') spec.N
      (fun a => spec.teleTerm_image_disjoint_lower_gen hZ h2 hδ hF a)
      (spec.combinedFreshBasis c) h_sum
  suffices h_strong : ∀ m : ℕ, m ≤ spec.N →
      ∀ ℓ : Fin spec.N, ℓ.val < m → c ℓ = 0 by
    intro k; exact h_strong spec.N le_rfl k k.isLt
  intro m hm
  induction m with
  | zero => intro ℓ hℓ; exact absurd hℓ (Nat.not_lt_zero _)
  | succ m ih =>
    have hm' : m ≤ spec.N := Nat.le_of_succ_le hm
    intro ℓ hℓ
    by_cases hℓm : ℓ.val < m
    · exact ih hm' ℓ hℓm
    · have hℓ_eq : ℓ.val = m := by omega
      have hm_lt : m < spec.N := hm
      have hT_m_val := hT_each ⟨m, hm_lt⟩
      rw [spec.teleTerm_combinedFreshBasis_gen Z hF c hm_lt] at hT_m_val
      have h_collapse :
          (∑ k : Fin spec.N, ∑ s : Fin (spec.L k),
              if k.val ≤ m then
                c k s • spec.teleTerm Z Z' m
                  (spec.basis (K := K) (spec.freshIdx k.isLt s))
              else 0)
          = ∑ s : Fin (spec.L ⟨m, hm_lt⟩),
              c ⟨m, hm_lt⟩ s • spec.teleTerm Z Z' m
                (spec.basis (K := K) (spec.freshIdx hm_lt s)) := by
        rw [Finset.sum_eq_single (⟨m, hm_lt⟩ : Fin spec.N)]
        · simp
        · intros k _ hk_ne
          apply Finset.sum_eq_zero; intros s _
          by_cases hk_le : k.val ≤ m
          · have hk_lt : k.val < m := by
              rcases Nat.lt_or_eq_of_le hk_le with h | h
              · exact h
              · exfalso; apply hk_ne; exact Fin.ext h
            rw [if_pos hk_le, ih hm' k hk_lt]; simp
          · simp [hk_le]
        · intro h; exact absurd (Finset.mem_univ _) h
      rw [h_collapse] at hT_m_val
      have hLI := spec.teleTerm_basis_freshIdx_linearIndependent_gen hZ h2 hδ hF hm_lt
      have h_c_zero : ∀ s : Fin (spec.L ⟨m, hm_lt⟩), c ⟨m, hm_lt⟩ s = 0 :=
        fun s => (Fintype.linearIndependent_iff.mp hLI) (c ⟨m, hm_lt⟩) hT_m_val s
      have hℓ_def : ℓ = ⟨m, hm_lt⟩ := Fin.ext hℓ_eq
      subst hℓ_def
      funext s; exact h_c_zero s

/-- **Gen: rank lower bound.** `rank(faultDiffGen) ≥ ∑ L`. -/
theorem faultDiffGen_rank_ge_sum_L [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : spec.OneFaultPerLayer F) :
    (∑ k : Fin spec.N, spec.L ⟨k.val, k.isLt⟩)
      ≤ Module.finrank K ↑(LinearMap.range (spec.faultDiffGen Z Z_repl F)) := by
  let R := LinearMap.range (spec.faultDiffGen Z Z_repl F)
  let φ : (Σ k : Fin spec.N, Fin (spec.L k)) → (Fin spec.n → K) :=
    fun p => spec.faultDiffGen Z Z_repl F
      (spec.basis (K := K) (spec.freshIdx p.1.isLt p.2))
  have hφ_LI : LinearIndependent K φ := by
    rw [Fintype.linearIndependent_iff]
    intro g hsum
    let c : ∀ (k : Fin spec.N), Fin (spec.L k) → K := fun k s => g ⟨k, s⟩
    have h_DK : spec.faultDiffGen Z Z_repl F (spec.combinedFreshBasis c) = 0 := by
      rw [spec.faultDiffGen_combinedFreshBasis Z Z_repl F c]
      rw [show (Finset.univ : Finset (Σ k : Fin spec.N, Fin (spec.L k)))
              = (Finset.univ : Finset (Fin spec.N)).sigma
                  (fun k => (Finset.univ : Finset (Fin (spec.L k)))) from rfl,
          Finset.sum_sigma] at hsum
      exact hsum
    have h_c : ∀ k : Fin spec.N, c k = 0 :=
      spec.combinedFreshBasis_injective_gen hZ h2 hδ hF c h_DK
    intro p
    exact congrFun (h_c p.1) p.2
  have h_mem : ∀ p, φ p ∈ R := fun p => LinearMap.mem_range_self _ _
  let ψ : (Σ k : Fin spec.N, Fin (spec.L k)) → R := fun p => ⟨φ p, h_mem p⟩
  have hψ_LI : LinearIndependent K ψ :=
    LinearIndependent.of_comp R.subtype (by show LinearIndependent K (R.subtype ∘ ψ); exact hφ_LI)
  rw [← spec.card_sigma_fresh]
  exact hψ_LI.fintype_card_le_finrank

end LayerSpec
end NttFaultRank
