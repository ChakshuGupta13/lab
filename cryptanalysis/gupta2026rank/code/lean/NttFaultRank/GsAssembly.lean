/-
Gentleman-Sande assembly: kernel inclusion via cascade identity,
rank upper/lower bounds.

Key identity: after j GS layers applied to ntt Z (basis p),
the state is 2^j · runPrefix Z (N-j) (basis p). This means
the GS perturbation at each layer sees b=a (from the CT support
invariant), making every telescope term vanish on NTT(e_p).
-/

import NttFaultRank.GsNtt
import NttFaultRank.Theorem2Subset
import NttFaultRank.RankCeiling
import NttFaultRank.IntImageInV
import NttFaultRank.Assembly
import NttFaultRank.GsCTDuality
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

-- NTT of basis vector.
def nttBasis (Z : spec.Twiddles K) (p : Fin spec.n) : Fin spec.n → K :=
  spec.ntt Z (spec.basis p)

/-! ### Cascade helper: GS ∘ CT = 2 on b-side-zero inputs -/

-- Per-group.
lemma gsGroupLayer_comp_groupLayer_bZero (L : ℕ) (z : K)
    (v : Fin 2 → Fin L → K) (hb : ∀ j, v 1 j = 0) :
    gsGroupLayer L z (groupLayer L z v) = (2 : K) • v := by
  funext s j; rcases fin2_cases s with rfl | rfl
  · simp [hb j, sub_zero, mul_zero, add_zero]; ring
  · simp [hb j, sub_zero, mul_zero, sub_self]

-- Full layer.
lemma gsFullLayer_comp_fullLayer_bZero (G L : ℕ) (zs : Fin G → K)
    (v : Fin G → Fin 2 → Fin L → K) (hb : ∀ g j, v g 1 j = 0) :
    gsFullLayer G L zs (fullLayer G L zs v) = (2 : K) • v := by
  funext g; show gsGroupLayer L (zs g) (groupLayer L (zs g) (v g)) = _
  rw [gsGroupLayer_comp_groupLayer_bZero L (zs g) (v g) (hb g)]
  simp [Pi.smul_apply]

-- Flat-form: gsLayer Z ℓ (layer Z ℓ v) = 2 • v when bSide v ℓ g j = 0 for all g j.
lemma gsLayer_comp_layer_bZero [CooleyTukeyLike spec]
    (Z : spec.Twiddles K) (ℓ : Fin spec.N) (v : Fin spec.n → K)
    (hb : ∀ g : Fin (spec.G ℓ), ∀ j : Fin (spec.L ℓ),
      spec.bSide v ℓ g j = 0) :
    spec.gsLayer Z ℓ (spec.layer Z ℓ v) = (2 : K) • v := by
  -- Now gsLayerOn uses funReindex directly (not a let-bound equiv),
  -- so the composition gsLayer ∘ layer simplifies through funReindex cancellation.
  show (gsLayerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z ℓ))
      ((layerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z ℓ)) v) = _
  unfold gsLayerOn layerOn
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe,
             LinearEquiv.symm_apply_apply]
  -- Goal: funReindex.symm(gsFullLayer'(Z ℓ, fullLayer'(Z ℓ, funReindex v))) = 2 • v
  unfold gsFullLayer' fullLayer'
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe,
             LinearEquiv.symm_apply_apply, LinearEquiv.apply_symm_apply]
  -- Goal: funReindex.symm(reshape.symm(gsFullLayer(Z ℓ, fullLayer(Z ℓ, reshape(funReindex v))))) = 2 • v
  rw [gsFullLayer_comp_fullLayer_bZero _ _ (Z ℓ) _ (fun g j => hb g j)]
  -- Goal: funReindex.symm(reshape.symm(2 • reshape(funReindex v))) = 2 • v
  simp [LinearEquiv.map_smul]

/-! ### Cascade identity -/

-- After CT layer k, bSide = aSide (CT butterfly (a,0) → (a,a)).
lemma runPrefix_succ_bSide_eq_aSide [CooleyTukeyLike spec]
    (Z : spec.Twiddles K) (p : Fin spec.n)
    (k : ℕ) (hk : k < spec.N) (hp : p.val < spec.L ⟨k, hk⟩)
    (g : Fin (spec.G ⟨k, hk⟩)) (j : Fin (spec.L ⟨k, hk⟩)) :
    spec.bSide
      (spec.runPrefix Z (k + 1) (by omega) (spec.basis p))
      ⟨k, hk⟩ g j =
    reshape K (spec.G ⟨k, hk⟩) (spec.L ⟨k, hk⟩)
      ((funReindex spec.n (spec.G ⟨k, hk⟩) (spec.L ⟨k, hk⟩) (spec.hGL ⟨k, hk⟩))
        (spec.runPrefix Z (k + 1) (by omega) (spec.basis p))) g 0 j := by
  rw [spec.runPrefix_succ Z k (by omega)]
  simp only [LinearMap.comp_apply]
  set v := spec.runPrefix Z k (by omega : k ≤ spec.N) (spec.basis (K := K) p)
  have hbz : ∀ j', spec.bSide v ⟨k, hk⟩ g j' = 0 :=
    fun j' => spec.runPrefix_basis_bSide_zero Z p ⟨k, hk⟩ hp g j'
  have hb0 : spec.bSide v ⟨k, hk⟩ g j = 0 := hbz j
  rw [show spec.bSide (spec.layer Z ⟨k, hk⟩ v) ⟨k, hk⟩ g j =
        spec.aSide v ⟨k, hk⟩ g j - Z ⟨k, hk⟩ g * spec.bSide v ⟨k, hk⟩ g j
      from spec.bSide_layer Z ⟨k, hk⟩ v g j]
  rw [hb0, mul_zero, sub_zero]
  change _ = spec.aSide (spec.layer Z ⟨k, hk⟩ v) ⟨k, hk⟩ g j
  rw [spec.aSide_layer Z ⟨k, hk⟩ v g j, hb0, mul_zero, add_zero]

/-! ### GS-CT roundtrip identity -/

-- The cascade: gsNtt Z (ntt Z (basis p)) = 2^N • basis p.
-- At each layer: CT has zero b-side, CT butterfly (a,0)→(a,a),
-- GS butterfly (a,a)→(2a,0). Twiddle z cancels: z*0 = 0 in both.
-- This is twiddle-independent.
theorem gsNtt_ntt_basis_eq [CooleyTukeyLike spec]
    (Z : spec.Twiddles K)
    (p : Fin spec.n)
    (hp : ∀ k : Fin spec.N, p.val < spec.L k) :
    spec.gsNtt Z (spec.ntt Z (spec.basis (K := K) p)) =
      (2 : K) ^ spec.N • spec.basis (K := K) p := by
  -- gsNtt Z = gsPartial Z N. ntt Z = runPrefix Z N.
  rw [spec.gsNtt_eq_gsPartial Z]
  change spec.gsPartial Z spec.N le_rfl
    (spec.runPrefix Z spec.N le_rfl (spec.basis (K := K) p)) = _
  -- Prove by induction: gsPartial Z m (runPrefix Z m (basis p)) = 2^m • basis p.
  suffices h : ∀ m : ℕ, (hm : m ≤ spec.N) →
      spec.gsPartial Z m hm (spec.runPrefix Z m hm (spec.basis (K := K) p)) =
        (2 : K) ^ m • spec.basis (K := K) p from h spec.N le_rfl
  intro m hm
  induction m with
  | zero =>
    simp [spec.gsPartial_zero, LayerSpec.runPrefix]
  | succ m ih =>
    rw [spec.gsPartial_succ Z m hm]
    simp only [LinearMap.comp_apply]
    rw [spec.runPrefix_succ Z m hm]
    simp only [LinearMap.comp_apply]
    -- Goal: gsPartial Z m _ (gsLayer Z ⟨m,hm⟩ (layer Z ⟨m,hm⟩ (runPrefix Z m _ (basis p)))) = ...
    rw [spec.gsLayer_comp_layer_bZero Z ⟨m, hm⟩
          (spec.runPrefix Z m (Nat.le_of_succ_le hm) (spec.basis (K := K) p))
          (fun g j => spec.runPrefix_basis_bSide_zero Z p ⟨m, hm⟩ (hp ⟨m, hm⟩) g j)]
    -- Goal: gsPartial Z m _ (2 • runPrefix Z m _ (basis p)) = 2^(m+1) • basis p
    rw [map_smul]
    rw [ih (Nat.le_of_succ_le hm)]
    rw [smul_comm, pow_succ, mul_smul]

/-! ### Kernel inclusion -/

-- nttBasis Z p ∈ ker(gsFaultDiffGen) by cascade + CT kernel.
theorem nttBasis_mem_gsKer [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K}
    {F : spec.FaultSet}
    (hF : spec.OneFaultPerLayer F)
    {p : Fin spec.n}
    (hp : ∀ k : Fin spec.N, p.val < spec.L k) :
    spec.nttBasis Z p ∈
      LinearMap.ker (spec.gsFaultDiffGen Z Z_repl F) := by
  rw [LinearMap.mem_ker]
  show spec.gsNtt Z (spec.ntt Z (spec.basis p)) -
       spec.gsNttFaultGen Z Z_repl F (spec.ntt Z (spec.basis p)) = 0
  rw [sub_eq_zero]
  -- LHS = 2^N • basis p by cascade
  rw [spec.gsNtt_ntt_basis_eq Z p hp]
  -- RHS: gsNtt Z' (ntt Z (basis p)) = gsNtt Z' (ntt Z' (basis p)) [CT kernel]
  --     = 2^N • basis p [cascade with Z']
  show (2 : K) ^ spec.N • spec.basis (K := K) p =
       spec.gsNtt (spec.faultedTwiddlesGen Z Z_repl F)
         (spec.ntt Z (spec.basis (K := K) p))
  have h_ct := spec.basis_mem_ker_gen (Z := Z) (Z_repl := Z_repl) hF p hp
  rw [LinearMap.mem_ker] at h_ct
  have h_ntt_eq : spec.ntt Z (spec.basis (K := K) p) =
      spec.ntt (spec.faultedTwiddlesGen Z Z_repl F) (spec.basis (K := K) p) := by
    have : (spec.ntt Z - spec.nttFaultGen Z Z_repl F) (spec.basis p) = 0 := h_ct
    rw [LinearMap.sub_apply, sub_eq_zero] at this; exact this
  rw [h_ntt_eq]
  rw [spec.gsNtt_ntt_basis_eq (spec.faultedTwiddlesGen Z Z_repl F) p hp]

/-! ### Rank bounds -/

-- Design note (historical): gsNtt ∘ ntt = 2^N • id holds only for
-- FIPS-specific twiddles (exact INTT), NOT for abstract twiddles.
-- The rank bounds use GS-CT butterfly duality (GsCTDuality.lean) instead:
-- gsLayer is the adjoint of layer with negated twiddles, so
-- rank(D_GS) = rank(D_CT(-Z)) by matrix transpose rank equality.

-- GS rank upper bound via rank-nullity + kernel inclusion.
theorem gsFaultDiffGen_rank_le_sum_L [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K}
    {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : spec.OneFaultPerLayer F) :
    Module.finrank K
      (LinearMap.range (spec.gsFaultDiffGen Z Z_repl F))
      ≤ ∑ k : Fin spec.N, spec.L ⟨k.val, k.isLt⟩ := by
  -- Rank-nullity + kernel inclusion.
  -- ker(D_GS) has ≥ L_{N-1} LI vectors (nttBasis Z p for p < L_{N-1}).
  -- rank ≤ n - L_{N-1} = Σ L_k.
  by_cases hN_pos : 0 < spec.N
  · -- Σ L_k = n - L_{N-1} by geometric series.
    rw [spec.sum_L_eq_n_sub_L_last hN_pos]
    -- Construct L_{N-1} LI vectors in ker.
    have hNlast_lt : spec.N - 1 < spec.N := Nat.sub_lt hN_pos Nat.one_pos
    have hLlast_le_n : spec.L ⟨spec.N - 1, hNlast_lt⟩ ≤ spec.n := by
      have := spec.two_L_le_n_F8 (spec.N - 1) hNlast_lt; omega
    have h_ntt_inj := (spec.ntt_bijective hZ h2).1
    -- Family: p ↦ (nttBasis Z p, proof it's in ker).
    have h_in_ker : ∀ p : Fin (spec.L ⟨spec.N - 1, hNlast_lt⟩),
        spec.nttBasis Z ⟨p.val, lt_of_lt_of_le p.isLt hLlast_le_n⟩ ∈
          LinearMap.ker (spec.gsFaultDiffGen Z Z_repl F) := fun p =>
      spec.nttBasis_mem_gsKer hZ h2 hF (fun k =>
        lt_of_lt_of_le p.isLt (spec.L_antitone hNlast_lt k.isLt (by omega)))
    -- LI proof: NTT is injective, basis vectors are LI.
    have h_LI : LinearIndependent K (fun p : Fin (spec.L ⟨spec.N - 1, hNlast_lt⟩) =>
        (⟨spec.nttBasis Z ⟨p.val, lt_of_lt_of_le p.isLt hLlast_le_n⟩,
          h_in_ker p⟩ :
          LinearMap.ker (spec.gsFaultDiffGen Z Z_repl F))) := by
      rw [Fintype.linearIndependent_iff]
      intro c hsum p
      have hsum_vec : ∑ q, c q • spec.nttBasis (K := K) Z
          ⟨q.val, lt_of_lt_of_le q.isLt hLlast_le_n⟩ = 0 := by
        have := congrArg Subtype.val hsum
        simp only [Submodule.coe_sum, Submodule.coe_smul, ZeroMemClass.coe_zero] at this
        exact this
      -- ntt Z (Σ c_q • basis q) = Σ c_q • nttBasis Z q = 0
      have hker : ∑ q, c q • spec.basis (K := K)
          ⟨q.val, lt_of_lt_of_le q.isLt hLlast_le_n⟩ = 0 :=
        h_ntt_inj (show spec.ntt Z _ = spec.ntt Z 0 by
          rw [map_sum, map_zero]; simp_rw [map_smul]; exact hsum_vec)
      -- Evaluate at position p: basis q evaluated at p is δ_{qp}.
      have heval := congrFun hker ⟨p.val, lt_of_lt_of_le p.isLt hLlast_le_n⟩
      simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply,
                  LayerSpec.basis] at heval
      -- heval: Σ c_q * (if ⟨p,_⟩ = ⟨q,_⟩ then 1 else 0) = 0.
      -- Since the basis vectors are distinct, only q = p contributes.
      convert heval using 1
      rw [Finset.sum_eq_single p
        (fun q _ hqp => by
          have : (⟨p.val, lt_of_lt_of_le p.isLt hLlast_le_n⟩ : Fin spec.n) ≠
                 ⟨q.val, lt_of_lt_of_le q.isLt hLlast_le_n⟩ := by
            intro h; exact hqp (Fin.ext (Fin.mk.inj h.symm))
          simp [this])
        (fun h => absurd (Finset.mem_univ _) h)]
      simp
    -- dim(ker) ≥ L_{N-1}.
    have h_ker_dim := h_LI.fintype_card_le_finrank
    simp only [Fintype.card_fin] at h_ker_dim
    -- rank-nullity.
    have h_rn := LinearMap.finrank_range_add_finrank_ker (spec.gsFaultDiffGen Z Z_repl F)
    rw [Module.finrank_fin_fun] at h_rn
    omega
  · -- N = 0: rank ≤ 0 since gsFaultDiffGen = 0 (gsNtt = id for 0 layers).
    push_neg at hN_pos
    have hN0 : spec.N = 0 := Nat.le_zero.mp hN_pos
    have : spec.gsFaultDiffGen Z Z_repl F = 0 := by
      unfold gsFaultDiffGen gsNttFaultGen gsNtt
      simp [hN0, Fin.foldl_zero]
    rw [this, LinearMap.range_zero, finrank_bot]
    exact Nat.zero_le _

-- GS rank lower bound via GS-CT duality (GsCTDuality.lean):
-- rank(D_GS(Z, Z_repl, F)) = rank(D_CT(-Z, -Z_repl, F)) ≥ Σ L_k.
theorem gsFaultDiffGen_rank_ge_sum_L [CooleyTukeyLike spec]
    {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {Z_repl : spec.Twiddles K}
    {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : spec.OneFaultPerLayer F) :
    ∑ k : Fin spec.N, spec.L ⟨k.val, k.isLt⟩
      ≤ Module.finrank K
          (LinearMap.range (spec.gsFaultDiffGen Z Z_repl F)) := by
  -- By GS-CT duality (GsCTDuality.lean): rank(D_GS) = rank(D_CT(-Z, -Z_repl, F)).
  rw [spec.gsFaultDiffGen_rank_eq Z Z_repl F]
  -- Apply the CT rank lower bound with negated twiddles.
  exact spec.faultDiffGen_rank_ge_sum_L
    (spec.negTwiddles_ne_zero hZ) h2
    (spec.negTwiddles_delta hδ) hF

end LayerSpec

/-! ### Instantiation -/

theorem gs_kyber_supset_gen
    {Z : kyberSpec.Twiddles K}
    (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0)
    {Z_repl : kyberSpec.Twiddles K}
    {F : kyberSpec.FaultSet}
    (hF : kyberSpec.OneFaultPerLayer F) :
    Submodule.span K
      ({kyberSpec.nttBasis Z ⟨0, by decide⟩,
        kyberSpec.nttBasis Z ⟨1, by decide⟩} :
        Set (Fin kyberSpec.n → K))
      ≤ LinearMap.ker (kyberSpec.gsFaultDiffGen Z Z_repl F) := by
  rw [Submodule.span_le]
  intro v hv
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hv
  rcases hv with rfl | rfl
  · exact kyberSpec.nttBasis_mem_gsKer hZ h2 hF
      (fun k => by fin_cases k <;> decide)
  · exact kyberSpec.nttBasis_mem_gsKer hZ h2 hF
      (fun k => by fin_cases k <;> decide)

theorem gs_dsa_supset_gen
    {Z : dsaSpec.Twiddles K}
    (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0)
    {Z_repl : dsaSpec.Twiddles K}
    {F : dsaSpec.FaultSet}
    (hF : dsaSpec.OneFaultPerLayer F) :
    Submodule.span K
      ({dsaSpec.nttBasis Z ⟨0, by decide⟩} :
        Set (Fin dsaSpec.n → K))
      ≤ LinearMap.ker (dsaSpec.gsFaultDiffGen Z Z_repl F) := by
  rw [Submodule.span_le]
  intro v hv
  simp only [Set.mem_singleton_iff] at hv
  rw [hv]
  exact dsaSpec.nttBasis_mem_gsKer hZ h2 hF
    (fun k => by fin_cases k <;> decide)

end NttFaultRank
