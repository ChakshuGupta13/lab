/-
Headline GS (inverse-NTT) theorems: ker(D_GS) = span{NTT(e₀), NTT(e₁)}
for ML-KEM and ker(D_GS) = span{NTT(e₀)} for ML-DSA.

Combines:
  * `gs_kyber_supset_gen` / `gs_dsa_supset_gen` (⊇ direction),
  * `gsFaultDiffGen_rank_le_sum_L` (rank ≤ Σ L),
  * `gsFaultDiffGen_rank_ge_sum_L` (rank ≥ Σ L),
  * rank–nullity for kernel dimension,
  * basis independence for `finrank(span S) = dim(ker)`.
-/

import NttFaultRank.GsAssembly
import NttFaultRank.IntImageInV

namespace NttFaultRank

variable {K : Type*} [Field K]

/-- **Theorem 2 (GS, ML-KEM, generalized).** For the incomplete NTT (7 layers)
    with all twiddles nonzero, `2 ≠ 0` in `K`, one fault per layer, and
    replacement twiddles differing from originals at every faulted group,
    the kernel of the GS fault-difference operator is exactly the
    2-dimensional `span{NTT(e₀), NTT(e₁)}`.

    This is the inverse-NTT analogue of `theorem2_kyber_final_gen`. -/
theorem theorem2_kyber_gs_final_gen
    {Z : kyberSpec.Twiddles K}
    (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0)
    {Z_repl : kyberSpec.Twiddles K}
    {F : kyberSpec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : kyberSpec.OneFaultPerLayer F) :
    LinearMap.ker (kyberSpec.gsFaultDiffGen Z Z_repl F) =
      Submodule.span K
        ({kyberSpec.nttBasis Z ⟨0, by decide⟩,
          kyberSpec.nttBasis Z ⟨1, by decide⟩} :
          Set (Fin kyberSpec.n → K)) := by
  set S : Submodule K (Fin kyberSpec.n → K) :=
    Submodule.span K
      ({kyberSpec.nttBasis Z ⟨0, by decide⟩,
        kyberSpec.nttBasis Z ⟨1, by decide⟩} :
        Set (Fin kyberSpec.n → K)) with hS_def
  -- (1) S ⊆ ker(D_GS).
  have h_le : S ≤ LinearMap.ker (kyberSpec.gsFaultDiffGen Z Z_repl F) :=
    gs_kyber_supset_gen hZ h2 hF
  -- (2) finrank(S) = 2.
  -- NTT is injective (bijective), so NTT(e₀) ≠ NTT(e₁) and both nonzero.
  -- NTT(e₀) and NTT(e₁) are linearly independent because NTT is a linear
  -- isomorphism and e₀, e₁ are linearly independent.
  have h_finrank_S : Module.finrank K S = 2 := by
    -- NTT is injective, so images of LI basis vectors are LI.
    have h_ntt_inj : Function.Injective (kyberSpec.ntt Z) :=
      (kyberSpec.ntt_bijective hZ h2).1
    -- The 2-element family [nttBasis Z 0, nttBasis Z 1].
    have h_LI : LinearIndependent K
        (fun i : Fin 2 =>
          if i = 0 then kyberSpec.nttBasis (K := K) Z ⟨0, by decide⟩
          else kyberSpec.nttBasis (K := K) Z ⟨1, by decide⟩) := by
      rw [Fintype.linearIndependent_iff]
      intro c hsum i
      -- hsum : ∑ i, c i • (if i = 0 then ntt(e₀) else ntt(e₁)) = 0
      -- i.e.: c 0 • ntt(e₀) + c 1 • ntt(e₁) = 0
      -- i.e.: ntt(c 0 • e₀ + c 1 • e₁) = 0 (by linearity)
      -- By injectivity: c 0 • e₀ + c 1 • e₁ = 0
      -- By pointwise evaluation at 0 and 1: c 0 = 0, c 1 = 0.
      have hlin : kyberSpec.ntt Z (∑ j : Fin 2,
          c j • kyberSpec.basis (K := K)
            (if j = 0 then ⟨0, by decide⟩ else ⟨1, by decide⟩)) = 0 := by
        simp only [map_sum, map_smul, LayerSpec.nttBasis]
        exact hsum
      have hker := h_ntt_inj (show kyberSpec.ntt Z (∑ j : Fin 2,
          c j • kyberSpec.basis (K := K)
            (if j = 0 then ⟨0, by decide⟩ else ⟨1, by decide⟩)) =
          kyberSpec.ntt Z 0 from by rw [hlin, map_zero])
      have heval := congrFun hker
        (if i = 0 then (⟨0, by decide⟩ : Fin kyberSpec.n)
         else ⟨1, by decide⟩)
      simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul,
                  Pi.zero_apply] at heval
      simp only [Fin.sum_univ_two] at heval
      fin_cases i <;> simp_all [LayerSpec.basis]
    have hspan_eq : S =
        Submodule.span K (Set.range
          (fun i : Fin 2 =>
            if i = 0 then kyberSpec.nttBasis (K := K) Z ⟨0, by decide⟩
            else kyberSpec.nttBasis (K := K) Z ⟨1, by decide⟩)) := by
      rw [hS_def]; congr 1; ext v
      simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro (rfl | rfl)
        · exact ⟨⟨0, by decide⟩, by simp⟩
        · exact ⟨⟨1, by decide⟩, by simp⟩
      · rintro ⟨i, rfl⟩; fin_cases i <;> simp
    rw [hspan_eq, finrank_span_eq_card h_LI]; simp
  -- (3) rank(D_GS) = 254.
  have h_rank_le : Module.finrank K
      ↑(LinearMap.range (kyberSpec.gsFaultDiffGen Z Z_repl F)) ≤ 254 := by
    have h := kyberSpec.gsFaultDiffGen_rank_le_sum_L (K := K) hZ h2 hδ hF
    have h_val : (∑ k : Fin kyberSpec.N,
                    kyberSpec.L ⟨k.val, k.isLt⟩) = 254 := by decide
    omega
  have h_rank_ge : 254 ≤ Module.finrank K
      ↑(LinearMap.range (kyberSpec.gsFaultDiffGen Z Z_repl F)) := by
    have h := kyberSpec.gsFaultDiffGen_rank_ge_sum_L (K := K) hZ h2 hδ hF
    have h_val : (∑ k : Fin kyberSpec.N,
                    kyberSpec.L ⟨k.val, k.isLt⟩) = 254 := by decide
    omega
  have h_rank : Module.finrank K
      ↑(LinearMap.range (kyberSpec.gsFaultDiffGen Z Z_repl F)) = 254 :=
    le_antisymm h_rank_le h_rank_ge
  -- (4) finrank(ker) = 2 by rank–nullity (256 - 254 = 2).
  have h_rn := LinearMap.finrank_range_add_finrank_ker
    (kyberSpec.gsFaultDiffGen Z Z_repl F)
  have h_dim_full : Module.finrank K (Fin kyberSpec.n → K) = 256 := by
    rw [Module.finrank_fin_fun]; decide
  rw [h_dim_full, h_rank] at h_rn
  have h_finrank_ker : Module.finrank K
      ↑(LinearMap.ker (kyberSpec.gsFaultDiffGen Z Z_repl F)) = 2 := by omega
  -- (5) Equality from containment + equal finrank.
  symm
  apply Submodule.eq_of_le_of_finrank_eq h_le
  rw [h_finrank_S, h_finrank_ker]

/-- **Theorem 2 (GS, ML-DSA, generalized).** For the complete NTT (8 layers)
    with all twiddles nonzero, `2 ≠ 0` in `K`, one fault per layer, and
    replacement twiddles differing from originals at every faulted group,
    the kernel of the GS fault-difference operator is exactly the
    1-dimensional `span{NTT(e₀)}`. -/
theorem theorem2_dsa_gs_final_gen
    {Z : dsaSpec.Twiddles K}
    (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0)
    {Z_repl : dsaSpec.Twiddles K}
    {F : dsaSpec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : dsaSpec.OneFaultPerLayer F) :
    LinearMap.ker (dsaSpec.gsFaultDiffGen Z Z_repl F) =
      Submodule.span K
        ({dsaSpec.nttBasis Z ⟨0, by decide⟩} : Set (Fin dsaSpec.n → K)) := by
  set S : Submodule K (Fin dsaSpec.n → K) :=
    Submodule.span K
      ({dsaSpec.nttBasis Z ⟨0, by decide⟩} : Set (Fin dsaSpec.n → K)) with hS_def
  -- (1) S ⊆ ker(D_GS).
  have h_le : S ≤ LinearMap.ker (dsaSpec.gsFaultDiffGen Z Z_repl F) :=
    gs_dsa_supset_gen hZ h2 hF
  -- (2) finrank(S) = 1.
  have h_finrank_S : Module.finrank K S = 1 := by
    -- NTT(e₀) is nonzero (NTT is injective and e₀ ≠ 0), so the singleton
    -- span has finrank 1.
    have h_ntt_inj : Function.Injective (dsaSpec.ntt Z) :=
      (dsaSpec.ntt_bijective hZ h2).1
    have h_LI : LinearIndependent K
        (fun _ : Fin 1 => dsaSpec.nttBasis (K := K) Z ⟨0, by decide⟩) := by
      rw [Fintype.linearIndependent_iff]
      intro c hsum i
      -- hsum at index 0 gives: c 0 • nttBasis Z 0 = 0
      -- NTT is injective, so nttBasis Z 0 ≠ 0 (since basis 0 ≠ 0).
      -- Therefore c 0 = 0.
      have h0 := congrFun hsum ⟨0, by decide⟩
      simp only [Fin.sum_univ_one, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at h0
      -- h0 : c 0 * (nttBasis Z 0) 0 = 0
      -- nttBasis Z 0 = ntt Z (basis 0). (basis 0) is e₀.
      -- ntt Z e₀ at index 0 = ... we just need it nonzero.
      -- Use injectivity: if c 0 • ntt(e₀) = 0, then ntt(c 0 • e₀) = 0,
      -- so c 0 • e₀ = 0, so c 0 = 0 (since (e₀) 0 = 1 ≠ 0).
      have hlin : dsaSpec.ntt Z (c ⟨0, by decide⟩ •
          dsaSpec.basis (K := K) ⟨0, by decide⟩) = 0 := by
        rw [map_smul]
        ext j
        have := congrFun hsum j
        simp only [Fin.sum_univ_one, Pi.smul_apply, smul_eq_mul, Pi.zero_apply,
                    LayerSpec.nttBasis] at this ⊢
        exact this
      have hker : c ⟨0, by decide⟩ • dsaSpec.basis (K := K) ⟨0, by decide⟩ = 0 :=
        h_ntt_inj (by rw [hlin, map_zero])
      have h0' := congrFun hker ⟨0, by decide⟩
      simp [LayerSpec.basis] at h0'
      fin_cases i; exact h0'
    have hspan_eq : S =
        Submodule.span K (Set.range
          (fun _ : Fin 1 => dsaSpec.nttBasis (K := K) Z ⟨0, by decide⟩)) := by
      rw [hS_def]; congr 1; ext v
      simp only [Set.mem_range, Set.mem_singleton_iff]
      constructor
      · rintro rfl; exact ⟨⟨0, by decide⟩, rfl⟩
      · rintro ⟨_, rfl⟩; rfl
    rw [hspan_eq, finrank_span_eq_card h_LI]
    simp [Fintype.card_fin]
  -- (3) rank(D_GS) = 255.
  have h_rank_le : Module.finrank K
      ↑(LinearMap.range (dsaSpec.gsFaultDiffGen Z Z_repl F)) ≤ 255 := by
    have h := dsaSpec.gsFaultDiffGen_rank_le_sum_L (K := K) hZ h2 hδ hF
    have h_val : (∑ k : Fin dsaSpec.N,
                    dsaSpec.L ⟨k.val, k.isLt⟩) = 255 := by decide
    omega
  have h_rank_ge : 255 ≤ Module.finrank K
      ↑(LinearMap.range (dsaSpec.gsFaultDiffGen Z Z_repl F)) := by
    have h := dsaSpec.gsFaultDiffGen_rank_ge_sum_L (K := K) hZ h2 hδ hF
    have h_val : (∑ k : Fin dsaSpec.N,
                    dsaSpec.L ⟨k.val, k.isLt⟩) = 255 := by decide
    omega
  have h_rank : Module.finrank K
      ↑(LinearMap.range (dsaSpec.gsFaultDiffGen Z Z_repl F)) = 255 :=
    le_antisymm h_rank_le h_rank_ge
  -- (4) finrank(ker) = 1 by rank–nullity (256 - 255 = 1).
  have h_rn := LinearMap.finrank_range_add_finrank_ker
    (dsaSpec.gsFaultDiffGen Z Z_repl F)
  have h_dim_full : Module.finrank K (Fin dsaSpec.n → K) = 256 := by
    rw [Module.finrank_fin_fun]; decide
  rw [h_dim_full, h_rank] at h_rn
  have h_finrank_ker : Module.finrank K
      ↑(LinearMap.ker (dsaSpec.gsFaultDiffGen Z Z_repl F)) = 1 := by omega
  -- (5) Equality from containment + equal finrank.
  symm
  apply Submodule.eq_of_le_of_finrank_eq h_le
  rw [h_finrank_S, h_finrank_ker]

end NttFaultRank
