/-
Phase F — Headline theorem closure for ML-DSA (complete NTT).

Mirrors `Theorem2Final.lean` (ML-KEM, incomplete NTT). The only
non-mechanical changes are:

  * `kyberSpec` → `dsaSpec`,
  * 2-vector basis `{e₀, e₁}` → 1-vector basis `{e₀}` (per
    `theorem2_dsa_supset_gen`'s 1-dim kernel claim),
  * `finrank(span S) = 2` → `= 1` (singleton, trivial LI),
  * `∑ L = 254` → `= 255` (one extra layer of length 1; `dsaL ⟨7,_⟩ = 1`),
  * `finrank(ker) = 2` → `= 1` (rank–nullity: 256 − 255 = 1).

Combines:
  * `theorem2_dsa_supset_gen` (⊇: `e₀ ∈ ker D_K^gen`),
  * `faultDiffGen_rank_le_sum_L` (spec-generic, used at `dsaSpec`),
  * `faultDiffGen_rank_ge_sum_L` (spec-generic, used at `dsaSpec`),
  * rank–nullity,
  * `Set.range_const` for the singleton-range equality.
-/

import NttFaultRank.Assembly
import NttFaultRank.Theorem2Subset
import NttFaultRank.RankCeiling

namespace NttFaultRank

variable {K : Type*} [Field K]

/-- **Theorem 2 (ML-DSA, generalized).** For the complete NTT (8 layers,
    `dsaL ⟨7,_⟩ = 1`) with all twiddles nonzero, `2 ≠ 0` in `K`, one
    fault per layer, and replacement twiddles differing from originals
    at every faulted group, the kernel of the generalized
    fault-difference operator is exactly the 1-dimensional `span{e₀}`.

    The 1-dimensional kernel (vs Kyber's 2-dimensional) reflects that
    the complete NTT's innermost layer ejects `e₁`: position `1` is a
    `b`-input at the layer-`7` butterfly `(0,1)`, so a fault at the
    innermost twiddle perturbs the trajectory of `e₁`. -/
theorem theorem2_dsa_final_gen
    {Z : dsaSpec.Twiddles K}
    (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0)
    {Z_repl : dsaSpec.Twiddles K}
    {F : dsaSpec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g)
    (hF : dsaSpec.OneFaultPerLayer F) :
    LinearMap.ker (dsaSpec.faultDiffGen Z Z_repl F) =
      Submodule.span K
        ({dsaSpec.basis ⟨0, by decide⟩} : Set (Fin dsaSpec.n → K)) := by
  set S : Submodule K (Fin dsaSpec.n → K) :=
    Submodule.span K
      ({dsaSpec.basis ⟨0, by decide⟩} : Set (Fin dsaSpec.n → K)) with hS_def
  -- (1) S ⊆ ker(D_K^gen).
  have h_le : S ≤ LinearMap.ker (dsaSpec.faultDiffGen Z Z_repl F) :=
    theorem2_dsa_supset_gen hF
  -- (2) finrank(S) = 1: e₀ is a single nonzero vector.
  have h_LI : LinearIndependent K
      (fun _ : Fin 1 => dsaSpec.basis (K := K) ⟨0, by decide⟩) := by
    rw [Fintype.linearIndependent_iff]
    intro c hsum i
    have h0 := congrFun hsum ⟨0, by decide⟩
    simp [LayerSpec.basis] at h0
    fin_cases i
    exact h0
  have h_finrank_S : Module.finrank K S = 1 := by
    have hspan_eq : S =
        Submodule.span K (Set.range
          (fun _ : Fin 1 => dsaSpec.basis (K := K) ⟨0, by decide⟩)) := by
      rw [hS_def]; congr 1; ext v
      simp only [Set.mem_range, Set.mem_singleton_iff]
      constructor
      · rintro rfl; exact ⟨⟨0, by decide⟩, rfl⟩
      · rintro ⟨_, rfl⟩; rfl
    rw [hspan_eq, finrank_span_eq_card h_LI]; simp
  -- (3) rank(D_K^gen) = 255 (upper + lower).
  have h_rank_le : Module.finrank K
      ↑(LinearMap.range (dsaSpec.faultDiffGen Z Z_repl F)) ≤ 255 := by
    have h := dsaSpec.faultDiffGen_rank_le_sum_L (K := K) hZ h2 hδ hF
    have h_val : (∑ k : Fin dsaSpec.N,
                    dsaSpec.L ⟨k.val, k.isLt⟩) = 255 := by decide
    omega
  have h_rank_ge : 255 ≤ Module.finrank K
      ↑(LinearMap.range (dsaSpec.faultDiffGen Z Z_repl F)) := by
    have h := dsaSpec.faultDiffGen_rank_ge_sum_L (K := K) hZ h2 hδ hF
    have h_val : (∑ k : Fin dsaSpec.N,
                    dsaSpec.L ⟨k.val, k.isLt⟩) = 255 := by decide
    omega
  have h_rank : Module.finrank K
      ↑(LinearMap.range (dsaSpec.faultDiffGen Z Z_repl F)) = 255 :=
    le_antisymm h_rank_le h_rank_ge
  -- (4) finrank(ker) = 1 by rank–nullity (256 − 255 = 1).
  have h_rn := LinearMap.finrank_range_add_finrank_ker
    (dsaSpec.faultDiffGen Z Z_repl F)
  have h_dim_full : Module.finrank K (Fin dsaSpec.n → K) = 256 := by
    rw [Module.finrank_fin_fun]; decide
  rw [h_dim_full, h_rank] at h_rn
  have h_finrank_ker : Module.finrank K
      ↑(LinearMap.ker (dsaSpec.faultDiffGen Z Z_repl F)) = 1 := by omega
  -- (5) Equality from containment + equal finrank.
  symm
  apply Submodule.eq_of_le_of_finrank_eq h_le
  rw [h_finrank_S, h_finrank_ker]

/-- **Theorem 2 (ML-DSA, zeroing-fault corollary).** For the complete NTT
    with all twiddles nonzero, `2 ≠ 0` in `K`, and one fault per layer,
    the kernel of the fault-difference operator is exactly
    `span{e₀}` (1-dimensional). -/
theorem theorem2_dsa_final
    {Z : dsaSpec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) (h2 : (2 : K) ≠ 0)
    {F : dsaSpec.FaultSet} (hF : dsaSpec.OneFaultPerLayer F) :
    LinearMap.ker (dsaSpec.faultDiff Z F) =
      Submodule.span K
        ({dsaSpec.basis ⟨0, by decide⟩} : Set (Fin dsaSpec.n → K)) := by
  have h := theorem2_dsa_final_gen hZ h2
    (Z_repl := fun _ _ => 0)
    (hδ := fun ℓ g _ => by simp [hZ ℓ g]) hF
  rwa [dsaSpec.faultDiffGen_zero] at h

end NttFaultRank
