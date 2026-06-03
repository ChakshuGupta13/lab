/-
Phase F — Headline theorem closure.

Combines the upper bound (`RankCeiling.faultDiff_kyber_rank_le`), the
new lower bound from the Assembly pass
(`Assembly.faultDiff_rank_ge_sum_L`), the kernel inclusion
(`Theorem2Subset.theorem2_kyber_supset`), and rank–nullity to prove
the full kernel equality

  ker(D_K) = span {e_0, e_1}

This subsumes the stale `theorem2_kyber_statement` (which carried
the project's only `sorry`); the latter is kept in `Theorem2.lean`
for the headline documentation but is logically replaced by
`theorem2_kyber_final` below.
-/

import NttFaultRank.Assembly
import NttFaultRank.Theorem2Subset
import NttFaultRank.RankCeiling

namespace NttFaultRank

variable {K : Type*} [Field K]

/-- **Theorem 2 (full kernel equality).** For Kyber with all twiddles
    nonzero, `2 ≠ 0` in `K`, and one fault per layer, the kernel of
    the fault-difference operator is exactly the 2-dimensional span
    `span {e_0, e_1}`.

    Combines:
      * `theorem2_kyber_supset` (⊇ direction: `e_0, e_1 ∈ ker D_K`),
      * `faultDiff_kyber_rank_le` (rank ≤ 254, F9 upper bound),
      * `faultDiff_rank_ge_sum_L` specialised to Kyber (rank ≥ 254,
        from the new `Assembly` block: per-term rank + image
        disjointness + block-triangular LI via Step 1's off-layer
        vanishing),
      * rank–nullity for the kernel dimension,
      * basis distinctness for `finrank(span {e_0, e_1}) = 2`. -/
theorem theorem2_kyber_final
    {Z : kyberSpec.Twiddles K}
    (hZ : ∀ ℓ g, Z ℓ g ≠ 0)
    (h2 : (2 : K) ≠ 0)
    {F : kyberSpec.FaultSet}
    (hF : kyberSpec.OneFaultPerLayer F) :
    LinearMap.ker (kyberSpec.faultDiff Z F) =
      Submodule.span K
        ({kyberSpec.basis ⟨0, by decide⟩, kyberSpec.basis ⟨1, by decide⟩} :
          Set (Fin kyberSpec.n → K)) := by
  set S : Submodule K (Fin kyberSpec.n → K) :=
    Submodule.span K
      ({kyberSpec.basis ⟨0, by decide⟩, kyberSpec.basis ⟨1, by decide⟩} :
        Set (Fin kyberSpec.n → K)) with hS_def
  -- (1) S ⊆ ker(D_K) by Theorem2Subset.
  have h_le : S ≤ LinearMap.ker (kyberSpec.faultDiff Z F) :=
    theorem2_kyber_supset hF
  -- (2) finrank(S) = 2: e_0 ≠ e_1 are linearly independent basis vectors.
  have h_LI : LinearIndependent K
      (fun i : Fin 2 =>
        if i = 0 then kyberSpec.basis (K := K) ⟨0, by decide⟩
        else kyberSpec.basis (K := K) ⟨1, by decide⟩) := by
    rw [Fintype.linearIndependent_iff]
    intro c hsum i
    have h0 := congrFun hsum ⟨0, by decide⟩
    have h1 := congrFun hsum ⟨1, by decide⟩
    simp [LayerSpec.basis, Fin.sum_univ_two] at h0 h1
    fin_cases i
    · exact h0
    · exact h1
  have h_finrank_S : Module.finrank K S = 2 := by
    have hspan_eq : S =
        Submodule.span K (Set.range
          (fun i : Fin 2 =>
            if i = 0 then kyberSpec.basis (K := K) ⟨0, by decide⟩
            else kyberSpec.basis (K := K) ⟨1, by decide⟩)) := by
      rw [hS_def]
      congr 1
      ext v
      simp only [Set.mem_range, Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · rintro (rfl | rfl)
        · exact ⟨⟨0, by decide⟩, by simp⟩
        · exact ⟨⟨1, by decide⟩, by simp⟩
      · rintro ⟨i, rfl⟩
        fin_cases i <;> simp
    rw [hspan_eq, finrank_span_eq_card h_LI]
    simp
  -- (3) rank(D_K) = 254 (upper bound + lower bound).
  have h_rank_le : Module.finrank K
      ↑(LinearMap.range (kyberSpec.faultDiff Z F)) ≤ 254 :=
    faultDiff_kyber_rank_le hZ h2 hF
  have h_rank_ge : 254 ≤ Module.finrank K
      ↑(LinearMap.range (kyberSpec.faultDiff Z F)) := by
    have h := kyberSpec.faultDiff_rank_ge_sum_L (K := K) hZ h2 hF
    have h_val : (∑ k : Fin kyberSpec.N,
                    kyberSpec.L ⟨k.val, k.isLt⟩) = 254 := by decide
    omega
  have h_rank : Module.finrank K
      ↑(LinearMap.range (kyberSpec.faultDiff Z F)) = 254 :=
    le_antisymm h_rank_le h_rank_ge
  -- (4) finrank(ker(D_K)) = 2 by rank–nullity.
  have h_rn := LinearMap.finrank_range_add_finrank_ker
    (kyberSpec.faultDiff Z F)
  have h_dim_full : Module.finrank K (Fin kyberSpec.n → K) = 256 := by
    rw [Module.finrank_fin_fun]; decide
  rw [h_dim_full, h_rank] at h_rn
  have h_finrank_ker : Module.finrank K
      ↑(LinearMap.ker (kyberSpec.faultDiff Z F)) = 2 := by omega
  -- (5) Submodule equality from containment + equal finrank.
  symm
  apply Submodule.eq_of_le_of_finrank_eq h_le
  rw [h_finrank_S, h_finrank_ker]

end NttFaultRank
