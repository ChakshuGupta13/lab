import TxGraffitiC1FC.Majorization
import TxGraffitiC1FC.Graphical

namespace SimpleGraph

open TxGraffitiC1FC

theorem majorizesSorted_graph {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (v : V) (hv : ∀ w, G.degree w ≤ G.degree v) :
    MajorizesSorted
      (((G.induce ({v}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·))
      (havelHakimiStep ((G.degreeMultiset).sort (· ≥ ·))) := by
  classical
  set l : List (ℕ × ℕ) :=
    (Finset.univ.erase v).toList.map (fun w => (G.degree w, if G.Adj w v then 1 else 0)) with hl
  have hbin_l : ∀ p ∈ l, p.2 ≤ 1 := by
    intro p hp; rw [hl, List.mem_map] at hp
    obtain ⟨w, _, rfl⟩ := hp; dsimp only; split_ifs <;> simp
  have hpair_l : ∀ p ∈ l, p.2 ≤ p.1 := by
    intro p hp; rw [hl, List.mem_map] at hp
    obtain ⟨w, _, rfl⟩ := hp; dsimp only
    split_ifs with hadj
    · exact (G.degree_pos_iff_exists_adj w).mpr ⟨v, hadj⟩
    · simp
  obtain ⟨rest, e, hsort, hbin, hlen, hesum, hrestperm, hπeq, hForall⟩ :=
    pairsort_align l hbin_l hpair_l
  -- Δ = e.sum = deg v
  have hΔ : e.sum = G.degree v := by
    rw [hesum, hl, List.map_map]; exact sum_adj_indicator_eq_degree G v
  -- σ.Perm (zipWith (·-·) rest e)
  have hσperm : (((G.induce ({v}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·)).Perm
      (List.zipWith (· - ·) rest e) := by
    rw [degreeMultiset_induce_compl]
    refine (sort_map_perm_toList_map (Finset.univ.erase v) _).trans ?_
    rw [hl, List.map_map] at hπeq
    exact hπeq
  -- rest = sort(non-v degrees)
  have hrest_eq : rest = ((Finset.univ.erase v).val.map (fun w => G.degree w)).sort (· ≥ ·) := by
    refine List.Perm.eq_of_sortedGE hsort.sortedGE (Multiset.pairwise_sort _ _).sortedGE ?_
    refine hrestperm.trans ?_
    rw [hl, List.map_map]
    exact (sort_map_perm_toList_map (Finset.univ.erase v) (fun w => G.degree w)).symm
  -- π'.Perm ((rest.take Δ).map (·-1) ++ rest.drop Δ)
  have hπperm : (havelHakimiStep ((G.degreeMultiset).sort (· ≥ ·))).Perm
      (((rest.take e.sum).map (· - 1)) ++ rest.drop e.sum) := by
    have hsG : (G.degreeMultiset).sort (· ≥ ·) = G.degree v :: rest := by
      rw [hrest_eq]; exact sortedDeg_cons G v hv
    rw [hsG, hΔ]
    unfold havelHakimiStep
    simp only [List.splitAt_eq]
    exact List.mergeSort_perm _ _
  -- σ.sum = π'.sum
  have hk : e.sum ≤ rest.length := by rw [← hlen]; exact sum_le_length_of_le_one e hbin
  have hpos : ∀ x ∈ rest.take e.sum, 1 ≤ x := take_sum_pos rest e hsort hForall hbin
  have hsum : (((G.induce ({v}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·)).sum
      = (havelHakimiStep ((G.degreeMultiset).sort (· ≥ ·))).sum := by
    rw [hσperm.sum_eq, hπperm.sum_eq, sum_zipWith_sub_of_forall₂ rest e hForall,
      sum_dec_top_eq rest e.sum hk hpos]
  exact majorizesSorted_of_dec rest e _ _ hbin hlen hsort
    (Multiset.pairwise_sort _ _) hσperm hπperm hsum

end SimpleGraph
