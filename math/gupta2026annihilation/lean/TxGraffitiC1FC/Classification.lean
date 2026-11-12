import TxGraffitiC1FC.Equality
import TxGraffitiC1FC.DegreeTwoStructure
import TxGraffitiC1FC.DegreeTwoResidue
import TxGraffitiC1FC.EqualityWitnesses

namespace SimpleGraph

open scoped Classical

namespace Iso

variable {V W : Type*} [Fintype V] [DecidableEq V] [Fintype W] [DecidableEq W]
  {G : SimpleGraph V} [DecidableRel G.Adj] {H : SimpleGraph W} [DecidableRel H.Adj]

theorem indepNum_le (e : G ≃g H) : G.indepNum ≤ H.indepNum := by
  obtain ⟨I, hind, hcard⟩ := G.exists_isNIndepSet_indepNum
  rw [← hcard, ← Finset.card_map e.toEquiv.toEmbedding (s := I)]
  apply SimpleGraph.IsIndepSet.card_le_indepNum
  intro x hx y hy hxy hadj
  obtain ⟨a, ha, rfl⟩ := Finset.mem_map.mp hx
  obtain ⟨b, hb, rfl⟩ := Finset.mem_map.mp hy
  exact hind ha hb (fun hab => hxy (congrArg e hab)) (e.map_rel_iff.mp hadj)

theorem indepNum_eq (e : G ≃g H) : G.indepNum = H.indepNum :=
  le_antisymm e.indepNum_le e.symm.indepNum_le

theorem degreeMultiset_eq (e : G ≃g H) : G.degreeMultiset = H.degreeMultiset := by
  have huniv : (Finset.univ : Finset W).val =
      (Finset.univ : Finset V).val.map e.toEquiv.toEmbedding := by
    have h : (Finset.univ : Finset W) =
        (Finset.univ : Finset V).map e.toEquiv.toEmbedding := by
      ext w
      simp
    rw [h, Finset.map_val]
  symm
  unfold SimpleGraph.degreeMultiset
  rw [huniv, Multiset.map_map]
  exact Multiset.map_congr rfl (fun x _ => e.degree_eq x)

theorem annihilationNumber_eq (e : G ≃g H) :
    G.annihilationNumber = H.annihilationNumber := by
  simp only [SimpleGraph.annihilationNumber, e.card_edgeFinset_eq, e.degreeMultiset_eq]

theorem residue_eq (e : G ≃g H) : G.residue = H.residue := by
  change residueAux (G.degreeMultiset.sort (· ≥ ·)) =
    residueAux (H.degreeMultiset.sort (· ≥ ·))
  rw [e.degreeMultiset_eq]

theorem txgraffiti_equality_iff (e : G ≃g H) :
    (G.maxDegree : ℚ) * (G.indepNum : ℚ) =
        (G.annihilationNumber : ℚ) + (G.residue : ℚ) ↔
      (H.maxDegree : ℚ) * (H.indepNum : ℚ) =
        (H.annihilationNumber : ℚ) + (H.residue : ℚ) := by
  rw [e.maxDegree_eq, e.indepNum_eq, e.annihilationNumber_eq, e.residue_eq]

end Iso

/-- The eight connected graph-isomorphism types attaining TxGraffiti equality. -/
theorem txgraffiti_equality_classification
    {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
    [DecidableRel G.Adj] (hconn : G.Connected) (hΔ : 2 ≤ G.maxDegree) :
    (G.maxDegree : ℚ) * (G.indepNum : ℚ) =
        (G.annihilationNumber : ℚ) + (G.residue : ℚ) ↔
      Nonempty (G ≃g pathGraph 3) ∨
      Nonempty (G ≃g pathGraph 4) ∨
      Nonempty (G ≃g pathGraph 6) ∨
      Nonempty (G ≃g cycleGraph 3) ∨
      Nonempty (G ≃g cycleGraph 4) ∨
      Nonempty (G ≃g cycleGraph 5) ∨
      Nonempty (G ≃g cycleGraph 7) ∨
      Nonempty (G ≃g completeGraph (Fin 4)) := by
  constructor
  · intro heq
    by_cases hΔ3 : 3 ≤ G.maxDegree
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
        (G.iso_completeGraph_four_of_maxDegree_ge_three hconn hΔ3 heq)))))))
    · have hΔ2 : G.maxDegree ≤ 2 := by omega
      have hn : 3 ≤ Fintype.card V := G.three_le_card_of_two_le_maxDegree hconn hΔ
      have hres : G.residue = G.indepNum := by
        have h := (G.txgraffiti_eq_iff_invariants hconn hΔ).mp heq
        exact_mod_cast h.2.symm
      rcases G.iso_pathGraph_or_cycleGraph_of_connected_of_maxDegree_le_two hconn hΔ2 with
        hpath | hcycle
      · obtain ⟨e⟩ := hpath
        have he : (pathGraph (Fintype.card V)).residue =
            (pathGraph (Fintype.card V)).indepNum := by
          rw [← e.residue_eq, ← e.indepNum_eq]
          exact hres
        rcases eq_three_or_four_or_six_of_pathGraph_residue_eq_indepNum
          (Fintype.card V) hn he with h3 | h4 | h6
        · exact Or.inl ⟨h3 ▸ e⟩
        · exact Or.inr (Or.inl ⟨h4 ▸ e⟩)
        · exact Or.inr (Or.inr (Or.inl ⟨h6 ▸ e⟩))
      · obtain ⟨e⟩ := hcycle
        have he : (cycleGraph (Fintype.card V)).residue =
            (cycleGraph (Fintype.card V)).indepNum := by
          rw [← e.residue_eq, ← e.indepNum_eq]
          exact hres
        rcases eq_three_or_four_or_five_or_seven_of_cycleGraph_residue_eq_indepNum
          (Fintype.card V) hn he with h3 | h4 | h5 | h7
        · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨h3 ▸ e⟩)))
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨h4 ▸ e⟩))))
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨h5 ▸ e⟩)))))
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨h7 ▸ e⟩))))))
  · rintro (hp3 | hp4 | hp6 | hc3 | hc4 | hc5 | hc7 | hk4)
    · obtain ⟨e⟩ := hp3
      exact e.txgraffiti_equality_iff.mpr p3_eq
    · obtain ⟨e⟩ := hp4
      exact e.txgraffiti_equality_iff.mpr p4_eq
    · obtain ⟨e⟩ := hp6
      exact e.txgraffiti_equality_iff.mpr p6_eq
    · obtain ⟨e⟩ := hc3
      exact e.txgraffiti_equality_iff.mpr c3_eq
    · obtain ⟨e⟩ := hc4
      exact e.txgraffiti_equality_iff.mpr c4_eq
    · obtain ⟨e⟩ := hc5
      exact e.txgraffiti_equality_iff.mpr c5_eq
    · obtain ⟨e⟩ := hc7
      exact e.txgraffiti_equality_iff.mpr c7_eq
    · obtain ⟨e⟩ := hk4
      exact e.txgraffiti_equality_iff.mpr k4_eq

#print axioms txgraffiti_equality_classification
#print axioms strict_of_maxDegree_ge_four

end SimpleGraph