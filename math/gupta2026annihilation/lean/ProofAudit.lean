import TxGraffitiC1FC

#check SimpleGraph.txgraffiti_conjecture_1
#print axioms SimpleGraph.annih_vehicle
#print axioms SimpleGraph.caroWei_le_indepNum
#print axioms SimpleGraph.indepNum_le_annihilationNumber
#print axioms SimpleGraph.annih_le_predMaxDegree_indepNum
#print axioms SimpleGraph.residue_le_residue_induce_compl_of_maxDegree
#print axioms SimpleGraph.residue_le_indepNum
#print axioms SimpleGraph.txgraffiti_conjecture_1
#print axioms SimpleGraph.txgraffiti_equality_classification
#print axioms SimpleGraph.strict_of_maxDegree_ge_four

universe u

#check (SimpleGraph.txgraffiti_conjecture_1.{u} :
  ∀ {V : Type u} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
    [DecidableRel G.Adj], G.Connected → 2 ≤ G.maxDegree →
    (G.annihilationNumber : ℚ) + (G.residue : ℚ) ≤
      (G.maxDegree : ℚ) * (G.indepNum : ℚ))

#check (SimpleGraph.txgraffiti_equality_classification.{u} :
  ∀ {V : Type u} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
    [DecidableRel G.Adj], G.Connected → 2 ≤ G.maxDegree →
    ((G.maxDegree : ℚ) * (G.indepNum : ℚ) =
        (G.annihilationNumber : ℚ) + (G.residue : ℚ) ↔
      Nonempty (G ≃g SimpleGraph.pathGraph 3) ∨
      Nonempty (G ≃g SimpleGraph.pathGraph 4) ∨
      Nonempty (G ≃g SimpleGraph.pathGraph 6) ∨
      Nonempty (G ≃g SimpleGraph.cycleGraph 3) ∨
      Nonempty (G ≃g SimpleGraph.cycleGraph 4) ∨
      Nonempty (G ≃g SimpleGraph.cycleGraph 5) ∨
      Nonempty (G ≃g SimpleGraph.cycleGraph 7) ∨
      Nonempty (G ≃g SimpleGraph.completeGraph (Fin 4))))

#check (SimpleGraph.strict_of_maxDegree_ge_four.{u} :
  ∀ {V : Type u} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
    [DecidableRel G.Adj], G.Connected → 4 ≤ G.maxDegree →
    (G.annihilationNumber : ℚ) + (G.residue : ℚ) <
      (G.maxDegree : ℚ) * (G.indepNum : ℚ))