import Mathlib

namespace OrigamiCone.Sequel.BlockGraph

open scoped BigOperators

inductive Vertex (RowBlock ColumnBlock Extremum : Type*) where
  | rowBlock : RowBlock -> Vertex RowBlock ColumnBlock Extremum
  | columnBlock : ColumnBlock -> Vertex RowBlock ColumnBlock Extremum
  | extremum : Extremum -> Vertex RowBlock ColumnBlock Extremum
  deriving DecidableEq, Fintype

def vertexEquivSum (RowBlock ColumnBlock Extremum : Type*) :
    Vertex RowBlock ColumnBlock Extremum ≃ (RowBlock ⊕ ColumnBlock) ⊕ Extremum where
  toFun := fun vertex => match vertex with
    | Vertex.rowBlock row => Sum.inl (Sum.inl row)
    | Vertex.columnBlock column => Sum.inl (Sum.inr column)
    | Vertex.extremum edge => Sum.inr edge
  invFun := fun vertex => match vertex with
    | Sum.inl (Sum.inl row) => Vertex.rowBlock row
    | Sum.inl (Sum.inr column) => Vertex.columnBlock column
    | Sum.inr edge => Vertex.extremum edge
  left_inv := by intro vertex; cases vertex <;> rfl
  right_inv := by rintro ((row | column) | edge) <;> rfl

def incidenceGraph {RowBlock ColumnBlock Extremum : Type*}
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) :
    SimpleGraph (Vertex RowBlock ColumnBlock Extremum) where
  Adj first second := match first, second with
    | Vertex.rowBlock row, Vertex.extremum edge => rowOf edge = row
    | Vertex.extremum edge, Vertex.rowBlock row => rowOf edge = row
    | Vertex.columnBlock column, Vertex.extremum edge => columnOf edge = column
    | Vertex.extremum edge, Vertex.columnBlock column => columnOf edge = column
    | _, _ => False
  symm := by intro first second adjacent; cases first <;> cases second <;> exact adjacent
  loopless := by intro vertex adjacent; cases vertex <;> exact adjacent

instance {RowBlock ColumnBlock Extremum : Type*} [DecidableEq RowBlock] [DecidableEq ColumnBlock]
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) :
    DecidableRel (incidenceGraph rowOf columnOf).Adj := by
  intro first second
  cases first <;> cases second <;> dsimp only [incidenceGraph] <;> exact inferInstance

theorem card_vertices (RowBlock ColumnBlock Extremum : Type*)
    [Fintype RowBlock] [Fintype ColumnBlock] [Fintype Extremum] :
    Fintype.card (Vertex RowBlock ColumnBlock Extremum) =
      Fintype.card RowBlock + Fintype.card ColumnBlock + Fintype.card Extremum := by
  simpa using Fintype.card_congr (vertexEquivSum RowBlock ColumnBlock Extremum)

theorem neighborSet_extremum {RowBlock ColumnBlock Extremum : Type*}
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) (edge : Extremum) :
    (incidenceGraph rowOf columnOf).neighborSet (Vertex.extremum edge) =
      {Vertex.rowBlock (rowOf edge), Vertex.columnBlock (columnOf edge)} := by
  ext vertex
  cases vertex <;> simp [SimpleGraph.neighborSet, incidenceGraph, eq_comm]

theorem neighborFinset_extremum {RowBlock ColumnBlock Extremum : Type*}
    [Fintype RowBlock] [Fintype ColumnBlock] [Fintype Extremum]
    [DecidableEq RowBlock] [DecidableEq ColumnBlock] [DecidableEq Extremum]
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) (edge : Extremum) :
    (incidenceGraph rowOf columnOf).neighborFinset (Vertex.extremum edge) =
      {Vertex.rowBlock (rowOf edge), Vertex.columnBlock (columnOf edge)} := by
  apply Finset.coe_injective
  simpa using neighborSet_extremum rowOf columnOf edge

theorem degree_extremum {RowBlock ColumnBlock Extremum : Type*}
    [Fintype RowBlock] [Fintype ColumnBlock] [Fintype Extremum]
    [DecidableEq RowBlock] [DecidableEq ColumnBlock] [DecidableEq Extremum]
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) (edge : Extremum) :
    (incidenceGraph rowOf columnOf).degree (Vertex.extremum edge) = 2 := by
  simp [SimpleGraph.degree, neighborFinset_extremum]

theorem incidence_bipartite {RowBlock ColumnBlock Extremum : Type*}
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) :
    (incidenceGraph rowOf columnOf).IsBipartiteWith
      (Set.range (Vertex.extremum : Extremum -> Vertex RowBlock ColumnBlock Extremum))
      (Set.range (Vertex.extremum : Extremum -> Vertex RowBlock ColumnBlock Extremum))ᶜ where
  disjoint := by
    apply Set.disjoint_left.mpr
    intro vertex member complement
    exact complement member
  mem_of_adj := by
    intro first second adjacent
    cases first <;> cases second <;> simp [incidenceGraph, Set.mem_range] at *

theorem card_edges {RowBlock ColumnBlock Extremum : Type*}
    [Fintype RowBlock] [Fintype ColumnBlock] [Fintype Extremum]
    [DecidableEq RowBlock] [DecidableEq ColumnBlock] [DecidableEq Extremum]
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) :
    (incidenceGraph rowOf columnOf).edgeFinset.card = 2 * Fintype.card Extremum := by
  classical
  let graph := incidenceGraph rowOf columnOf
  let left : Finset (Vertex RowBlock ColumnBlock Extremum) := Finset.univ.image Vertex.extremum
  have bipartite : graph.IsBipartiteWith left leftᶜ := by
    simpa [left] using incidence_bipartite rowOf columnOf
  calc
    _ = ∑ vertex ∈ left, graph.degree vertex :=
      (SimpleGraph.isBipartiteWith_sum_degrees_eq_card_edges (s := left) (t := leftᶜ)
        (by simpa using bipartite)).symm
    _ = ∑ edge : Extremum, graph.degree (Vertex.extremum edge) :=
      Finset.sum_image (by intro first _ second _ equality; simpa using equality)
    _ = _ := by simp [graph, degree_extremum, Nat.mul_comm]

theorem degree_rowBlock {RowBlock ColumnBlock Extremum : Type*}
    [Fintype RowBlock] [Fintype ColumnBlock] [Fintype Extremum]
    [DecidableEq RowBlock] [DecidableEq ColumnBlock] [DecidableEq Extremum]
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) (row : RowBlock) :
    (incidenceGraph rowOf columnOf).degree (Vertex.rowBlock row) =
      (Finset.univ.filter (fun edge => rowOf edge = row)).card := by
  classical
  unfold SimpleGraph.degree
  symm
  apply Finset.card_bij (fun edge _ => Vertex.extremum edge)
  · intro edge member
    simpa [SimpleGraph.mem_neighborFinset, incidenceGraph] using (Finset.mem_filter.mp member).2
  · intro first _ second _ equality
    simpa using equality
  · intro vertex member
    cases vertex with
    | rowBlock other => simp [SimpleGraph.mem_neighborFinset, incidenceGraph] at member
    | columnBlock other => simp [SimpleGraph.mem_neighborFinset, incidenceGraph] at member
    | extremum edge =>
      refine ⟨edge, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, rfl⟩
      simpa [SimpleGraph.mem_neighborFinset, incidenceGraph] using member

theorem degree_columnBlock {RowBlock ColumnBlock Extremum : Type*}
    [Fintype RowBlock] [Fintype ColumnBlock] [Fintype Extremum]
    [DecidableEq RowBlock] [DecidableEq ColumnBlock] [DecidableEq Extremum]
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) (column : ColumnBlock) :
    (incidenceGraph rowOf columnOf).degree (Vertex.columnBlock column) =
      (Finset.univ.filter (fun edge => columnOf edge = column)).card := by
  classical
  unfold SimpleGraph.degree
  symm
  apply Finset.card_bij (fun edge _ => Vertex.extremum edge)
  · intro edge member
    simpa [SimpleGraph.mem_neighborFinset, incidenceGraph] using (Finset.mem_filter.mp member).2
  · intro first _ second _ equality
    simpa using equality
  · intro vertex member
    cases vertex with
    | rowBlock other => simp [SimpleGraph.mem_neighborFinset, incidenceGraph] at member
    | columnBlock other => simp [SimpleGraph.mem_neighborFinset, incidenceGraph] at member
    | extremum edge =>
      refine ⟨edge, Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩, rfl⟩
      simpa [SimpleGraph.mem_neighborFinset, incidenceGraph] using member

theorem no_isolated_vertices {RowBlock ColumnBlock Extremum : Type*}
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock)
    (rowsCovered : Function.Surjective rowOf) (columnsCovered : Function.Surjective columnOf)
    (vertex : Vertex RowBlock ColumnBlock Extremum) :
    ∃ other, (incidenceGraph rowOf columnOf).Adj vertex other := by
  cases vertex with
  | rowBlock row =>
    obtain ⟨edge, equality⟩ := rowsCovered row
    exact ⟨Vertex.extremum edge, equality⟩
  | columnBlock column =>
    obtain ⟨edge, equality⟩ := columnsCovered column
    exact ⟨Vertex.extremum edge, equality⟩
  | extremum edge => exact ⟨Vertex.rowBlock (rowOf edge), rfl⟩

theorem unique_of_fiber_card_one {Domain Target : Type*} [Fintype Domain] [DecidableEq Target]
    (function : Domain -> Target) (target : Target)
    (one : (Finset.univ.filter (fun point => function point = target)).card = 1) :
    ∃! point : Domain, function point = target := by
  classical
  obtain ⟨point, singleton⟩ := Finset.card_eq_one.mp one
  refine ⟨point, ?_, ?_⟩
  · have member : point ∈ Finset.univ.filter (fun point => function point = target) := by
      rw [singleton]
      exact Finset.mem_singleton_self _
    exact (Finset.mem_filter.mp member).2
  · intro other equality
    have member : other ∈ Finset.univ.filter (fun point => function point = target) :=
      Finset.mem_filter.mpr ⟨Finset.mem_univ other, equality⟩
    rw [singleton] at member
    exact Finset.mem_singleton.mp member

theorem row_leaf_unique {RowBlock ColumnBlock Extremum : Type*}
    [Fintype RowBlock] [Fintype ColumnBlock] [Fintype Extremum]
    [DecidableEq RowBlock] [DecidableEq ColumnBlock] [DecidableEq Extremum]
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) (row : RowBlock)
    (leaf : (incidenceGraph rowOf columnOf).degree (Vertex.rowBlock row) = 1) :
    ∃! edge : Extremum, rowOf edge = row := by
  rw [degree_rowBlock] at leaf
  exact unique_of_fiber_card_one rowOf row leaf

theorem column_leaf_unique {RowBlock ColumnBlock Extremum : Type*}
    [Fintype RowBlock] [Fintype ColumnBlock] [Fintype Extremum]
    [DecidableEq RowBlock] [DecidableEq ColumnBlock] [DecidableEq Extremum]
    (rowOf : Extremum -> RowBlock) (columnOf : Extremum -> ColumnBlock) (column : ColumnBlock)
    (leaf : (incidenceGraph rowOf columnOf).degree (Vertex.columnBlock column) = 1) :
    ∃! edge : Extremum, columnOf edge = column := by
  rw [degree_columnBlock] at leaf
  exact unique_of_fiber_card_one columnOf column leaf

#print axioms vertexEquivSum
#print axioms incidenceGraph
#print axioms card_vertices
#print axioms neighborSet_extremum
#print axioms neighborFinset_extremum
#print axioms degree_extremum
#print axioms incidence_bipartite
#print axioms card_edges
#print axioms degree_rowBlock
#print axioms degree_columnBlock
#print axioms no_isolated_vertices
#print axioms unique_of_fiber_card_one
#print axioms row_leaf_unique
#print axioms column_leaf_unique

end OrigamiCone.Sequel.BlockGraph
