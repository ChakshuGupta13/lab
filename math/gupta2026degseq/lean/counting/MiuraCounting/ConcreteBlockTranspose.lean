import MiuraCounting.ConcreteOccupiedBlocks

namespace OrigamiCone.Sequel.BlockTranspose

open OrigamiCone OccupiedBlocks

def transposeExtremumEquiv {rows width : Nat} (height : Cell rows width -> Int) :
    Extrema (swapH height) ≃ Extrema height :=
  (Equiv.prodComm (Fin width) (Fin rows)).subtypeEquiv
    (fun point => swapH_strictExtremum_iff (h := height) point)

def transposeVertexEquiv {rows width : Nat} (height : Cell rows width -> Int) :
    GraphVertex height ≃ GraphVertex (swapH height) where
  toFun := fun vertex => match vertex with
    | BlockGraph.Vertex.rowBlock row => BlockGraph.Vertex.columnBlock row
    | BlockGraph.Vertex.columnBlock column => BlockGraph.Vertex.rowBlock column
    | BlockGraph.Vertex.extremum point => BlockGraph.Vertex.extremum ((transposeExtremumEquiv height).symm point)
  invFun := fun vertex => match vertex with
    | BlockGraph.Vertex.rowBlock row => BlockGraph.Vertex.columnBlock row
    | BlockGraph.Vertex.columnBlock column => BlockGraph.Vertex.rowBlock column
    | BlockGraph.Vertex.extremum point => BlockGraph.Vertex.extremum (transposeExtremumEquiv height point)
  left_inv := by
    intro vertex
    cases vertex with
    | rowBlock row => rfl
    | columnBlock column => rfl
    | extremum point => exact congrArg BlockGraph.Vertex.extremum ((transposeExtremumEquiv height).apply_symm_apply point)
  right_inv := by
    intro vertex
    cases vertex with
    | rowBlock row => rfl
    | columnBlock column => rfl
    | extremum point => exact congrArg BlockGraph.Vertex.extremum ((transposeExtremumEquiv height).symm_apply_apply point)

def transposeGraphIso {rows width : Nat} (height : Cell rows width -> Int) :
    graph height ≃g graph (swapH height) where
  toEquiv := transposeVertexEquiv height
  map_rel_iff' := by
    intro first second
    cases first <;> cases second <;> rfl

abbrev transposePoint {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    Extrema (swapH height) := (transposeExtremumEquiv height).symm point

theorem transpose_row_degree {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    (graph (swapH height)).degree (BlockGraph.Vertex.columnBlock (columnOf (swapH height) (transposePoint height point))) =
      (graph height).degree (BlockGraph.Vertex.rowBlock (rowOf height point)) :=
  (transposeGraphIso height).degree_eq (BlockGraph.Vertex.rowBlock (rowOf height point))

theorem transpose_column_degree {rows width : Nat} (height : Cell rows width -> Int) (point : Extrema height) :
    (graph (swapH height)).degree (BlockGraph.Vertex.rowBlock (rowOf (swapH height) (transposePoint height point))) =
      (graph height).degree (BlockGraph.Vertex.columnBlock (columnOf height point)) :=
  (transposeGraphIso height).degree_eq (BlockGraph.Vertex.columnBlock (columnOf height point))

theorem transpose_reachable_iff {rows width : Nat} (height : Cell rows width -> Int) (first second : Extrema height) :
    (graph (swapH height)).Reachable (BlockGraph.Vertex.extremum (transposePoint height first))
      (BlockGraph.Vertex.extremum (transposePoint height second)) ↔
    (graph height).Reachable (BlockGraph.Vertex.extremum first) (BlockGraph.Vertex.extremum second) :=
  (transposeGraphIso height).reachable_iff
    (u := BlockGraph.Vertex.extremum first) (v := BlockGraph.Vertex.extremum second)

#print axioms transposeExtremumEquiv
#print axioms transposeVertexEquiv
#print axioms transposeGraphIso
#print axioms transpose_row_degree
#print axioms transpose_column_degree
#print axioms transpose_reachable_iff

end OrigamiCone.Sequel.BlockTranspose
