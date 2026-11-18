import Mathlib

namespace OrigamiCone.Sequel.TreeInventory

noncomputable section

open scoped BigOperators

theorem two_tree_indices {Component : Type*} [Fintype Component]
    (vertices edges : Component -> Nat)
    (connectedBound : ∀ component, vertices component ≤ edges component + 1)
    (surplus : (∑ component, edges component) + 2 ≤ ∑ component, vertices component) :
    ∃ first second : Component, first ≠ second ∧
      edges first + 1 = vertices first ∧ edges second + 1 = vertices second := by
  classical
  let trees := Finset.univ.filter (fun component => edges component + 1 = vertices component)
  have pointwise : ∀ component,
      vertices component ≤ edges component + (if edges component + 1 = vertices component then 1 else 0) := by
    intro component
    have bound := connectedBound component
    by_cases tree : edges component + 1 = vertices component
    · rw [if_pos tree]
      exact bound
    · rw [if_neg tree]
      omega
  have indicator : trees.card =
      ∑ component, if edges component + 1 = vertices component then 1 else 0 := by
    change (Finset.univ.filter (fun component => edges component + 1 = vertices component)).card = _
    rw [Finset.card_eq_sum_ones, Finset.sum_filter]
  have global : (∑ component, vertices component) ≤ (∑ component, edges component) + trees.card := by
    calc
      _ ≤ ∑ component, (edges component + (if edges component + 1 = vertices component then 1 else 0)) :=
        Finset.sum_le_sum (fun component _ => pointwise component)
      _ = (∑ component, edges component) +
          ∑ component, if edges component + 1 = vertices component then 1 else 0 := Finset.sum_add_distrib
      _ = _ := congrArg (fun total : Nat => (∑ component, edges component) + total) indicator.symm
  have atLeastTwo : 1 < trees.card := by omega
  obtain ⟨first, firstMember, second, secondMember, different⟩ := Finset.one_lt_card.mp atLeastTwo
  exact ⟨first, second, different, (Finset.mem_filter.mp firstMember).2, (Finset.mem_filter.mp secondMember).2⟩

section Components

variable {Vertex : Type*} [Fintype Vertex] (graph : SimpleGraph Vertex)

local instance : Fintype graph.ConnectedComponent := Fintype.ofFinite _

local instance (component : graph.ConnectedComponent) : Fintype component := Fintype.ofFinite _

local instance [DecidableRel graph.Adj] (component : graph.ConnectedComponent) :
    DecidableRel component.toSimpleGraph.Adj :=
  fun first second => inferInstanceAs (Decidable (graph.Adj first.val second.val))

def componentVertexEquiv : (Sigma fun component : graph.ConnectedComponent => component) ≃ Vertex where
  toFun := fun point => point.snd.val
  invFun := fun vertex => ⟨graph.connectedComponentMk vertex,
    ⟨vertex, SimpleGraph.ConnectedComponent.connectedComponentMk_mem⟩⟩
  left_inv := by
    rintro ⟨component, ⟨vertex, member⟩⟩
    change graph.connectedComponentMk vertex = component at member
    cases member
    rfl
  right_inv := fun _ => rfl

theorem sum_component_vertices :
    (∑ component : graph.ConnectedComponent, Fintype.card component) = Fintype.card Vertex := by
  rw [← Fintype.card_sigma]
  exact Fintype.card_congr (componentVertexEquiv graph)

theorem component_degree [DecidableRel graph.Adj] (component : graph.ConnectedComponent) (vertex : component) :
    component.toSimpleGraph.degree vertex = graph.degree vertex.val := by
  classical
  unfold SimpleGraph.degree
  apply Finset.card_bij (fun neighbor _ => neighbor.val)
  · intro neighbor member
    simpa only [SimpleGraph.mem_neighborFinset] using member
  · intro first _ second _ equality
    exact Subtype.ext equality
  · intro neighbor member
    have adjacent : graph.Adj vertex.val neighbor := by
      simpa only [SimpleGraph.mem_neighborFinset] using member
    let original : component := ⟨neighbor, component.mem_supp_of_adj_mem_supp vertex.property adjacent⟩
    exact ⟨original, by simpa only [SimpleGraph.mem_neighborFinset] using adjacent, rfl⟩

theorem sum_component_degrees [DecidableRel graph.Adj] :
    (∑ component : graph.ConnectedComponent, ∑ vertex : component, graph.degree vertex.val) =
      ∑ vertex : Vertex, graph.degree vertex := by
  classical
  rw [← Fintype.sum_sigma (fun point : (Σ component : graph.ConnectedComponent, component) => graph.degree point.snd.val)]
  apply Finset.sum_bij (fun point _ => point.snd.val)
  · intro point _
    exact Finset.mem_univ _
  · intro first _ second _ equality
    exact (componentVertexEquiv graph).injective equality
  · intro vertex _
    exact ⟨(componentVertexEquiv graph).symm vertex, Finset.mem_univ _,
      (componentVertexEquiv graph).apply_symm_apply vertex⟩
  · intro point _
    rfl

theorem sum_component_edges [DecidableEq Vertex] [DecidableRel graph.Adj] :
    (∑ component : graph.ConnectedComponent, component.toSimpleGraph.edgeFinset.card) = graph.edgeFinset.card := by
  classical
  have twice : 2 * (∑ component : graph.ConnectedComponent, component.toSimpleGraph.edgeFinset.card) =
      2 * graph.edgeFinset.card := by
    calc
      _ = ∑ component : graph.ConnectedComponent, (2 * component.toSimpleGraph.edgeFinset.card) := by
        rw [Finset.mul_sum]
      _ = ∑ component : graph.ConnectedComponent, ∑ vertex : component, component.toSimpleGraph.degree vertex := by
        apply Finset.sum_congr rfl
        intro component _
        exact component.toSimpleGraph.sum_degrees_eq_twice_card_edges.symm
      _ = ∑ component : graph.ConnectedComponent, ∑ vertex : component, graph.degree vertex.val := by
        apply Finset.sum_congr rfl
        intro component _
        exact Finset.sum_congr rfl (fun vertex _ => component_degree graph component vertex)
      _ = ∑ vertex : Vertex, graph.degree vertex := sum_component_degrees graph
      _ = _ := graph.sum_degrees_eq_twice_card_edges
  omega

theorem component_connected_bound [DecidableEq Vertex] [DecidableRel graph.Adj]
    (component : graph.ConnectedComponent) :
    Fintype.card component ≤ component.toSimpleGraph.edgeFinset.card + 1 := by
  classical
  have bound := component.connected_toSimpleGraph.card_vert_le_card_edgeSet_add_one
  simpa only [Nat.card_eq_fintype_card, ← SimpleGraph.edgeFinset_card] using bound

theorem component_tree_of_count [DecidableEq Vertex] [DecidableRel graph.Adj]
    (component : graph.ConnectedComponent)
    (treeCount : component.toSimpleGraph.edgeFinset.card + 1 = Fintype.card component) :
    component.toSimpleGraph.IsTree := by
  classical
  apply SimpleGraph.isTree_iff_connected_and_card.mpr
  refine ⟨component.connected_toSimpleGraph, ?_⟩
  simpa only [Nat.card_eq_fintype_card, ← SimpleGraph.edgeFinset_card] using treeCount

theorem two_tree_components [DecidableEq Vertex] [DecidableRel graph.Adj]
    (surplus : graph.edgeFinset.card + 2 ≤ Fintype.card Vertex) :
    ∃ first second : graph.ConnectedComponent, first ≠ second ∧
      first.toSimpleGraph.IsTree ∧ second.toSimpleGraph.IsTree := by
  classical
  have componentSurplus : (∑ component : graph.ConnectedComponent, component.toSimpleGraph.edgeFinset.card) + 2 ≤
      ∑ component : graph.ConnectedComponent, Fintype.card component := by
    rw [sum_component_edges graph, sum_component_vertices graph]
    exact surplus
  obtain ⟨first, second, different, firstCount, secondCount⟩ :=
    two_tree_indices (fun component : graph.ConnectedComponent => Fintype.card component)
      (fun component => component.toSimpleGraph.edgeFinset.card) (component_connected_bound graph) componentSurplus
  exact ⟨first, second, different, component_tree_of_count graph first firstCount,
    component_tree_of_count graph second secondCount⟩

end Components

#print axioms two_tree_indices
#print axioms componentVertexEquiv
#print axioms sum_component_vertices
#print axioms component_degree
#print axioms sum_component_degrees
#print axioms sum_component_edges
#print axioms component_connected_bound
#print axioms component_tree_of_count
#print axioms two_tree_components

end

end OrigamiCone.Sequel.TreeInventory
