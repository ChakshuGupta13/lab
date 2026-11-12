import TxGraffitiC1FC.Delta2

set_option linter.unusedSectionVars false

namespace SimpleGraph

open Finset Walk
open scoped Classical

variable {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V) [DecidableRel G.Adj]

/-- If `H : G.Subgraph` has `H.verts.Nonempty`, `G` is connected, and every `z ∈ H.verts`
has `H.degree z = G.degree z`, then `H = ⊤` (`H.verts = Set.univ` and `H.Adj = G.Adj`). -/
theorem Subgraph.eq_top_of_degree_eq_of_connected
    (hconn : G.Connected) (H : G.Subgraph) [DecidableRel H.Adj]
    (hne : H.verts.Nonempty)
    (hdeg : ∀ z ∈ H.verts, H.degree z = G.degree z) :
    H = ⊤ := by
  have hall_adj : ∀ z ∈ H.verts, ∀ w, G.Adj z w → H.Adj z w := by
    intro z hz w hadj
    have hsub : (H.neighborSet z).toFinset ⊆ G.neighborFinset z := by
      intro x hx
      rw [Set.mem_toFinset, Subgraph.mem_neighborSet] at hx
      rw [mem_neighborFinset]
      exact H.adj_sub hx
    have hcard : (H.neighborSet z).toFinset.card = (G.neighborFinset z).card := by
      rw [H.finset_card_neighborSet_eq_degree, card_neighborFinset_eq_degree, hdeg z hz]
    have heq := Finset.eq_of_subset_of_card_le hsub hcard.ge
    have hw : w ∈ (H.neighborSet z).toFinset := by
      rw [heq, mem_neighborFinset]; exact hadj
    rwa [Set.mem_toFinset, Subgraph.mem_neighborSet] at hw
  obtain ⟨v₀, hv₀⟩ := hne
  have hverts : H.verts = Set.univ := by
    ext w
    simp only [Set.mem_univ, iff_true]
    exact (hconn v₀ w).mem_subgraphVerts hall_adj hv₀
  ext x y
  · rw [hverts]; rfl
  · simp only [Subgraph.top_adj]
    exact ⟨fun h => H.adj_sub h, fun h => hall_adj x (hverts ▸ Set.mem_univ x) y h⟩

/-- A bijective `Copy H G` where every vertex of `H` already has degree at least its image's
degree in `G` is a graph isomorphism `H ≃g G`. -/
noncomputable def Copy.toIsoOfBijectiveOfDegreeGe
    {W : Type*} [Fintype W] [DecidableEq W] {H : SimpleGraph W} [DecidableRel H.Adj]
    (f : Copy H G) (hbij : Function.Bijective f)
    (hdeg : ∀ w : W, G.degree (f w) ≤ H.degree w) :
    H ≃g G where
  toEquiv := Equiv.ofBijective f hbij
  map_rel_iff' {x y} := by
    refine ⟨fun hadj => ?_, fun hadj => f.toHom.map_adj hadj⟩
    -- Since f.mapNeighborSet x : H.neighborSet x ↪ G.neighborSet (f x) and card ≥ card, it is surjective!
    have hcard_eq : Fintype.card (H.neighborSet x) = Fintype.card (G.neighborSet (f x)) := by
      rw [card_neighborSet_eq_degree, card_neighborSet_eq_degree]
      exact le_antisymm (f.degree_le x) (hdeg x)
    have hsurj : Function.Surjective (f.mapNeighborSet x) :=
      ((Fintype.bijective_iff_injective_and_card (f.mapNeighborSet x)).mpr
        ⟨(f.mapNeighborSet x).injective, hcard_eq⟩).2
    obtain ⟨⟨z, hz⟩, hfz⟩ := hsurj ⟨f y, hadj⟩
    have hzy : z = y := f.injective (Subtype.mk.inj hfz)
    exact hzy ▸ hz

/-- **Cycle classification for connected `Δ ≤ 2` graphs.**
If `G` is connected with `G.maxDegree ≤ 2` and `c : G.Walk u u` is a cycle (`c.IsCycle`),
then `G` is isomorphic to `cycleGraph c.length` (and `c.length = Fintype.card V`). -/
theorem iso_cycleGraph_of_isCycle_of_maxDegree_le_two
    (hconn : G.Connected) (hΔ : G.maxDegree ≤ 2) {u : V} {c : G.Walk u u} (hc : c.IsCycle) :
    c.length = Fintype.card V ∧ Nonempty (G ≃g cycleGraph c.length) := by
  have h3 : 3 ≤ c.length := hc.three_le_length
  have htail_len : c.support.tail.length = c.length := by
    rw [List.length_tail, c.length_support, Nat.add_sub_cancel]
  have hlen_le : c.length ≤ Fintype.card V := by
    rw [← htail_len]
    exact hc.support_nodup.length_le_card
  have hcont : cycleGraph c.length ⊑ G := by
    rw [cycleGraph_isContained_iff (by omega)]
    exact ⟨u, c, hc, rfl⟩
  obtain ⟨f⟩ := hcont
  have hdeg_cycle : ∀ n : ℕ, 3 ≤ n → ∀ w : Fin n, (cycleGraph n).degree w = 2 := by
    intro n hn
    obtain ⟨k, rfl⟩ : ∃ k, n = k + 3 := ⟨n - 3, by omega⟩
    intro w
    exact cycleGraph_degree_three_le
  -- Now consider the image of `f` in `G`: every vertex `f w` has degree ≥ 2 in the subgraph,
  -- hence degree = 2 in `G`!

  -- which would give a vertex `f w` of degree ≥ 3 in `G`, impossible!
  have hrange : Set.range f = Set.univ := by
    ext z
    simp only [Set.mem_univ, iff_true]
    obtain ⟨p⟩ := hconn (f ⟨0, by omega⟩) z
    by_contra hz
    obtain ⟨d, _, hdfst, hdsnd⟩ :=
      p.exists_boundary_dart (Set.range f) ⟨⟨0, by omega⟩, rfl⟩ hz
    obtain ⟨w, hfw⟩ := hdfst
    -- All neighbors of `w` in `cycleGraph c.length` map into `Set.range f ∩ G.neighborSet (f w)`,
    -- whereas `d.snd ∉ Set.range f` is ALSO in `G.neighborSet (f w)`!
    have hmap_sub : Finset.univ.map (f.mapNeighborSet w) ⊆ Finset.univ := Finset.subset_univ _
    have hnot_mem : (⟨d.snd, hfw ▸ d.adj⟩ : G.neighborSet (f w)) ∉ Finset.univ.map (f.mapNeighborSet w) := by
      simp only [Finset.mem_map, Finset.mem_univ, true_and, not_exists]
      rintro ⟨w', hw'⟩ heq
      apply hdsnd
      exact ⟨w', Subtype.mk.inj heq⟩
    have hssub : Finset.univ.map (f.mapNeighborSet w) ⊂ Finset.univ :=
      Finset.ssubset_iff_of_subset hmap_sub |>.mpr ⟨⟨d.snd, hfw ▸ d.adj⟩, Finset.mem_univ _, hnot_mem⟩
    have hlt := Finset.card_lt_card hssub
    rw [Finset.card_map, Finset.card_univ, Finset.card_univ,
        card_neighborSet_eq_degree, card_neighborSet_eq_degree, hdeg_cycle c.length h3 w] at hlt
    have hle2 := G.degree_le_two hΔ (f w)
    omega
  have hsurj : Function.Surjective f := Set.range_eq_univ.mp hrange
  have hbij : Function.Bijective f := ⟨f.injective, hsurj⟩
  have hlen_eq : c.length = Fintype.card V := by
    rw [← Fintype.card_fin c.length]
    exact Fintype.card_of_bijective hbij
  have hdeg_ge : ∀ w : Fin c.length, G.degree (f w) ≤ (cycleGraph c.length).degree w := by
    intro w
    rw [hdeg_cycle c.length h3 w]
    exact G.degree_le_two hΔ (f w)
  exact ⟨hlen_eq, ⟨(f.toIsoOfBijectiveOfDegreeGe G hbij hdeg_ge).symm⟩⟩

#print axioms iso_cycleGraph_of_isCycle_of_maxDegree_le_two

/-- **Path classification for trees with `Δ ≤ 2`.**
If `G` is a tree (`G.IsTree`) with `G.maxDegree ≤ 2`, take a path `p : G.Walk u v` of
maximal length in `G`. No neighbor of `u` or `v` can lie outside `p.support` (else `p`
could be extended by `IsPath.cons` or `IsPath.concat`), and by acyclicity (`IsAcyclic.eq_snd_of_adj_start`
/ `eq_penultimate_of_adj_end`) the only neighbor of `u` (resp. `v`) in `G` is `p.snd` (resp. `p.penultimate`).
Internal vertices `p.getVert i` (`0 < i < p.length`) already have 2 neighbors in `p.toSubgraph`,
so `G.maxDegree ≤ 2` forces every vertex of `p.toSubgraph` to have `p.toSubgraph.degree z = G.degree z`.
By connectivity `p.toSubgraph = ⊤`, and `hp.pathGraphIsoToSubgraph` composed with `Subgraph.topIso`
gives `pathGraph (p.length + 1) ≃g G` with `p.length + 1 = Fintype.card V`. -/
theorem iso_pathGraph_of_isTree_of_maxDegree_le_two
    (htree : G.IsTree) (hΔ : G.maxDegree ≤ 2) :
    Nonempty (G ≃g pathGraph (Fintype.card V)) := by
  haveI : Nonempty V := htree.connected.nonempty
  obtain ⟨u, v, p, hp, hmax⟩ := Walk.exists_isPath_forall_isPath_length_le_length G
  -- First show every neighbor in G of any vertex z ∈ p.support is already in p.toSubgraph!
  have hall_adj : ∀ z ∈ p.toSubgraph.verts, ∀ w, G.Adj z w → p.toSubgraph.Adj z w := by
    intro z hz w hadj
    rw [p.mem_verts_toSubgraph, Walk.mem_support_iff_exists_getVert] at hz
    obtain ⟨i, rfl, hi⟩ := hz
    by_cases hi0 : i = 0
    · subst hi0
      rw [Walk.getVert_zero] at hadj ⊢
      have hw_supp : w ∈ p.support := by
        by_contra hwnot
        have hp' : (Walk.cons hadj.symm p).IsPath := hp.cons hwnot
        have hle := hmax w v (Walk.cons hadj.symm p) hp'
        simp only [Walk.length_cons] at hle
        omega
      have hnp : ¬ p.Nil := by
        intro hnil
        rw [Walk.nil_iff_support_eq.mp hnil, List.mem_singleton] at hw_supp
        exact hadj.ne hw_supp.symm
      have hwsnd : w = p.snd := htree.isAcyclic.eq_snd_of_adj_start hp hadj hw_supp
      rw [hwsnd]
      exact p.toSubgraph_adj_snd hnp
    · by_cases hil : i = p.length
      · subst hil
        rw [Walk.getVert_length] at hadj ⊢
        have hw_supp : w ∈ p.support := by
          by_contra hwnot
          have hp' : (p.concat hadj).IsPath := hp.concat hwnot hadj
          have hle := hmax u w (p.concat hadj) hp'
          simp only [Walk.length_concat] at hle
          omega
        have hnp : ¬ p.Nil := by
          intro hnil
          rw [Walk.nil_iff_support_eq.mp hnil, List.mem_singleton] at hw_supp
          exact hadj.ne ( (Walk.Nil.eq hnil).symm.trans hw_supp.symm )
        have hwpen : w = p.penultimate := htree.isAcyclic.eq_penultimate_of_adj_end hp hadj hw_supp
        rw [hwpen]
        exact (p.toSubgraph_adj_penultimate hnp).symm
      · -- Internal vertex 0 < i < p.length: already has degree 2 in p.toSubgraph!
        have hilt : i < p.length := Nat.lt_of_le_of_ne hi hil
        have hncard := hp.ncard_neighborSet_toSubgraph_internal_eq_two hi0 hilt
        have hcard2 : (p.toSubgraph.neighborSet (p.getVert i)).toFinset.card = 2 := by
          rw [← Set.ncard_eq_toFinset_card']
          exact hncard
        have hsub : (p.toSubgraph.neighborSet (p.getVert i)).toFinset ⊆ G.neighborFinset (p.getVert i) := by
          intro x hx
          rw [Set.mem_toFinset, Subgraph.mem_neighborSet] at hx
          rw [mem_neighborFinset]
          exact p.toSubgraph.adj_sub hx
        have hle2 : (G.neighborFinset (p.getVert i)).card ≤ 2 := by
          rw [card_neighborFinset_eq_degree]
          exact G.degree_le_two hΔ (p.getVert i)
        have heq := Finset.eq_of_subset_of_card_le hsub (by omega)
        have hw_mem : w ∈ (p.toSubgraph.neighborSet (p.getVert i)).toFinset := by
          rw [heq, mem_neighborFinset]; exact hadj
        rwa [Set.mem_toFinset, Subgraph.mem_neighborSet] at hw_mem
  have hverts : p.toSubgraph.verts = Set.univ := by
    ext w
    simp only [Set.mem_univ, iff_true]
    exact (htree.connected u w).mem_subgraphVerts hall_adj p.start_mem_verts_toSubgraph
  have htop : p.toSubgraph = ⊤ := by
    ext x y
    · rw [hverts]; rfl
    · simp only [Subgraph.top_adj]
      exact ⟨fun h => p.toSubgraph.adj_sub h, fun h => hall_adj x (hverts ▸ Set.mem_univ x) y h⟩
  have hcard_eq : p.length + 1 = Fintype.card V := by
    apply le_antisymm
    · rw [← p.length_support]
      exact hp.support_nodup.length_le_card
    · have hsub : (Finset.univ : Finset V) ⊆ p.support.toFinset := by
        intro z _
        rw [List.mem_toFinset, ← p.mem_verts_toSubgraph, hverts]
        exact Set.mem_univ z
      calc Fintype.card V = (Finset.univ : Finset V).card := Finset.card_univ.symm
        _ ≤ p.support.toFinset.card := Finset.card_le_card hsub
        _ = p.support.length := List.toFinset_card_of_nodup hp.support_nodup
        _ = p.length + 1 := p.length_support
  -- Now `hp.pathGraphIsoToSubgraph : pathGraph (p.length + 1) ≃g p.toSubgraph.coe`
  -- and since `p.toSubgraph = ⊤`, `p.toSubgraph.coe ≃g G`!
  let isoSub : p.toSubgraph.coe ≃g G :=
    { toFun := fun x => x.val
      invFun := fun x => ⟨x, hverts ▸ Set.mem_univ x⟩
      left_inv := fun ⟨_, _⟩ => rfl
      right_inv := fun _ => rfl
      map_rel_iff' := by
        intro a b
        simp only [Equiv.coe_fn_mk, Subgraph.coe_adj]
        exact ⟨fun h => hall_adj a.val a.prop b.val h, fun h => p.toSubgraph.adj_sub h⟩ }
  have hiso : G ≃g pathGraph (p.length + 1) :=
    (isoSub.comp hp.pathGraphIsoToSubgraph).symm
  exact ⟨hcard_eq ▸ hiso⟩

#print axioms iso_pathGraph_of_isTree_of_maxDegree_le_two

/-- **Structure theorem for connected graphs with `Δ ≤ 2`.**
Every finite simple connected graph `G` with `G.maxDegree ≤ 2` is isomorphic either to
`pathGraph (Fintype.card V)` (when `G` is acyclic, hence a tree) or to
`cycleGraph (Fintype.card V)` (when `G` has a cycle). -/
theorem iso_pathGraph_or_cycleGraph_of_connected_of_maxDegree_le_two
    (hconn : G.Connected) (hΔ : G.maxDegree ≤ 2) :
    Nonempty (G ≃g pathGraph (Fintype.card V)) ∨
    Nonempty (G ≃g cycleGraph (Fintype.card V)) := by
  by_cases hacyc : G.IsAcyclic
  · left
    exact G.iso_pathGraph_of_isTree_of_maxDegree_le_two ⟨hconn, hacyc⟩ hΔ
  · right
    unfold IsAcyclic at hacyc
    push_neg at hacyc
    obtain ⟨u, c, hc⟩ := hacyc
    obtain ⟨hlen, ⟨hiso⟩⟩ := G.iso_cycleGraph_of_isCycle_of_maxDegree_le_two hconn hΔ hc
    exact ⟨hlen ▸ hiso⟩

/-- A connected graph with `2 ≤ G.maxDegree` has at least 3 vertices (`3 ≤ Fintype.card V`). -/
theorem three_le_card_of_two_le_maxDegree (hconn : G.Connected) (hΔ : 2 ≤ G.maxDegree) :
    3 ≤ Fintype.card V := by
  haveI : Nonempty V := hconn.nonempty
  obtain ⟨v, hv⟩ := G.exists_maximal_degree_vertex
  have hlt : G.degree v < Fintype.card V := G.degree_lt_card_verts v
  omega

#print axioms iso_pathGraph_or_cycleGraph_of_connected_of_maxDegree_le_two
#print axioms three_le_card_of_two_le_maxDegree

end SimpleGraph
