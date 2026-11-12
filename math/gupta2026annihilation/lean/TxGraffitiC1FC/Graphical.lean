import TxGraffitiC1FC.Invariants

namespace SimpleGraph

open Classical

/-- A multiset of naturals is **graphical** when it is the degree multiset of some finite
simple graph. -/
def Graphical (m : Multiset ℕ) : Prop :=
  ∃ (V : Type) (_ : Fintype V) (_ : DecidableEq V) (G : SimpleGraph V) (_ : DecidableRel G.Adj),
    G.degreeMultiset = m

/-- Every finite simple graph's degree multiset is graphical: the graph itself witnesses it. -/
theorem graphical_degreeMultiset {V : Type} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
    [DecidableRel G.Adj] : Graphical G.degreeMultiset :=
  ⟨V, inferInstance, inferInstance, G, inferInstance, rfl⟩

theorem graphical_induce_compl {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (v : V) :
    Graphical (G.induce ({v}ᶜ : Set V)).degreeMultiset :=
  graphical_degreeMultiset _

/-- **Vertex-deletion degree formula.** The degree of a vertex `w ≠ v₀` in `G - v₀`
(`= G.induce {v₀}ᶜ`) is its `G`-degree minus one if it was adjacent to `v₀`. A general graph
fact Mathlib lacks (it only has the no-edge-loss case `degree_induce_of_neighborSet_subset`);
the first concrete step of the graph assemble (`σ = deg(G - v)` as a decrement of `rest`). -/
theorem degree_induce_compl_singleton {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (v₀ : V) (w : ({v₀}ᶜ : Set V)) :
    (G.induce ({v₀}ᶜ : Set V)).degree w
      = G.degree (w : V) - (if G.Adj (w : V) v₀ then 1 else 0) := by
  have hA : (G.induce ({v₀}ᶜ : Set V)).degree w
      = (G.neighborFinset (w : V) ∩ ({v₀}ᶜ : Set V).toFinset).card := by
    rw [← card_neighborFinset_eq_degree, ← map_neighborFinset_induce, Finset.card_map]
  have hB : G.neighborFinset (w : V) ∩ ({v₀}ᶜ : Set V).toFinset
      = (G.neighborFinset (w : V)).erase v₀ := by
    ext x
    simp only [Finset.mem_inter, Set.mem_toFinset, Set.mem_compl_iff, Set.mem_singleton_iff,
      Finset.mem_erase]
    tauto
  rw [hA, hB, Finset.card_erase_eq_ite]
  simp only [mem_neighborFinset, card_neighborFinset_eq_degree]
  split_ifs <;> simp

theorem degreeMultiset_induce_compl {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (v : V) :
    (G.induce ({v}ᶜ : Set V)).degreeMultiset
      = (Finset.univ.erase v).val.map (fun w => G.degree w - if G.Adj w v then 1 else 0) := by
  unfold degreeMultiset
  rw [Multiset.map_congr rfl (fun u _ => degree_induce_compl_singleton G v u)]
  have hres : (fun u : ({v}ᶜ : Set V) => G.degree (u : V) - (if G.Adj (u : V) v then 1 else 0))
      = Subtype.restrict (· ∈ ({v}ᶜ : Set V)) (fun w => G.degree w - if G.Adj w v then 1 else 0) :=
    rfl
  rw [hres, Finset.univ_val_map_subtype_restrict]
  have hfe : Finset.univ.filter (· ∈ ({v}ᶜ : Set V)) = Finset.univ.erase v := by
    ext w; simp
  rw [hfe]

theorem sum_adj_indicator_eq_degree {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (v : V) :
    ((Finset.univ.erase v).toList.map (fun w => if G.Adj w v then 1 else 0)).sum
      = G.degree v := by
  rw [← List.sum_toFinset _ (Finset.nodup_toList _), Finset.toList_toFinset, ← Finset.card_filter,
    ← card_neighborFinset_eq_degree]
  congr 1
  ext w
  simp only [Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, and_true,
    mem_neighborFinset]
  constructor
  · rintro ⟨-, hadj⟩; exact G.adj_symm hadj
  · intro hadj; exact ⟨(G.ne_of_adj hadj).symm, G.adj_symm hadj⟩

/-- **Sorted degree sequence splits off the max.** For a maximum-degree vertex `v`, the descending
sort of all degrees is `deg v` followed by the descending sort of the non-`v` degrees. This is the
`sort(deg G) = Δ :: rest` structure the π'-side of the graph assemble needs: `havelHakimiStep`
applied to `sort(deg G)` then decrements the top-`Δ` entries of `rest`. -/
theorem sortedDeg_cons {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (v : V) (hv : ∀ w, G.degree w ≤ G.degree v) :
    (Finset.univ.val.map (fun w => G.degree w)).sort (· ≥ ·)
      = G.degree v :: ((Finset.univ.erase v).val.map (fun w => G.degree w)).sort (· ≥ ·) := by
  have hcons : Finset.univ.val.map (fun w => G.degree w)
      = G.degree v ::ₘ (Finset.univ.erase v).val.map (fun w => G.degree w) := by
    conv_lhs => rw [← Multiset.cons_erase (Finset.mem_univ v)]
    rw [Multiset.map_cons, Finset.erase_val]
  have hle : ∀ b ∈ (Finset.univ.erase v).val.map (fun w => G.degree w), G.degree v ≥ b := by
    intro b hb
    rw [Multiset.mem_map] at hb
    obtain ⟨w, _, rfl⟩ := hb
    exact hv w
  rw [hcons]
  exact Multiset.sort_cons _ _ _ hle

end SimpleGraph
