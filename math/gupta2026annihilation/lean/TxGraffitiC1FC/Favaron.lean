import Mathlib
import TxGraffitiC1FC.Invariants
import TxGraffitiC1FC.ResidueMonotonicity

/-!
# Favaron's residue bound

Deleting a positive-maximum-degree vertex does not decrease the residue, by
the native proof in `ResidueMonotonicity`. Strong induction on the vertex count
then proves `R(G) ≤ α(G)`, using monotonicity of the independence number under
induced subgraphs. No custom axiom is used.
-/

namespace SimpleGraph

open Finset
open scoped Classical

set_option linter.unusedSectionVars false

variable {V : Type*} [Fintype V] [DecidableEq V]

/-! ## Sequence lemmas about `residueAux` (subtype-free) -/

/-- The residue of an all-zero list is its length (every entry is an isolated vertex).
This is the base case: an edgeless graph has all-zero degree sequence. -/
theorem residueAux_replicate_zero (n : ℕ) : residueAux (List.replicate n 0) = n := by
  cases n with
  | zero => simp [residueAux]
  | succ k =>
    rw [List.replicate_succ]
    simp only [residueAux, List.length_replicate]
    omega

/-- The residue never exceeds the list length (`R ≤ n`). -/
theorem residueAux_le_length (l : List ℕ) : residueAux l ≤ l.length := by
  induction l using residueAux.induct with
  | case1 => simp [residueAux]
  | case2 s =>
    simp only [residueAux, List.length_cons]
    omega
  | case3 d rest hd ih =>
    rw [residueAux]
    · calc residueAux (havelHakimiStep (d :: rest))
          ≤ (havelHakimiStep (d :: rest)).length := ih
        _ = rest.length := havelHakimiStep_length_cons d rest
        _ ≤ (d :: rest).length := by simp
    · exact hd

/-! ## Graph base case: an edgeless graph has `R = card V = α` -/

/-- The independence number never exceeds the number of vertices. -/
theorem indepNum_le_card (G : SimpleGraph V) : G.indepNum ≤ Fintype.card V := by
  obtain ⟨s, _, hcard⟩ := G.exists_isNIndepSet_indepNum
  rw [← hcard]
  exact Finset.card_le_univ s

/-- **α-bridge**: an independent set of an induced subgraph is an independent set of
the ambient graph, so `α(G[s]) ≤ α(G)`. (Not gated on the crux; closes the
`α(G−v) ≤ α(G)` link of the induction.) -/
theorem indepNum_induce_le (G : SimpleGraph V) (s : Set V) :
    (G.induce s).indepNum ≤ G.indepNum := by
  classical
  obtain ⟨t, htind, htcard⟩ := (G.induce s).exists_isNIndepSet_indepNum
  rw [← htcard, ← Finset.card_map (Function.Embedding.subtype (· ∈ s)) (s := t)]
  apply SimpleGraph.IsIndepSet.card_le_indepNum
  intro a ha b hb hne hadj
  simp only [Finset.coe_map, Set.mem_image, Finset.mem_coe,
    Function.Embedding.coe_subtype] at ha hb
  obtain ⟨a', ha't, rfl⟩ := ha
  obtain ⟨b', hb't, rfl⟩ := hb
  exact htind ha't hb't (fun h => hne (by rw [h])) hadj

/-- Residue of an edgeless graph (all degrees zero): every vertex is isolated, so
`R = card V`. Stated over `G` with its ambient instance to avoid `⊥`-instance clashes. -/
theorem residue_eq_card_of_degree_zero (G : SimpleGraph V) [DecidableRel G.Adj]
    (h : ∀ v, G.degree v = 0) : G.residue = Fintype.card V := by
  unfold residue
  have hsort :
      (Finset.univ.val.map fun v => G.degree v).sort (· ≥ ·)
        = List.replicate (Fintype.card V) 0 := by
    rw [List.eq_replicate_iff]
    refine ⟨?_, ?_⟩
    · rw [Multiset.length_sort, Multiset.card_map, Finset.card_val, Finset.card_univ]
    · intro b hb
      rw [Multiset.mem_sort, Multiset.mem_map] at hb
      obtain ⟨v, _, rfl⟩ := hb
      exact h v
  rw [hsort, residueAux_replicate_zero]

/-- Independence number of an edgeless graph (all degrees zero): the whole vertex set
is independent, so `α = card V`. -/
theorem indepNum_eq_card_of_degree_zero (G : SimpleGraph V) [DecidableRel G.Adj]
    (h : ∀ v, G.degree v = 0) : G.indepNum = Fintype.card V := by
  refine le_antisymm G.indepNum_le_card ?_
  have huniv : G.IsIndepSet (Finset.univ : Finset V) := by
    intro a _ b _ _ hab
    have hpos : 0 < G.degree a := (G.degree_pos_iff_exists_adj a).mpr ⟨b, hab⟩
    rw [h a] at hpos; exact absurd hpos (lt_irrefl 0)
  calc Fintype.card V = (Finset.univ : Finset V).card := (Finset.card_univ).symm
    _ ≤ G.indepNum := huniv.card_le_indepNum
/-! ## Residue monotonicity under maximum-degree vertex deletion -/

/-- Deleting a positive-maximum-degree vertex does not decrease the residue. -/
theorem residue_le_residue_induce_compl_of_maxDegree
    (G : SimpleGraph V) [DecidableRel G.Adj] {v : V}
    (hv : ∀ w, G.degree w ≤ G.degree v) (hpos : 0 < G.degree v) :
    G.residue ≤ (G.induce ({v}ᶜ : Set V)).residue :=
  residue_le_residue_induce_compl_of_maxDegree' G hv hpos

/-! ## Main theorem: `R ≤ α` -/

/-- Favaron's residue bound, proved by strong induction and residue monotonicity. -/
theorem residue_le_indepNum (G : SimpleGraph V) [DecidableRel G.Adj] :
    G.residue ≤ G.indepNum := by
  suffices h : ∀ (n : ℕ) (W : Type _) [Fintype W] [DecidableEq W]
      (H : SimpleGraph W) [DecidableRel H.Adj], Fintype.card W = n → H.residue ≤ H.indepNum by
    exact h (Fintype.card V) V G rfl
  intro n
  induction n using Nat.strong_induction_on with
  | _ n IH =>
    intro W _ _ H _ hcard
    rcases isEmpty_or_nonempty W with hem | hne
    · have hz : ∀ w, H.degree w = 0 := fun w => (hem.false w).elim
      rw [residue_eq_card_of_degree_zero H hz]
      exact (indepNum_eq_card_of_degree_zero H hz).ge
    · obtain ⟨v, hv⟩ := Finite.exists_max (fun w => H.degree w)
      by_cases hdeg : H.degree v = 0
      · have hz : ∀ w, H.degree w = 0 := fun w => Nat.le_zero.mp (hdeg ▸ hv w)
        rw [residue_eq_card_of_degree_zero H hz]
        exact (indepNum_eq_card_of_degree_zero H hz).ge
      · have hcard' : Fintype.card ↥({v}ᶜ : Set W) = n - 1 := by
          rw [← hcard, ← Set.toFinset_card, Set.toFinset_compl, Set.toFinset_singleton,
            Finset.card_compl, Finset.card_singleton]
        have hlt : n - 1 < n := by
          have hpos : 0 < Fintype.card W := Fintype.card_pos
          omega
        have hIH := IH (n - 1) hlt ↥({v}ᶜ : Set W) (H.induce _) hcard'
        calc H.residue
            ≤ (H.induce ({v}ᶜ : Set W)).residue :=
              H.residue_le_residue_induce_compl_of_maxDegree hv (Nat.pos_of_ne_zero hdeg)
          _ ≤ (H.induce ({v}ᶜ : Set W)).indepNum := hIH
          _ ≤ H.indepNum := H.indepNum_induce_le _

/-- Rational-valued form, for the `Conjecture.lean` assembly. -/
theorem residue_le_indepNum_rat (G : SimpleGraph V) [DecidableRel G.Adj] :
    (G.residue : ℚ) ≤ (G.indepNum : ℚ) := by
  exact_mod_cast G.residue_le_indepNum

end SimpleGraph
