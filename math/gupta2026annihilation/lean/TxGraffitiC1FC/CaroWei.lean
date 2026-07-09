import Mathlib
import TxGraffitiC1FC.Invariants

/-!
# Item 5 — Caro–Wei bound `W ≤ α` via a subtype-free (Finset-relative) Wei induction

Formalizes `∑_v 1/(deg v + 1) ≤ α` (Caro–Wei; not in Mathlib). To avoid
induced-subgraph-over-subtype bookkeeping, everything is relativized to a
`Finset V` `s`: `degIn s v` = number of `s`-neighbours of `v`, `alphaIn s` = max
card of an independent subset of `s`. Wei's induction (delete a min-degree closed
neighbourhood) runs over Finsets of `V`; the `s = univ` case is Caro–Wei.
-/

open Finset
open scoped Classical

set_option linter.unusedSectionVars false

namespace SimpleGraph

variable {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
  [DecidableRel G.Adj]

/-- Degree of `v` within `s`: the number of `s`-vertices adjacent to `v`. -/
def degIn (s : Finset V) (v : V) : ℕ := (s.filter (fun w => G.Adj v w)).card

/-- Independence number within `s`: the largest cardinality of an independent
subset of `s`. -/
noncomputable def alphaIn (s : Finset V) : ℕ :=
  s.powerset.sup (fun I => if G.IsIndepSet (I : Set V) then I.card else 0)

/-- Any independent subset of `s` has card at most `alphaIn s`. -/
theorem card_le_alphaIn {s I : Finset V} (hIs : I ⊆ s)
    (hind : G.IsIndepSet (I : Set V)) : I.card ≤ G.alphaIn s := by
  have h := Finset.le_sup (s := s.powerset)
    (f := fun J : Finset V => if G.IsIndepSet (J : Set V) then J.card else 0)
    (Finset.mem_powerset.mpr hIs)
  simp only [if_pos hind] at h
  exact h

/-- `alphaIn s` is attained by some independent subset of `s`. -/
theorem exists_alphaIn (s : Finset V) :
    ∃ I ⊆ s, G.IsIndepSet (I : Set V) ∧ I.card = G.alphaIn s := by
  obtain ⟨I, hI, hsup⟩ := Finset.exists_mem_eq_sup s.powerset
    ⟨∅, Finset.empty_mem_powerset s⟩
    (fun J => if G.IsIndepSet (J : Set V) then J.card else 0)
  simp only [alphaIn]
  by_cases hind : G.IsIndepSet (I : Set V)
  · exact ⟨I, Finset.mem_powerset.mp hI, hind, by rw [hsup, if_pos hind]⟩
  · exact ⟨∅, Finset.empty_subset _, by simp, by rw [hsup, if_neg hind]; simp⟩

/-- Degree within a set is monotone in the set. -/
theorem degIn_mono {s t : Finset V} (h : s ⊆ t) (v : V) :
    G.degIn s v ≤ G.degIn t v :=
  Finset.card_le_card (Finset.filter_subset_filter _ h)

/-- Within all of `V`, `degIn` is the graph degree. -/
theorem degIn_univ (v : V) : G.degIn Finset.univ v = G.degree v := by
  simp only [degIn, SimpleGraph.degree, SimpleGraph.neighborFinset_eq_filter]

/-- Within all of `V`, `alphaIn` is the independence number. -/
theorem alphaIn_univ : G.alphaIn Finset.univ = G.indepNum := by
  apply le_antisymm
  · obtain ⟨I, _, hind, hcard⟩ := G.exists_alphaIn Finset.univ
    rw [← hcard]
    exact hind.card_le_indepNum
  · obtain ⟨t, hind, hcard⟩ := G.exists_isNIndepSet_indepNum
    rw [← hcard]
    exact G.card_le_alphaIn (Finset.subset_univ _) hind

/-- **Caro–Wei, Finset-relative** (Wei's deletion induction): for every `s`,
`∑_{v ∈ s} 1/(degIn s v + 1) ≤ alphaIn s`. -/
theorem caroWei_finset (s : Finset V) :
    (∑ v ∈ s, 1 / ((G.degIn s v : ℚ) + 1)) ≤ (G.alphaIn s : ℚ) := by
  induction s using Finset.strongInduction with
  | _ s ih =>
    rcases s.eq_empty_or_nonempty with rfl | hne
    · simp
    · obtain ⟨v, hvs, hvmin⟩ := s.exists_min_image (G.degIn s) hne
      set Nnb := s.filter (fun w => G.Adj v w) with hNnb
      set N := insert v Nnb with hN
      set s' := s \ N with hs'def
      have hvNnb : v ∉ Nnb := by
        rw [hNnb, Finset.mem_filter]; exact fun h => G.irrefl h.2
      have hNsub : N ⊆ s :=
        Finset.insert_subset hvs (Finset.filter_subset _ _)
      have hs'ss : s' ⊂ s :=
        Finset.sdiff_ssubset hNsub ⟨v, Finset.mem_insert_self _ _⟩
      have hIH := ih s' hs'ss
      obtain ⟨I', hI'sub, hI'ind, hI'card⟩ := G.exists_alphaIn s'
      have hvI' : v ∉ I' := by
        intro hv
        have hmem := hI'sub hv
        rw [hs'def, Finset.mem_sdiff] at hmem
        exact hmem.2 (Finset.mem_insert_self _ _)
      have hvadj : ∀ u ∈ I', ¬ G.Adj v u := by
        intro u hu hadj
        have hmem := hI'sub hu
        rw [hs'def, Finset.mem_sdiff] at hmem
        apply hmem.2
        rw [hN, Finset.mem_insert]; right
        rw [hNnb, Finset.mem_filter]; exact ⟨hmem.1, hadj⟩
      have hins_ind : G.IsIndepSet ((insert v I' : Finset V) : Set V) := by
        rw [isIndepSet_iff, Finset.coe_insert]
        refine (Iff.mp G.isIndepSet_iff hI'ind).insert ?_
        intro u hu _
        have hnadj : ¬ G.Adj v u := hvadj u (Finset.mem_coe.mp hu)
        exact ⟨hnadj, fun huv => hnadj huv.symm⟩
      have hins_sub : (insert v I' : Finset V) ⊆ s :=
        Finset.insert_subset hvs (hI'sub.trans Finset.sdiff_subset)
      have hext : (I'.card : ℚ) + 1 ≤ (G.alphaIn s : ℚ) := by
        have hle := G.card_le_alphaIn hins_sub hins_ind
        rw [Finset.card_insert_of_notMem hvI'] at hle
        exact_mod_cast hle
      have hsplit : (∑ v' ∈ s, 1 / ((G.degIn s v' : ℚ) + 1))
          = (∑ v' ∈ s', 1 / ((G.degIn s v' : ℚ) + 1))
            + (∑ v' ∈ N, 1 / ((G.degIn s v' : ℚ) + 1)) :=
        (Finset.sum_sdiff hNsub).symm
      have hNnbcard : Nnb.card = G.degIn s v := by simp only [degIn, hNnb]
      have hNcard : N.card = G.degIn s v + 1 := by
        rw [hN, Finset.card_insert_of_notMem hvNnb, hNnbcard]
      have hNpart : (∑ v' ∈ N, 1 / ((G.degIn s v' : ℚ) + 1)) ≤ 1 := by
        have hb : ∀ u ∈ N, 1 / ((G.degIn s u : ℚ) + 1)
            ≤ 1 / ((G.degIn s v : ℚ) + 1) := by
          intro u hu
          apply one_div_le_one_div_of_le (by positivity)
          exact_mod_cast Nat.add_le_add_right (hvmin u (hNsub hu)) 1
        have h0 : (G.degIn s v : ℚ) + 1 ≠ 0 := by positivity
        calc (∑ v' ∈ N, 1 / ((G.degIn s v' : ℚ) + 1))
            ≤ ∑ _u ∈ N, 1 / ((G.degIn s v : ℚ) + 1) := Finset.sum_le_sum hb
          _ = (N.card : ℚ) / ((G.degIn s v : ℚ) + 1) := by
                rw [Finset.sum_const, nsmul_eq_mul]; ring
          _ = 1 := by rw [hNcard]; push_cast; rw [div_self h0]
      have hs'part : (∑ v' ∈ s', 1 / ((G.degIn s v' : ℚ) + 1))
          ≤ (G.alphaIn s' : ℚ) := by
        refine le_trans (Finset.sum_le_sum ?_) hIH
        intro u _
        apply one_div_le_one_div_of_le (by positivity)
        exact_mod_cast Nat.add_le_add_right (G.degIn_mono Finset.sdiff_subset u) 1
      have hcardQ : (I'.card : ℚ) = (G.alphaIn s' : ℚ) := by exact_mod_cast hI'card
      rw [hsplit]
      linarith [hNpart, hs'part, hext, hcardQ]

/-- **Caro–Wei bound** (`W ≤ α`): `∑_v 1/(deg v + 1) ≤ indepNum`. Discharges the
C1 hole `caroWei_le_indepNum` (modulo the `caroWeiWeight` unfolding, item 7). -/
theorem caroWei_bound :
    (∑ v : V, 1 / ((G.degree v : ℚ) + 1)) ≤ (G.indepNum : ℚ) := by
  have h := G.caroWei_finset Finset.univ
  simp only [degIn_univ, alphaIn_univ] at h
  exact h

end SimpleGraph
