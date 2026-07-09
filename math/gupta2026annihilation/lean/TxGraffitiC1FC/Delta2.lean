import Mathlib
import TxGraffitiC1FC.Invariants
import TxGraffitiC1FC.Vehicle

/-!
# Item 3 — the Δ≤2 branch: connected `Δ ≤ 2 ⟹ a ≤ α`

For a connected graph with maximum degree `≤ 2` (a path or a cycle) the
annihilation number `a` equals the independence number `α`; in particular
`a ≤ α`. This file builds the bound from the ground up (no Mathlib support:
Mathlib has no path/cycle independence, no Δ≤2 classification, no annihilation
lemmas). Structural facts first, then the two bounds.
-/

open Finset
open scoped Classical

set_option linter.unusedSectionVars false

namespace SimpleGraph

variable {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
  [DecidableRel G.Adj]

/-- Every degree is `≤ 2` when `Δ ≤ 2`. -/
theorem degree_le_two (hΔ : G.maxDegree ≤ 2) (v : V) : G.degree v ≤ 2 :=
  le_trans (G.degree_le_maxDegree v) hΔ

/-- For `Δ ≤ 2`, the edge count is at most the vertex count (`2m = Σ deg ≤ 2n`). -/
theorem card_edges_le_of_maxDeg_le_two (hΔ : G.maxDegree ≤ 2) :
    G.edgeFinset.card ≤ Fintype.card V := by
  have h2 : 2 * G.edgeFinset.card = ∑ v, G.degree v :=
    (G.sum_degrees_eq_twice_card_edges).symm
  have hb : ∑ v, G.degree v ≤ ∑ _v : V, 2 :=
    Finset.sum_le_sum (fun v _ => G.degree_le_two hΔ v)
  rw [Finset.sum_const, Finset.card_univ, smul_eq_mul] at hb
  omega

/-- **a-side** (uses only `Δ ≤ 2`, via item 2's `exists_annih_head`):
`2·a ≤ 2n − m`. The annihilation head-set `H` (`|H| = a`, `Σ_H deg ≤ m`) has
complement degree-sum `2m − Σ_H deg ≥ m` and `≤ 2(n − a)`, so `m ≤ 2(n − a)`. -/
theorem two_mul_annihilationNumber_le (hΔ : G.maxDegree ≤ 2) :
    2 * G.annihilationNumber ≤ 2 * Fintype.card V - G.edgeFinset.card := by
  obtain ⟨H, hHcard, hHsum⟩ := G.exists_annih_head
  have han : H.card ≤ Fintype.card V := by
    rw [← Finset.card_univ]; exact Finset.card_le_card (H.subset_univ)
  have hsplit : (∑ v ∈ H, G.degree v) + (∑ v ∈ Finset.univ \ H, G.degree v)
      = 2 * G.edgeFinset.card := by
    have key := Finset.sum_sdiff (H.subset_univ) (f := fun v => G.degree v)
    rw [G.sum_degrees_eq_twice_card_edges] at key
    omega
  have htail : (∑ v ∈ Finset.univ \ H, G.degree v)
      ≤ 2 * (Fintype.card V - H.card) := by
    calc (∑ v ∈ Finset.univ \ H, G.degree v)
        ≤ ∑ _v ∈ Finset.univ \ H, 2 :=
          Finset.sum_le_sum (fun v _ => G.degree_le_two hΔ v)
      _ = (Finset.univ \ H).card * 2 := by rw [Finset.sum_const, smul_eq_mul]
      _ = 2 * (Fintype.card V - H.card) := by
            rw [← Finset.compl_eq_univ_sdiff, Finset.card_compl]; ring
  rw [← hHcard]
  omega

/-- **α-side, 2-colorable case**: if `G` is 2-colorable then `n ≤ 2·α`. The two
colour classes are independent sets covering all `n` vertices, so each has card
`≤ α` and together `= n`. (Covers the acyclic/tree case of Δ≤2.) -/
theorem card_le_two_mul_indepNum_of_colorable_two (hcol : G.Colorable 2) :
    Fintype.card V ≤ 2 * G.indepNum := by
  obtain ⟨c⟩ := hcol
  have hind : ∀ i : Fin 2,
      (Finset.univ.filter (fun v => c v = i)).card ≤ G.indepNum := by
    intro i
    apply SimpleGraph.IsIndepSet.card_le_indepNum
    intro u hu w hw _ hadj
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hu hw
    exact c.valid hadj (hu.trans hw.symm)
  have hpart : (Finset.univ.filter (fun v => c v = 0)).card
      + (Finset.univ.filter (fun v => c v = 1)).card = Fintype.card V := by
    have hcongr : (Finset.univ.filter (fun v => c v = 1))
        = Finset.univ.filter (fun v => ¬ (c v = 0)) := by
      apply Finset.filter_congr
      intro v _
      rw [Fin.ext_iff, Fin.ext_iff]
      have := (c v).isLt
      omega
    rw [hcongr, Finset.card_filter_add_card_filter_not, Finset.card_univ]
  have h0 := hind 0
  have h1 := hind 1
  omega

/-- **Item 3 main bound**: connected `Δ ≤ 2 ⟹ a ≤ α`. Combines the a-side
(`2a ≤ 2n − m`), connectivity (`m ≥ n − 1`), and the α-side (`n ≤ 2α` when
2-colourable; the odd-cycle case handled separately). -/
theorem annih_le_indepNum_of_maxDegree_le_two
    (hconn : G.Connected) (hΔ : G.maxDegree ≤ 2) :
    (G.annihilationNumber : ℚ) ≤ (G.indepNum : ℚ) := by
  suffices h : G.annihilationNumber ≤ G.indepNum by exact_mod_cast h
  have hA := G.two_mul_annihilationNumber_le hΔ
  have hmle := G.card_edges_le_of_maxDeg_le_two hΔ
  obtain ⟨T, hTle, hTtree⟩ := hconn.exists_isTree_le
  have hmge : Fintype.card V - 1 ≤ G.edgeFinset.card := by
    have h1 : T.edgeFinset.card ≤ G.edgeFinset.card :=
      Finset.card_le_card (edgeFinset_mono hTle)
    have h2 : T.edgeFinset.card + 1 = Fintype.card V := hTtree.card_edgeFinset
    omega
  by_cases hcol : G.Colorable 2
  · have hα := G.card_le_two_mul_indepNum_of_colorable_two hcol
    omega
  · -- Odd-cycle case: `¬Colorable 2 ⟹ m = n`, and the spanning tree's 2-colouring
    -- has a single monochromatic edge, so the two colour classes (minus one shared
    -- endpoint) are independent and give `2α ≥ n − 1`.
    have hmn : G.edgeFinset.card = Fintype.card V := by
      by_contra hne
      have hmeq : G.edgeFinset.card + 1 = Fintype.card V := by omega
      have htree : G.IsTree := by
        rw [isTree_iff_connected_and_card]
        refine ⟨hconn, ?_⟩
        rw [Nat.card_eq_fintype_card, Nat.card_eq_fintype_card, ← edgeFinset_card]
        exact hmeq
      exact hcol htree.colorable_two
    have hTsub : T.edgeFinset ⊆ G.edgeFinset := edgeFinset_mono hTle
    have hTcard : T.edgeFinset.card + 1 = Fintype.card V := hTtree.card_edgeFinset
    have hdiff : (G.edgeFinset \ T.edgeFinset).card = 1 := by
      rw [Finset.card_sdiff, Finset.inter_eq_left.mpr hTsub]; omega
    obtain ⟨e, he⟩ := Finset.card_eq_one.mp hdiff
    obtain ⟨x, y, rfl⟩ : ∃ x y, e = s(x, y) := Sym2.ind (fun x y => ⟨x, y, rfl⟩) e
    have c := hTtree.coloringTwo
    -- Every `G`-edge that is not a `T`-edge must be the unique extra edge `s(x,y)`.
    have key : ∀ a b, G.Adj a b → ¬ T.Adj a b → s(a, b) = s(x, y) := by
      intro a b hab hnab
      have hmem : s(a, b) ∈ G.edgeFinset \ T.edgeFinset := by
        rw [Finset.mem_sdiff]
        refine ⟨?_, ?_⟩
        · rw [mem_edgeFinset, mem_edgeSet]; exact hab
        · rw [mem_edgeFinset, mem_edgeSet]; exact hnab
      rw [he, Finset.mem_singleton] at hmem
      exact hmem
    -- The extra edge is monochromatic (else `c` would properly colour `G`).
    have hcxy : c x = c y := by
      by_contra hne
      refine hcol ⟨SimpleGraph.Coloring.mk (fun v => c v) ?_⟩
      intro a b hab
      by_cases hT : T.Adj a b
      · exact c.valid hT
      · have hs := key a b hab hT
        rw [Sym2.eq_iff] at hs
        rcases hs with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
        · exact hne
        · exact fun h => hne h.symm
    -- Colour classes: `A` contains `x` (and `y`); `B` is the rest.
    set A := Finset.univ.filter (fun v => c v = c x) with hA_def
    set B := Finset.univ.filter (fun v => ¬ c v = c x) with hB_def
    have hxA : x ∈ A := by rw [hA_def]; simp
    have hfin2 : ∀ p q r : Fin 2, p ≠ r → q ≠ r → p = q := by decide
    -- `A.erase x` is independent: any internal `G`-edge is the mono edge `{x,y}`.
    have hAind : G.IsIndepSet (↑(A.erase x) : Set V) := by
      intro a ha b hb _ hadj
      rw [Finset.coe_erase] at ha hb
      simp only [Set.mem_sdiff, Finset.mem_coe, hA_def, Finset.mem_filter, Finset.mem_univ,
        true_and, Set.mem_singleton_iff] at ha hb
      obtain ⟨hca, hax⟩ := ha
      obtain ⟨hcb, hbx⟩ := hb
      have hnT : ¬ T.Adj a b := fun ht => c.valid ht (hca.trans hcb.symm)
      have hs := key a b hadj hnT
      rw [Sym2.eq_iff] at hs
      rcases hs with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · exact hax rfl
      · exact hbx rfl
    -- `B` is independent: two `B`-vertices share the other colour, again forcing `{x,y}`.
    have hBind : G.IsIndepSet (↑B : Set V) := by
      intro a ha b hb _ hadj
      simp only [Finset.mem_coe, hB_def, Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
      have hcab : c a = c b := hfin2 (c a) (c b) (c x) ha hb
      have hnT : ¬ T.Adj a b := fun ht => c.valid ht hcab
      have hs := key a b hadj hnT
      rw [Sym2.eq_iff] at hs
      rcases hs with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · exact ha rfl
      · exact hb rfl
    -- `|A.erase x| + |B| = n − 1`, and each is `≤ α`, so `n − 1 ≤ 2α`.
    have hAB : A.card + B.card = Fintype.card V := by
      rw [hA_def, hB_def, ← Finset.card_univ]
      exact Finset.card_filter_add_card_filter_not (fun v => c v = c x)
    have hApos : 0 < A.card := Finset.card_pos.mpr ⟨x, hxA⟩
    have hcard : (A.erase x).card + B.card = Fintype.card V - 1 := by
      rw [Finset.card_erase_of_mem hxA]; omega
    have hAle : (A.erase x).card ≤ G.indepNum := hAind.card_le_indepNum
    have hBle : B.card ≤ G.indepNum := hBind.card_le_indepNum
    omega

end SimpleGraph
