import TxGraffitiC1FC.Conjecture
import TxGraffitiC1FC.Corollaries

set_option linter.unusedSectionVars false

namespace SimpleGraph

open Finset
open scoped Classical

variable {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V) [DecidableRel G.Adj]

/-- Every nonempty finite graph has positive independence number (`1 ≤ α(G)`). -/
theorem indepNum_pos [Nonempty V] : 0 < G.indepNum := by
  obtain ⟨v⟩ := ‹Nonempty V›
  have hind : G.IsIndepSet ({v} : Finset V) := by simp
  have hle := hind.card_le_indepNum
  simp only [Finset.card_singleton] at hle
  omega

/-- Invariant sharpness criterion at the graph level: for connected `G` with `Δ ≥ 2`,
`Δ · α = a + R ↔ ((Δ - 1) · α = a ∧ α = R)`. -/
theorem txgraffiti_eq_iff_invariants (hconn : G.Connected) (hΔ : 2 ≤ G.maxDegree) :
    (G.maxDegree : ℚ) * (G.indepNum : ℚ) = (G.annihilationNumber : ℚ) + (G.residue : ℚ) ↔
      (((G.maxDegree : ℚ) - 1) * (G.indepNum : ℚ) = (G.annihilationNumber : ℚ) ∧
       (G.indepNum : ℚ) = (G.residue : ℚ)) :=
  TxGraffitiC1FC.sharpness_iff _ _ _ _
    (G.annih_le_predMaxDegree_indepNum hconn hΔ)
    G.residue_le_indepNum_rat

/-- If `G` has `Δ ≥ 3` and `(Δ - 1) · α = a`, then `Δ = 3` and `W(G) = α(G)`. -/
theorem maxDegree_eq_three_and_caroWei_eq_of_annih_eq
    [Nonempty V] (hΔ3 : 3 ≤ G.maxDegree)
    (ha : ((G.maxDegree : ℚ) - 1) * (G.indepNum : ℚ) = (G.annihilationNumber : ℚ)) :
    G.maxDegree = 3 ∧ G.caroWeiWeight = (G.indepNum : ℚ) := by
  have hv := G.annih_vehicle (by omega)
  have hcw := G.caroWei_le_indepNum
  have hΔQ : (3 : ℚ) ≤ (G.maxDegree : ℚ) := by exact_mod_cast hΔ3
  have hαpos : (0 : ℚ) < (G.indepNum : ℚ) := by exact_mod_cast G.indepNum_pos
  have hhalf_pos : (0 : ℚ) < ((G.maxDegree : ℚ) + 1) / 2 := by linarith
  have hratio : ((G.maxDegree : ℚ) + 1) / 2 ≤ (G.maxDegree : ℚ) - 1 := by linarith
  have hmid : ((G.maxDegree : ℚ) + 1) / 2 * G.caroWeiWeight
      ≤ ((G.maxDegree : ℚ) + 1) / 2 * (G.indepNum : ℚ) :=
    mul_le_mul_of_nonneg_left hcw hhalf_pos.le
  have htop : ((G.maxDegree : ℚ) + 1) / 2 * (G.indepNum : ℚ)
      ≤ ((G.maxDegree : ℚ) - 1) * (G.indepNum : ℚ) :=
    mul_le_mul_of_nonneg_right hratio hαpos.le
  have htight_top : ((G.maxDegree : ℚ) + 1) / 2 * (G.indepNum : ℚ)
      = ((G.maxDegree : ℚ) - 1) * (G.indepNum : ℚ) := by linarith
  have htight_mid : ((G.maxDegree : ℚ) + 1) / 2 * G.caroWeiWeight
      = ((G.maxDegree : ℚ) + 1) / 2 * (G.indepNum : ℚ) := by linarith
  have hΔQ_eq : (G.maxDegree : ℚ) = 3 := by nlinarith
  have hcw_eq : G.caroWeiWeight = (G.indepNum : ℚ) :=
    mul_left_cancel₀ hhalf_pos.ne' htight_mid
  exact ⟨by exact_mod_cast hΔQ_eq, hcw_eq⟩

/-- **Strictness for `Δ ≥ 4` (no connected graph with `Δ ≥ 4` attains equality).** -/
theorem strict_of_maxDegree_ge_four (hconn : G.Connected) (hΔ4 : 4 ≤ G.maxDegree) :
    (G.annihilationNumber : ℚ) + (G.residue : ℚ) < (G.maxDegree : ℚ) * (G.indepNum : ℚ) := by
  haveI : Nonempty V := hconn.nonempty
  have hle := G.txgraffiti_conjecture_1 hconn (by omega)
  refine lt_of_le_of_ne hle ?_
  intro heq
  have ⟨ha, _⟩ := (G.txgraffiti_eq_iff_invariants hconn (by omega)).mp heq.symm
  have ⟨hΔ3, _⟩ := G.maxDegree_eq_three_and_caroWei_eq_of_annih_eq (by omega) ha
  omega

/-- **Caro–Wei equality forces a connected graph to be a complete graph.**
If `G` is connected and `W(G) = α(G)`, let `v` be a vertex of minimum degree `δ`.
Then `N[v]` has `δ + 1` vertices each of degree `≥ δ`, and deleting `N[v]` leaves `s'`
with `W(G) ≤ 1 + W(G[s']) ≤ 1 + α(G[s']) ≤ α(G)`. Tightness forces every `u ∈ N[v]`
to have degree `δ` and every `u ∈ s'` to have `degIn s' u = deg u`, meaning no edge
connects `N[v]` to `s'`. By connectedness `s' = ∅`, so `univ = N[v]`, and every vertex
has degree `|V| - 1`, making `G = ⊤`. -/
theorem eq_top_of_connected_of_caroWei_eq (hconn : G.Connected)
    (hcw : G.caroWeiWeight = (G.indepNum : ℚ)) :
    G = ⊤ := by
  haveI : Nonempty V := hconn.nonempty
  obtain ⟨v, _, hvmin⟩ := (Finset.univ : Finset V).exists_min_image
    (G.degIn Finset.univ) Finset.univ_nonempty
  set Nnb := (Finset.univ : Finset V).filter (fun w => G.Adj v w) with hNnb
  set N := insert v Nnb with hN
  set s' := (Finset.univ : Finset V) \ N with hs'def
  have hvNnb : v ∉ Nnb := by
    rw [hNnb, Finset.mem_filter]; exact fun h => G.irrefl h.2
  have hNsub : N ⊆ Finset.univ := Finset.subset_univ N
  have hNnbcard : Nnb.card = G.degIn Finset.univ v := by simp only [degIn, hNnb]
  have hNcard : N.card = G.degIn Finset.univ v + 1 := by
    rw [hN, Finset.card_insert_of_notMem hvNnb, hNnbcard]
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
  have hext : (G.alphaIn s' : ℚ) + 1 ≤ (G.indepNum : ℚ) := by
    have hle := G.card_le_alphaIn (Finset.subset_univ (insert v I')) hins_ind
    rw [Finset.card_insert_of_notMem hvI', hI'card, alphaIn_univ] at hle
    exact_mod_cast hle
  have hsplit : G.caroWeiWeight
      = (∑ v' ∈ s', 1 / ((G.degIn Finset.univ v' : ℚ) + 1))
        + (∑ v' ∈ N, 1 / ((G.degIn Finset.univ v' : ℚ) + 1)) := by
    unfold caroWeiWeight
    simp_rw [← degIn_univ]
    exact (Finset.sum_sdiff hNsub).symm
  have hbN : ∀ u ∈ N, 1 / ((G.degIn Finset.univ u : ℚ) + 1)
      ≤ 1 / ((G.degIn Finset.univ v : ℚ) + 1) := by
    intro u _
    apply one_div_le_one_div_of_le (by positivity)
    exact_mod_cast Nat.add_le_add_right (hvmin u (Finset.mem_univ u)) 1
  have hNconst : (∑ _u ∈ N, 1 / ((G.degIn Finset.univ v : ℚ) + 1)) = 1 := by
    have h0 : (G.degIn Finset.univ v : ℚ) + 1 ≠ 0 := by positivity
    calc (∑ _u ∈ N, 1 / ((G.degIn Finset.univ v : ℚ) + 1))
        = (N.card : ℚ) / ((G.degIn Finset.univ v : ℚ) + 1) := by
            rw [Finset.sum_const, nsmul_eq_mul]; ring
      _ = 1 := by rw [hNcard]; push_cast; rw [div_self h0]
  have hNpart : (∑ v' ∈ N, 1 / ((G.degIn Finset.univ v' : ℚ) + 1)) ≤ 1 := by
    exact (Finset.sum_le_sum hbN).trans_eq hNconst
  have hbs' : ∀ u ∈ s', 1 / ((G.degIn Finset.univ u : ℚ) + 1)
      ≤ 1 / ((G.degIn s' u : ℚ) + 1) := by
    intro u _
    apply one_div_le_one_div_of_le (by positivity)
    exact_mod_cast Nat.add_le_add_right (G.degIn_mono (Finset.subset_univ s') u) 1
  have hs'le : (∑ v' ∈ s', 1 / ((G.degIn Finset.univ v' : ℚ) + 1))
      ≤ (∑ v' ∈ s', 1 / ((G.degIn s' v' : ℚ) + 1)) := Finset.sum_le_sum hbs'
  have hs'cw := G.caroWei_finset s'
  -- All inequalities in the Caro–Wei step must be tight:
  have hNsumeq : (∑ v' ∈ N, 1 / ((G.degIn Finset.univ v' : ℚ) + 1))
      = ∑ _u ∈ N, 1 / ((G.degIn Finset.univ v : ℚ) + 1) := by
    linarith [hcw, hsplit, hNpart, hNconst, hs'le, hs'cw, hext]
  have hs'sumeq : (∑ v' ∈ s', 1 / ((G.degIn Finset.univ v' : ℚ) + 1))
      = (∑ v' ∈ s', 1 / ((G.degIn s' v' : ℚ) + 1)) := by
    linarith [hcw, hsplit, hNpart, hNconst, hs'le, hs'cw, hext]
  -- Pointwise equality on s': every u ∈ s' has degIn univ u = degIn s' u
  have hs'ptw := (Finset.sum_eq_sum_iff_of_le hbs').mp hs'sumeq
  -- Pointwise equality on N: every u ∈ N has degIn univ u = degIn univ v
  have hNptw := (Finset.sum_eq_sum_iff_of_le hbN).mp hNsumeq
  -- First show s' = ∅: if w ∈ s', take a walk from v to w; boundary dart crosses N to s'
  have hs'empty : s' = ∅ := by
    rw [Finset.eq_empty_iff_forall_notMem]
    intro w hw
    obtain ⟨p⟩ := hconn v w
    have hvS : v ∈ (N : Set V) := Finset.mem_coe.mpr (Finset.mem_insert_self v Nnb)
    have hwS : w ∉ (N : Set V) := by
      rw [hs'def, Finset.mem_sdiff] at hw
      exact fun h => hw.2 (Finset.mem_coe.mp h)
    obtain ⟨d, _, hdfst, hdsnd⟩ := p.exists_boundary_dart (N : Set V) hvS hwS
    have hdsnd_s' : d.snd ∈ s' := by
      rw [hs'def, Finset.mem_sdiff]
      exact ⟨Finset.mem_univ _, fun h => hdsnd (Finset.mem_coe.mpr h)⟩
    have heq_rec := hs'ptw d.snd hdsnd_s'
    have hdegeq : G.degIn Finset.univ d.snd = G.degIn s' d.snd := by
      have hpos1 : (G.degIn Finset.univ d.snd : ℚ) + 1 ≠ 0 := by positivity
      have hpos2 : (G.degIn s' d.snd : ℚ) + 1 ≠ 0 := by positivity
      have hmul := (div_eq_div_iff hpos1 hpos2).mp heq_rec
      have hdegeqQ : (G.degIn Finset.univ d.snd : ℚ) + 1 = (G.degIn s' d.snd : ℚ) + 1 := by
        linarith
      exact_mod_cast add_right_cancel hdegeqQ
    -- But d.fst ∈ N is adjacent to d.snd, so d.fst ∈ univ.filter (G.Adj d.snd) \ s'.filter (G.Adj d.snd)
    have hsub_strict : s'.filter (fun z => G.Adj d.snd z) ⊂ Finset.univ.filter (fun z => G.Adj d.snd z) := by
      rw [Finset.ssubset_iff_of_subset (Finset.filter_subset_filter _ (Finset.subset_univ s'))]
      refine ⟨d.fst, ?_, ?_⟩
      · rw [Finset.mem_filter]; exact ⟨Finset.mem_univ _, d.adj.symm⟩
      · rw [Finset.mem_filter]; rintro ⟨hfst_s', _⟩
        rw [hs'def, Finset.mem_sdiff] at hfst_s'
        exact hfst_s'.2 (Finset.mem_coe.mp hdfst)
    have hlt_card := Finset.card_lt_card hsub_strict
    change G.degIn s' d.snd < G.degIn Finset.univ d.snd at hlt_card
    omega
  -- Since s' = univ \ N = ∅, every vertex is in N
  have hunivN : (Finset.univ : Finset V) = N := by
    ext u
    simp only [Finset.mem_univ, true_iff]
    by_contra huN
    have hus' : u ∈ s' := Finset.mem_sdiff.mpr ⟨Finset.mem_univ u, huN⟩
    rw [hs'empty] at hus'
    exact Finset.notMem_empty u hus'
  -- Every vertex u ∈ N has degree = degIn univ v = N.card - 1 = card V - 1
  have hcardV : Fintype.card V = G.degIn Finset.univ v + 1 := by
    rw [← Finset.card_univ, ← hNcard, hunivN]
  ext x y
  simp only [top_adj]
  refine ⟨fun h => h.ne, fun hne => ?_⟩
  have hxN : x ∈ N := by rw [← hunivN]; exact Finset.mem_univ x
  have heq_x := hNptw x hxN
  have hdegx : G.degree x = Fintype.card V - 1 := by
    have hpos1 : (G.degIn Finset.univ x : ℚ) + 1 ≠ 0 := by positivity
    have hpos2 : (G.degIn Finset.univ v : ℚ) + 1 ≠ 0 := by positivity
    have hmul := (div_eq_div_iff hpos1 hpos2).mp heq_x
    have hdegeqQ : (G.degIn Finset.univ x : ℚ) + 1 = (G.degIn Finset.univ v : ℚ) + 1 := by
      linarith
    have hx0 : G.degIn Finset.univ x = G.degIn Finset.univ v := by
      exact_mod_cast add_right_cancel hdegeqQ
    rw [← degIn_univ, hx0, hcardV]; omega
  -- Since (G.neighborFinset x).card = card (univ.erase x) and neighborFinset x ⊆ univ.erase x, they are equal!
  have hsub_erase : G.neighborFinset x ⊆ Finset.univ.erase x := by
    intro z hz
    rw [Finset.mem_erase]
    exact ⟨(G.ne_of_adj ((mem_neighborFinset G x z).mp hz)).symm, Finset.mem_univ z⟩
  have hcard_eq : (G.neighborFinset x).card = (Finset.univ.erase x).card := by
    rw [card_neighborFinset_eq_degree, hdegx, Finset.card_erase_of_mem (Finset.mem_univ x),
      Finset.card_univ]
  have heq_finset := Finset.eq_of_subset_of_card_le hsub_erase hcard_eq.ge
  have hy_mem : y ∈ G.neighborFinset x := by
    rw [heq_finset, Finset.mem_erase]
    exact ⟨hne.symm, Finset.mem_univ y⟩
  exact (mem_neighborFinset G x y).mp hy_mem

/-- **Complete classification for `Δ ≥ 3`**: if `G` is connected with `Δ ≥ 3` and attains
TxGraffiti equality, then `G ≃g completeGraph (Fin 4)`. -/
theorem iso_completeGraph_four_of_maxDegree_ge_three
    (hconn : G.Connected) (hΔ3 : 3 ≤ G.maxDegree)
    (heq : (G.maxDegree : ℚ) * (G.indepNum : ℚ) = (G.annihilationNumber : ℚ) + (G.residue : ℚ)) :
    Nonempty (G ≃g completeGraph (Fin 4)) := by
  haveI : Nonempty V := hconn.nonempty
  have ⟨ha, _⟩ := (G.txgraffiti_eq_iff_invariants hconn (by omega)).mp heq
  have ⟨hΔ_eq, hcw_eq⟩ := G.maxDegree_eq_three_and_caroWei_eq_of_annih_eq hΔ3 ha
  have htop := G.eq_top_of_connected_of_caroWei_eq hconn hcw_eq
  have hreg : G.IsRegularOfDegree (Fintype.card V - 1) := by
    intro v
    have hsub : G.neighborFinset v = Finset.univ.erase v := by
      ext w
      simp only [mem_neighborFinset, Finset.mem_erase, Finset.mem_univ, and_true]
      rw [htop]
      exact ⟨fun h => h.symm, fun h => h.symm⟩
    rw [← card_neighborFinset_eq_degree, hsub, Finset.card_erase_of_mem (Finset.mem_univ v),
      Finset.card_univ]
  have hmax_top : G.maxDegree = Fintype.card V - 1 := hreg.maxDegree_eq
  have hcard4 : Fintype.card V = 4 := by
    have hpos : 0 < Fintype.card V := Fintype.card_pos
    omega
  refine ⟨?_⟩
  rw [htop]
  exact Iso.completeGraph (Fintype.equivFinOfCardEq hcard4)

#print axioms strict_of_maxDegree_ge_four
#print axioms iso_completeGraph_four_of_maxDegree_ge_three

end SimpleGraph
