import TxGraffitiC1FC.Conjecture
import TxGraffitiC1FC.Corollaries
import TxGraffitiC1FC.ResidueMonotonicity
import TxGraffitiC1FC.DegreeTwoResidue

set_option linter.unusedSectionVars false

namespace SimpleGraph

open Finset

/-- **Pepper's theorem (`α ≤ a`).** For any independent set `I`, no edge has both endpoints
in `I`, so `∑_{v ∈ I} deg v ≤ ∑_{v ∈ univ \ I} deg v`. By the handshaking lemma
`∑_{v ∈ I} deg v + ∑_{v ∈ univ \ I} deg v = 2m`, whence `∑_{v ∈ I} deg v ≤ m` and `|I| ≤ a`. -/
theorem sum_degrees_indepSet_le_card_edges {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] {I : Finset V}
    (hind : G.IsIndepSet (I : Set V)) :
    (∑ v ∈ I, G.degree v) ≤ G.edgeFinset.card := by
  have h1 : ∀ v ∈ I, G.degree v = ∑ w ∈ Finset.univ \ I, (if G.Adj v w then 1 else 0) := by
    intro v hv
    rw [← Finset.card_filter, ← card_neighborFinset_eq_degree, neighborFinset_eq_filter]
    congr 1; ext w
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_sdiff]
    exact ⟨fun hadj => ⟨fun hwI => hind hv hwI hadj.ne hadj, hadj⟩, fun h => h.2⟩
  have h2 : ∀ w ∈ Finset.univ \ I,
      (∑ v ∈ I, (if G.Adj v w then 1 else 0)) ≤ G.degree w := by
    intro w _
    rw [← Finset.card_filter, ← card_neighborFinset_eq_degree, neighborFinset_eq_filter]
    refine Finset.card_le_card (fun v hv => ?_)
    rw [Finset.mem_filter] at hv ⊢
    exact ⟨Finset.mem_univ v, hv.2.symm⟩
  have hdart : (∑ v ∈ I, G.degree v) ≤ ∑ w ∈ Finset.univ \ I, G.degree w := by
    rw [Finset.sum_congr rfl h1, Finset.sum_comm]
    exact Finset.sum_le_sum h2
  have hsplit := Finset.sum_sdiff (Finset.subset_univ I) (f := fun v => G.degree v)
  rw [G.sum_degrees_eq_twice_card_edges] at hsplit
  omega

theorem card_le_annihilationNumber_of_isIndepSet {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] {I : Finset V}
    (hind : G.IsIndepSet (I : Set V)) :
    I.card ≤ G.annihilationNumber := by
  have hmem : (I.val.map fun v => G.degree v) ∈
      (Finset.Iic (degreeMultiset G)).filter (fun S => Multiset.sum S ≤ G.edgeFinset.card) := by
    rw [Finset.mem_filter, Finset.mem_Iic]
    refine ⟨Multiset.map_le_map (Finset.val_le_iff.mpr (Finset.subset_univ I)), ?_⟩
    exact G.sum_degrees_indepSet_le_card_edges hind
  have hsup := Finset.le_sup (f := Multiset.card) hmem
  rwa [Multiset.card_map, Finset.card_val] at hsup

theorem indepNum_le_annihilationNumber {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] :
    G.indepNum ≤ G.annihilationNumber := by
  obtain ⟨I, hind, hcard⟩ := G.exists_isNIndepSet_indepNum
  rw [← hcard]
  exact G.card_le_annihilationNumber_of_isIndepSet hind

#print axioms indepNum_le_annihilationNumber

theorem havelHakimiStep_eq_of_sorted {d : ℕ} {rest l : List ℕ}
    (hsort : l.Pairwise (· ≥ ·))
    (hperm : (rawDec d rest).Perm l) :
    havelHakimiStep (d :: rest) = l := by
  rw [havelHakimiStep_cons_eq_mergeSort_rawDec]
  exact List.Perm.eq_of_sortedGE (List.pairwise_mergeSort' _ _).sortedGE hsort.sortedGE
    ((List.mergeSort_perm _ _).trans hperm)

@[simp]
theorem residueAux_zero_cons (s : List ℕ) : residueAux (0 :: s) = 1 + s.length := by
  rw [residueAux]

theorem degree_sort_eq_of_sorted {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] {l : List ℕ}
    (hsort : l.Pairwise (· ≥ ·))
    (hperm : (Finset.univ.val.map fun v => G.degree v) = (l : Multiset ℕ)) :
    (Finset.univ.val.map fun v => G.degree v).sort (· ≥ ·) = l := by
  apply List.Perm.eq_of_sortedGE (Multiset.pairwise_sort _ _).sortedGE hsort.sortedGE
  rw [← Multiset.coe_eq_coe, Multiset.sort_eq, hperm]

theorem eq_and_eq_of_indepSet_and_le_annih {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (k : ℕ)
    (I : Finset V) (hcard : I.card = k) (hind : G.IsIndepSet (I : Set V))
    (hle : G.annihilationNumber ≤ k) :
    G.indepNum = k ∧ G.annihilationNumber = k := by
  have h1 : k ≤ G.indepNum := hcard ▸ hind.card_le_indepNum
  have h2 : G.indepNum ≤ G.annihilationNumber := G.indepNum_le_annihilationNumber
  exact ⟨by omega, by omega⟩

lemma p3_eq : ((pathGraph 3).maxDegree : ℚ) * ((pathGraph 3).indepNum : ℚ)
    = ((pathGraph 3).annihilationNumber : ℚ) + ((pathGraph 3).residue : ℚ) := by
  have hm : (pathGraph 3).maxDegree = 2 := by decide
  have ⟨hi, ha⟩ := eq_and_eq_of_indepSet_and_le_annih (pathGraph 3) 2
    ({0, 2} : Finset (Fin 3)) (by decide) (by decide)
    (by
      have h2 := (pathGraph 3).two_mul_annihilationNumber_le (by decide)
      have hm' : (pathGraph 3).edgeFinset.card = 2 := by decide
      have hn' : Fintype.card (Fin 3) = 3 := Fintype.card_fin 3
      omega)
  have hr : (pathGraph 3).residue = 2 := by
    unfold residue
    rw [degree_sort_eq_of_sorted (pathGraph 3) (by decide : [2, 1, 1].Pairwise (· ≥ ·)) (by decide),
        residueAux_cons_ne_zero 2 [1, 1] (by decide)]
    have h1 : havelHakimiStep [2, 1, 1] = [0, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h1, residueAux_zero_cons]
    rfl
  rw [hm, hi, ha, hr]; norm_num

lemma p4_eq : ((pathGraph 4).maxDegree : ℚ) * ((pathGraph 4).indepNum : ℚ)
    = ((pathGraph 4).annihilationNumber : ℚ) + ((pathGraph 4).residue : ℚ) := by
  have hm : (pathGraph 4).maxDegree = 2 := by decide
  have ⟨hi, ha⟩ := eq_and_eq_of_indepSet_and_le_annih (pathGraph 4) 2
    ({0, 2} : Finset (Fin 4)) (by decide) (by decide)
    (by
      have h2 := (pathGraph 4).two_mul_annihilationNumber_le (by decide)
      have hm' : (pathGraph 4).edgeFinset.card = 3 := by decide
      have hn' : Fintype.card (Fin 4) = 4 := Fintype.card_fin 4
      omega)
  have hr : (pathGraph 4).residue = 2 := by
    unfold residue
    rw [degree_sort_eq_of_sorted (pathGraph 4) (by decide : [2, 2, 1, 1].Pairwise (· ≥ ·)) (by decide),
        residueAux_cons_ne_zero 2 [2, 1, 1] (by decide)]
    have h1 : havelHakimiStep [2, 2, 1, 1] = [1, 1, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h1, residueAux_cons_ne_zero 1 [1, 0] (by decide)]
    have h2 : havelHakimiStep [1, 1, 0] = [0, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h2, residueAux_zero_cons]
    rfl
  rw [hm, hi, ha, hr]; norm_num

lemma p6_eq : ((pathGraph 6).maxDegree : ℚ) * ((pathGraph 6).indepNum : ℚ)
    = ((pathGraph 6).annihilationNumber : ℚ) + ((pathGraph 6).residue : ℚ) := by
  have hm : (pathGraph 6).maxDegree = 2 := by decide
  have ⟨hi, ha⟩ := eq_and_eq_of_indepSet_and_le_annih (pathGraph 6) 3
    ({0, 2, 4} : Finset (Fin 6)) (by decide) (by decide)
    (by
      have h2 := (pathGraph 6).two_mul_annihilationNumber_le (by decide)
      have hm' : (pathGraph 6).edgeFinset.card = 5 := by decide
      have hn' : Fintype.card (Fin 6) = 6 := Fintype.card_fin 6
      omega)
  have hr : (pathGraph 6).residue = 3 := by
    unfold residue
    rw [degree_sort_eq_of_sorted (pathGraph 6) (by decide : [2, 2, 2, 2, 1, 1].Pairwise (· ≥ ·)) (by decide),
        residueAux_cons_ne_zero 2 [2, 2, 2, 1, 1] (by decide)]
    have h1 : havelHakimiStep [2, 2, 2, 2, 1, 1] = [2, 1, 1, 1, 1] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h1, residueAux_cons_ne_zero 2 [1, 1, 1, 1] (by decide)]
    have h2 : havelHakimiStep [2, 1, 1, 1, 1] = [1, 1, 0, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h2, residueAux_cons_ne_zero 1 [1, 0, 0] (by decide)]
    have h3 : havelHakimiStep [1, 1, 0, 0] = [0, 0, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h3, residueAux_zero_cons]
    rfl
  rw [hm, hi, ha, hr]; norm_num

lemma c3_eq : ((cycleGraph 3).maxDegree : ℚ) * ((cycleGraph 3).indepNum : ℚ)
    = ((cycleGraph 3).annihilationNumber : ℚ) + ((cycleGraph 3).residue : ℚ) := by
  have hm : (cycleGraph 3).maxDegree = 2 := by decide
  have ⟨hi, ha⟩ := eq_and_eq_of_indepSet_and_le_annih (cycleGraph 3) 1
    ({0} : Finset (Fin 3)) (by decide) (by decide)
    (by
      have h2 := (cycleGraph 3).two_mul_annihilationNumber_le (by decide)
      have hm' : (cycleGraph 3).edgeFinset.card = 3 := by decide
      have hn' : Fintype.card (Fin 3) = 3 := Fintype.card_fin 3
      omega)
  have hr : (cycleGraph 3).residue = 1 := by
    unfold residue
    rw [degree_sort_eq_of_sorted (cycleGraph 3) (by decide : [2, 2, 2].Pairwise (· ≥ ·)) (by decide),
        residueAux_cons_ne_zero 2 [2, 2] (by decide)]
    have h1 : havelHakimiStep [2, 2, 2] = [1, 1] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h1, residueAux_cons_ne_zero 1 [1] (by decide)]
    have h2 : havelHakimiStep [1, 1] = [0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h2, residueAux_zero_cons]
    rfl
  rw [hm, hi, ha, hr]; norm_num

lemma c4_eq : ((cycleGraph 4).maxDegree : ℚ) * ((cycleGraph 4).indepNum : ℚ)
    = ((cycleGraph 4).annihilationNumber : ℚ) + ((cycleGraph 4).residue : ℚ) := by
  have hm : (cycleGraph 4).maxDegree = 2 := by decide
  have ⟨hi, ha⟩ := eq_and_eq_of_indepSet_and_le_annih (cycleGraph 4) 2
    ({0, 2} : Finset (Fin 4)) (by decide) (by decide)
    (by
      have h2 := (cycleGraph 4).two_mul_annihilationNumber_le (by decide)
      have hm' : (cycleGraph 4).edgeFinset.card = 4 := by decide
      have hn' : Fintype.card (Fin 4) = 4 := Fintype.card_fin 4
      omega)
  have hr : (cycleGraph 4).residue = 2 := by
    unfold residue
    rw [degree_sort_eq_of_sorted (cycleGraph 4) (by decide : [2, 2, 2, 2].Pairwise (· ≥ ·)) (by decide),
        residueAux_cons_ne_zero 2 [2, 2, 2] (by decide)]
    have h1 : havelHakimiStep [2, 2, 2, 2] = [2, 1, 1] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h1, residueAux_cons_ne_zero 2 [1, 1] (by decide)]
    have h2 : havelHakimiStep [2, 1, 1] = [0, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h2, residueAux_zero_cons]
    rfl
  rw [hm, hi, ha, hr]; norm_num

lemma c5_eq : ((cycleGraph 5).maxDegree : ℚ) * ((cycleGraph 5).indepNum : ℚ)
    = ((cycleGraph 5).annihilationNumber : ℚ) + ((cycleGraph 5).residue : ℚ) := by
  have hm : (cycleGraph 5).maxDegree = 2 := by decide
  have ⟨hi, ha⟩ := eq_and_eq_of_indepSet_and_le_annih (cycleGraph 5) 2
    ({0, 2} : Finset (Fin 5)) (by decide) (by decide)
    (by
      have h2 := (cycleGraph 5).two_mul_annihilationNumber_le (by decide)
      have hm' : (cycleGraph 5).edgeFinset.card = 5 := by decide
      have hn' : Fintype.card (Fin 5) = 5 := Fintype.card_fin 5
      omega)
  have hr : (cycleGraph 5).residue = 2 := by
    unfold residue
    rw [degree_sort_eq_of_sorted (cycleGraph 5) (by decide : [2, 2, 2, 2, 2].Pairwise (· ≥ ·)) (by decide),
        residueAux_cons_ne_zero 2 [2, 2, 2, 2] (by decide)]
    have h1 : havelHakimiStep [2, 2, 2, 2, 2] = [2, 2, 1, 1] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h1, residueAux_cons_ne_zero 2 [2, 1, 1] (by decide)]
    have h2 : havelHakimiStep [2, 2, 1, 1] = [1, 1, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h2, residueAux_cons_ne_zero 1 [1, 0] (by decide)]
    have h3 : havelHakimiStep [1, 1, 0] = [0, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h3, residueAux_zero_cons]
    rfl
  rw [hm, hi, ha, hr]; norm_num

lemma c7_eq : ((cycleGraph 7).maxDegree : ℚ) * ((cycleGraph 7).indepNum : ℚ)
    = ((cycleGraph 7).annihilationNumber : ℚ) + ((cycleGraph 7).residue : ℚ) := by
  have hm : (cycleGraph 7).maxDegree = 2 := by decide
  have ⟨hi, ha⟩ := eq_and_eq_of_indepSet_and_le_annih (cycleGraph 7) 3
    ({0, 2, 4} : Finset (Fin 7)) (by decide) (by decide)
    (by
      have h2 := (cycleGraph 7).two_mul_annihilationNumber_le (by decide)
      have hm' : (cycleGraph 7).edgeFinset.card = 7 := by decide
      have hn' : Fintype.card (Fin 7) = 7 := Fintype.card_fin 7
      omega)
  have hr : (cycleGraph 7).residue = 3 := by
    unfold residue
    rw [degree_sort_eq_of_sorted (cycleGraph 7) (by decide : [2, 2, 2, 2, 2, 2, 2].Pairwise (· ≥ ·)) (by decide),
        residueAux_cons_ne_zero 2 [2, 2, 2, 2, 2, 2] (by decide)]
    have h1 : havelHakimiStep [2, 2, 2, 2, 2, 2, 2] = [2, 2, 2, 2, 1, 1] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h1, residueAux_cons_ne_zero 2 [2, 2, 2, 1, 1] (by decide)]
    have h2 : havelHakimiStep [2, 2, 2, 2, 1, 1] = [2, 1, 1, 1, 1] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h2, residueAux_cons_ne_zero 2 [1, 1, 1, 1] (by decide)]
    have h3 : havelHakimiStep [2, 1, 1, 1, 1] = [1, 1, 0, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h3, residueAux_cons_ne_zero 1 [1, 0, 0] (by decide)]
    have h4 : havelHakimiStep [1, 1, 0, 0] = [0, 0, 0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h4, residueAux_zero_cons]
    rfl
  rw [hm, hi, ha, hr]; norm_num

lemma k4_eq : ((completeGraph (Fin 4)).maxDegree : ℚ) * ((completeGraph (Fin 4)).indepNum : ℚ)
    = ((completeGraph (Fin 4)).annihilationNumber : ℚ) + ((completeGraph (Fin 4)).residue : ℚ) := by
  have hm : (completeGraph (Fin 4)).maxDegree = 3 := by decide
  have hi : (completeGraph (Fin 4)).indepNum = 1 := by
    rw [← alphaIn_univ]; decide
  have ha : (completeGraph (Fin 4)).annihilationNumber = 2 := by decide
  have hr : (completeGraph (Fin 4)).residue = 1 := by
    unfold residue
    rw [degree_sort_eq_of_sorted (completeGraph (Fin 4)) (by decide : [3, 3, 3, 3].Pairwise (· ≥ ·)) (by decide),
        residueAux_cons_ne_zero 3 [3, 3, 3] (by decide)]
    have h1 : havelHakimiStep [3, 3, 3, 3] = [2, 2, 2] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h1, residueAux_cons_ne_zero 2 [2, 2] (by decide)]
    have h2 : havelHakimiStep [2, 2, 2] = [1, 1] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h2, residueAux_cons_ne_zero 1 [1] (by decide)]
    have h3 : havelHakimiStep [1, 1] = [0] :=
      havelHakimiStep_eq_of_sorted (by decide) (by decide)
    rw [h3, residueAux_zero_cons]
    rfl
  rw [hm, hi, ha, hr]; norm_num

#print axioms p3_eq
#print axioms p4_eq
#print axioms p6_eq
#print axioms c3_eq
#print axioms c4_eq
#print axioms c5_eq
#print axioms c7_eq
#print axioms k4_eq

end SimpleGraph
