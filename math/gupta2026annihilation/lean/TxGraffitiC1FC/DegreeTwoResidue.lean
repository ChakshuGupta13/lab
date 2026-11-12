import TxGraffitiC1FC.Favaron
import TxGraffitiC1FC.ResidueLemmas
import TxGraffitiC1FC.ResidueMonotonicity

set_option linter.unusedSectionVars false

namespace SimpleGraph

open Finset

/-- Decreasing list `2, ..., 2, 1, ..., 1` with `k` twos followed by `h` ones. -/
def twosOnes (k h : ℕ) : List ℕ :=
  List.replicate k 2 ++ List.replicate h 1

theorem pairwise_twosOnes (k h : ℕ) : (twosOnes k h).Pairwise (· ≥ ·) := by
  unfold twosOnes
  rw [List.pairwise_append]
  refine ⟨List.pairwise_replicate.mpr (Or.inr le_rfl),
          List.pairwise_replicate.mpr (Or.inr le_rfl), ?_⟩
  intro a ha b hb
  rw [List.mem_replicate] at ha hb
  omega

theorem rawDec_twosOnes_ge_three (k h : ℕ) :
    rawDec 2 (twosOnes (k + 2) h) = [1, 1] ++ twosOnes k h := by
  unfold rawDec twosOnes
  have hrep : List.replicate (k + 2) 2 = [2, 2] ++ List.replicate k 2 := by
    rw [Nat.add_comm k 2, List.replicate_add]
    rfl
  rw [hrep, List.append_assoc, List.take_append_of_le_length (by decide),
      List.drop_append_of_le_length (by decide)]
  rfl

theorem havelHakimiStep_twosOnes_ge_three (k h : ℕ) :
    havelHakimiStep (twosOnes (k + 3) h) = twosOnes k (h + 2) := by
  have hcons : twosOnes (k + 3) h = 2 :: twosOnes (k + 2) h := by
    unfold twosOnes
    rw [show k + 3 = (k + 2) + 1 by omega, List.replicate_succ, List.cons_append]
  rw [hcons, havelHakimiStep_cons_eq_mergeSort_rawDec, rawDec_twosOnes_ge_three]
  apply List.Perm.eq_of_sortedGE (List.pairwise_mergeSort' _ _).sortedGE
    (pairwise_twosOnes k (h + 2)).sortedGE
  refine (List.mergeSort_perm _ _).trans ?_
  unfold twosOnes
  rw [show h + 2 = 2 + h by omega, List.replicate_add]
  change ([1, 1] ++ (List.replicate k 2 ++ List.replicate h 1)).Perm
    (List.replicate k 2 ++ ([1, 1] ++ List.replicate h 1))
  rw [← List.append_assoc, ← List.append_assoc]
  exact List.perm_append_comm.append_right _

theorem residueAux_twosOnes_step_three (k h : ℕ) :
    residueAux (twosOnes (k + 3) h) = residueAux (twosOnes k (h + 2)) := by
  have hcons : twosOnes (k + 3) h = 2 :: twosOnes (k + 2) h := by
    unfold twosOnes
    rw [show k + 3 = (k + 2) + 1 by omega, List.replicate_succ, List.cons_append]
  conv_lhs => rw [hcons]
  rw [residueAux_cons_ne_zero 2 (twosOnes (k + 2) h) (by decide),
      ← hcons, havelHakimiStep_twosOnes_ge_three]

/-- `onesZeros h z` is `h` ones followed by `z` zeros. -/
def onesZeros (h z : ℕ) : List ℕ :=
  List.replicate h 1 ++ List.replicate z 0

theorem pairwise_onesZeros (h z : ℕ) : (onesZeros h z).Pairwise (· ≥ ·) := by
  unfold onesZeros
  rw [List.pairwise_append]
  refine ⟨List.pairwise_replicate.mpr (Or.inr le_rfl),
          List.pairwise_replicate.mpr (Or.inr le_rfl), ?_⟩
  intro a ha b hb
  rw [List.mem_replicate] at ha hb
  omega

theorem havelHakimiStep_onesZeros (h z : ℕ) :
    havelHakimiStep (onesZeros (h + 2) z) = onesZeros h (z + 1) := by
  have hcons : onesZeros (h + 2) z = 1 :: (1 :: onesZeros h z) := by
    unfold onesZeros
    rw [show h + 2 = (h + 1) + 1 by omega, List.replicate_succ, List.replicate_succ,
        List.cons_append, List.cons_append]
  rw [hcons, havelHakimiStep_cons_eq_mergeSort_rawDec]
  have hraw : rawDec 1 (1 :: onesZeros h z) = 0 :: onesZeros h z := rfl
  rw [hraw]
  apply List.Perm.eq_of_sortedGE (List.pairwise_mergeSort' _ _).sortedGE
    (pairwise_onesZeros h (z + 1)).sortedGE
  refine (List.mergeSort_perm _ _).trans ?_
  unfold onesZeros
  rw [List.replicate_succ, ← List.singleton_append, ← List.singleton_append (l := List.replicate z 0),
      ← List.append_assoc, ← List.append_assoc]
  exact List.perm_append_comm.append_right _

theorem residueAux_onesZeros_even (m z : ℕ) :
    residueAux (onesZeros (2 * m) z) = m + z := by
  induction m generalizing z with
  | zero =>
    unfold onesZeros
    simp only [Nat.mul_zero, List.replicate_zero, List.nil_append, Nat.zero_add]
    exact residueAux_replicate_zero z
  | succ m' ih =>
    have hcons : onesZeros (2 * (m' + 1)) z = 1 :: (1 :: onesZeros (2 * m') z) := by
      unfold onesZeros
      rw [show 2 * (m' + 1) = (2 * m' + 1) + 1 by omega, List.replicate_succ, List.replicate_succ,
          List.cons_append, List.cons_append]
    rw [hcons, residueAux_cons_ne_zero 1 _ (by decide), ← hcons,
        show 2 * (m' + 1) = 2 * m' + 2 by omega, havelHakimiStep_onesZeros, ih (z + 1)]
    omega

theorem residueAux_twosOnes_zero (m : ℕ) :
    residueAux (twosOnes 0 (2 * m)) = m := by
  have heq : twosOnes 0 (2 * m) = onesZeros (2 * m) 0 := by
    unfold twosOnes onesZeros; simp
  rw [heq, residueAux_onesZeros_even]
  omega

theorem residueAux_twosOnes_one (m : ℕ) :
    residueAux (twosOnes 1 (2 * (m + 1))) = m + 2 := by
  have hcons : twosOnes 1 (2 * (m + 1)) = 2 :: (1 :: 1 :: List.replicate (2 * m) 1) := by
    unfold twosOnes
    rw [show 2 * (m + 1) = (2 * m + 1) + 1 by omega, List.replicate_succ, List.replicate_succ]
    rfl
  rw [hcons, residueAux_cons_ne_zero 2 _ (by decide)]
  have hstep : havelHakimiStep (2 :: 1 :: 1 :: List.replicate (2 * m) 1) = onesZeros (2 * m) 2 := by
    rw [havelHakimiStep_cons_eq_mergeSort_rawDec]
    have hraw : rawDec 2 (1 :: 1 :: List.replicate (2 * m) 1) = [0, 0] ++ List.replicate (2 * m) 1 := rfl
    rw [hraw]
    apply List.Perm.eq_of_sortedGE (List.pairwise_mergeSort' _ _).sortedGE
      (pairwise_onesZeros (2 * m) 2).sortedGE
    exact (List.mergeSort_perm _ _).trans List.perm_append_comm
  rw [hstep, residueAux_onesZeros_even]

theorem havelHakimiStep_eq_of_sorted' {d : ℕ} {rest l : List ℕ}
    (hsort : l.Pairwise (· ≥ ·))
    (hperm : (rawDec d rest).Perm l) :
    havelHakimiStep (d :: rest) = l := by
  rw [havelHakimiStep_cons_eq_mergeSort_rawDec]
  exact List.Perm.eq_of_sortedGE (List.pairwise_mergeSort' _ _).sortedGE hsort.sortedGE
    ((List.mergeSort_perm _ _).trans hperm)

theorem residueAux_twosOnes_two (m : ℕ) :
    residueAux (twosOnes 2 (2 * (m + 1))) = m + 2 := by
  have hcons : twosOnes 2 (2 * (m + 1)) = 2 :: (2 :: 1 :: List.replicate (2 * m + 1) 1) := by
    unfold twosOnes
    rw [show 2 * (m + 1) = (2 * m + 1) + 1 by omega, List.replicate_succ]
    rfl
  rw [hcons, residueAux_cons_ne_zero 2 _ (by decide)]
  have hstep : havelHakimiStep (2 :: 2 :: 1 :: List.replicate (2 * m + 1) 1)
      = onesZeros (2 * (m + 1)) 1 := by
    rw [havelHakimiStep_cons_eq_mergeSort_rawDec]
    have hraw : rawDec 2 (2 :: 1 :: List.replicate (2 * m + 1) 1)
        = [1] ++ [0] ++ List.replicate (2 * m + 1) 1 := rfl
    rw [hraw]
    apply List.Perm.eq_of_sortedGE (List.pairwise_mergeSort' _ _).sortedGE
      (pairwise_onesZeros (2 * (m + 1)) 1).sortedGE
    refine (List.mergeSort_perm _ _).trans ?_
    have hone : onesZeros (2 * (m + 1)) 1 = [1] ++ List.replicate (2 * m + 1) 1 ++ [0] := by
      unfold onesZeros
      rw [show 2 * (m + 1) = (2 * m + 1) + 1 by omega, List.replicate_succ]
      rfl
    rw [hone, List.append_assoc, List.append_assoc]
    exact List.Perm.append_left [1] List.perm_append_comm
  rw [hstep, residueAux_onesZeros_even]

/-- **Exact residue formula for `twosOnes k (2 * (m + 1))`** (Favaron et al. 1991, Theorem 1.5
for `p = 2` with `h = 2(m+1) ≥ 2`): `residueAux (twosOnes k (2 * (m + 1))) = (k + 2) / 3 + m + 1`. -/
theorem residueAux_twosOnes_even_pos :
    ∀ k m : ℕ,
      residueAux (twosOnes k (2 * (m + 1))) = (k + 2) / 3 + m + 1
  | 0, m => by
    rw [residueAux_twosOnes_zero]; omega
  | 1, m => by
    rw [residueAux_twosOnes_one]; omega
  | 2, m => by
    rw [residueAux_twosOnes_two]; omega
  | k + 3, m => by
    have hstep := residueAux_twosOnes_step_three k (2 * (m + 1))
    have h2 : 2 * (m + 1) + 2 = 2 * (m + 2) := by omega
    rw [hstep, h2, residueAux_twosOnes_even_pos k (m + 1)]
    have hdiv : (k + 3 + 2) / 3 = (k + 2) / 3 + 1 := by
      rw [show k + 3 + 2 = (k + 2) + 3 by omega, Nat.add_div_right _ (by decide)]
    omega

/-- **Path degree sequence (`n - 2` twos and `2` ones, for `n ≥ 3`)**:
`residueAux (twosOnes (n - 2) 2) = (n + 3) / 3` (`= ⌈(n + 1) / 3⌉`). -/
theorem residueAux_path_degSeq (n : ℕ) (hn : 3 ≤ n) :
    residueAux (twosOnes (n - 2) 2) = (n + 3) / 3 := by
  have h := residueAux_twosOnes_even_pos (n - 2) 0
  have hdiv : (n + 3) / 3 = (n - 2 + 2) / 3 + 1 := by
    rw [show n + 3 = (n - 2 + 2) + 3 by omega, Nat.add_div_right _ (by decide)]
  simpa [hdiv] using h

/-- **Cycle degree sequence (`n` twos, for `n ≥ 3`)**:
`residueAux (twosOnes n 0) = (n + 2) / 3` (`= ⌈n / 3⌉`). -/
theorem residueAux_cycle_degSeq (n : ℕ) (hn : 3 ≤ n) :
    residueAux (twosOnes n 0) = (n + 2) / 3 := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 3 := ⟨n - 3, by omega⟩
  rw [residueAux_twosOnes_step_three k 0]
  have h := residueAux_twosOnes_even_pos k 0
  have hdiv : (k + 3 + 2) / 3 = (k + 2) / 3 + 1 := by
    rw [show k + 3 + 2 = (k + 2) + 3 by omega, Nat.add_div_right _ (by decide)]
  simpa [hdiv] using h

#print axioms residueAux_path_degSeq
#print axioms residueAux_cycle_degSeq

/-- If every element of `m : Multiset ℕ` is `1` or `2` and `m.sum = 2 * m.card - 2`,
then `m.sort (· ≥ ·) = twosOnes (m.card - 2) 2`. -/
theorem sort_eq_twosOnes_of_sum_eq_two_mul_card_sub_two
    (m : Multiset ℕ) (hmem : ∀ d ∈ m, d = 1 ∨ d = 2)
    (hcard : 3 ≤ m.card) (hsum : m.sum = 2 * m.card - 2) :
    m.sort (· ≥ ·) = twosOnes (m.card - 2) 2 := by
  set k := (m.filter (· = 2)).card
  set h := (m.filter (· = 1)).card
  have hpart : m = m.filter (· = 2) + m.filter (· = 1) := by
    have hsplit := (Multiset.filter_add_not (· = 2) m).symm
    have hnot : m.filter (fun d => ¬ d = 2) = m.filter (· = 1) := by
      apply Multiset.filter_congr
      intro d hd
      rcases hmem d hd with rfl | rfl <;> decide
    rwa [hnot] at hsplit
  have hf2 : m.filter (· = 2) = Multiset.replicate k 2 := by
    rw [Multiset.eq_replicate]
    exact ⟨rfl, fun d hd => (Multiset.mem_filter.mp hd).2⟩
  have hf1 : m.filter (· = 1) = Multiset.replicate h 1 := by
    rw [Multiset.eq_replicate]
    exact ⟨rfl, fun d hd => (Multiset.mem_filter.mp hd).2⟩
  have hm_rep : m = Multiset.replicate k 2 + Multiset.replicate h 1 := by
    rw [hpart, hf2, hf1]
  have hcard_eq : m.card = k + h := by
    have hc := congrArg Multiset.card hm_rep
    simpa using hc
  have hsum_eq : m.sum = 2 * k + h := by
    have hs := congrArg Multiset.sum hm_rep
    simp only [Multiset.sum_add, Multiset.sum_replicate, smul_eq_mul, mul_one] at hs
    omega
  have hk_eq : k = m.card - 2 := by omega
  have hh_eq : h = 2 := by omega
  apply List.Perm.eq_of_sortedGE (Multiset.pairwise_sort _ _).sortedGE
    (pairwise_twosOnes (m.card - 2) 2).sortedGE
  rw [← Multiset.coe_eq_coe, Multiset.sort_eq]
  conv_lhs => rw [hm_rep]
  rw [hk_eq, hh_eq]
  rfl

/-- If every element of `m : Multiset ℕ` is `1` or `2` and `m.sum = 2 * m.card`,
then `m.sort (· ≥ ·) = twosOnes m.card 0`. -/
theorem sort_eq_twosOnes_of_sum_eq_two_mul_card
    (m : Multiset ℕ) (hmem : ∀ d ∈ m, d = 1 ∨ d = 2)
    (hsum : m.sum = 2 * m.card) :
    m.sort (· ≥ ·) = twosOnes m.card 0 := by
  set k := (m.filter (· = 2)).card
  set h := (m.filter (· = 1)).card
  have hpart : m = m.filter (· = 2) + m.filter (· = 1) := by
    have hsplit := (Multiset.filter_add_not (· = 2) m).symm
    have hnot : m.filter (fun d => ¬ d = 2) = m.filter (· = 1) := by
      apply Multiset.filter_congr
      intro d hd
      rcases hmem d hd with rfl | rfl <;> decide
    rwa [hnot] at hsplit
  have hf2 : m.filter (· = 2) = Multiset.replicate k 2 := by
    rw [Multiset.eq_replicate]
    exact ⟨rfl, fun d hd => (Multiset.mem_filter.mp hd).2⟩
  have hf1 : m.filter (· = 1) = Multiset.replicate h 1 := by
    rw [Multiset.eq_replicate]
    exact ⟨rfl, fun d hd => (Multiset.mem_filter.mp hd).2⟩
  have hm_rep : m = Multiset.replicate k 2 + Multiset.replicate h 1 := by
    rw [hpart, hf2, hf1]
  have hcard_eq : m.card = k + h := by
    have hc := congrArg Multiset.card hm_rep
    simpa using hc
  have hsum_eq : m.sum = 2 * k + h := by
    have hs := congrArg Multiset.sum hm_rep
    simp only [Multiset.sum_add, Multiset.sum_replicate, smul_eq_mul, mul_one] at hs
    omega
  have hk_eq : k = m.card := by omega
  have hh_eq : h = 0 := by omega
  apply List.Perm.eq_of_sortedGE (Multiset.pairwise_sort _ _).sortedGE
    (pairwise_twosOnes m.card 0).sortedGE
  rw [← Multiset.coe_eq_coe, Multiset.sort_eq]
  conv_lhs => rw [hm_rep]
  rw [hk_eq, hh_eq]
  rfl

#print axioms sort_eq_twosOnes_of_sum_eq_two_mul_card_sub_two
#print axioms sort_eq_twosOnes_of_sum_eq_two_mul_card

open scoped Classical

instance (n : ℕ) : DecidableRel (pathGraph n).Adj :=
  fun u v => decidable_of_iff (u.val + 1 = v.val ∨ v.val + 1 = u.val) pathGraph_adj.symm

theorem pathGraph_neighborFinset_le (n : ℕ) (v : Fin n) :
    (pathGraph n).neighborFinset v ⊆
      (if h0 : 0 < v.val then {⟨v.val - 1, by omega⟩} else ∅) ∪
      (if h1 : v.val + 1 < n then {⟨v.val + 1, h1⟩} else ∅) := by
  intro w hw
  rw [mem_neighborFinset, pathGraph_adj] at hw
  rw [Finset.mem_union]
  rcases hw with hw | hw
  · right
    have h1 : v.val + 1 < n := hw ▸ w.isLt
    rw [dif_pos h1, Finset.mem_singleton, Fin.ext_iff]
    exact hw.symm
  · left
    have h0 : 0 < v.val := by omega
    rw [dif_pos h0, Finset.mem_singleton, Fin.ext_iff]
    dsimp; omega

theorem pathGraph_degree_le_two (n : ℕ) (v : Fin n) :
    (pathGraph n).degree v ≤ 2 := by
  rw [← card_neighborFinset_eq_degree]
  calc ((pathGraph n).neighborFinset v).card
      ≤ ((if h0 : 0 < v.val then ({⟨v.val - 1, by omega⟩} : Finset (Fin n)) else ∅) ∪
         (if h1 : v.val + 1 < n then ({⟨v.val + 1, h1⟩} : Finset (Fin n)) else ∅)).card :=
        Finset.card_le_card (pathGraph_neighborFinset_le n v)
    _ ≤ (if h0 : 0 < v.val then ({⟨v.val - 1, by omega⟩} : Finset (Fin n)) else ∅).card +
        (if h1 : v.val + 1 < n then ({⟨v.val + 1, h1⟩} : Finset (Fin n)) else ∅).card :=
        Finset.card_union_le _ _
    _ ≤ 1 + 1 := by
        apply Nat.add_le_add
        · split_ifs <;> simp
        · split_ifs <;> simp

theorem pathGraph_degree_zero_le_one (n : ℕ) (hn : 0 < n) :
    (pathGraph n).degree ⟨0, hn⟩ ≤ 1 := by
  rw [← card_neighborFinset_eq_degree]
  have hsub := pathGraph_neighborFinset_le n ⟨0, hn⟩
  simp only [lt_self_iff_false, ↓reduceDIte, Finset.empty_union] at hsub
  calc ((pathGraph n).neighborFinset ⟨0, hn⟩).card
      ≤ (if h1 : 0 + 1 < n then ({⟨0 + 1, h1⟩} : Finset (Fin n)) else ∅).card :=
        Finset.card_le_card hsub
    _ ≤ 1 := by split_ifs <;> simp

theorem pathGraph_degree_last_le_one (n : ℕ) (hn : 0 < n) :
    (pathGraph n).degree ⟨n - 1, by omega⟩ ≤ 1 := by
  rw [← card_neighborFinset_eq_degree]
  have hsub := pathGraph_neighborFinset_le n ⟨n - 1, by omega⟩
  have hnot : ¬ (n - 1 + 1 < n) := by omega
  simp only [hnot, ↓reduceDIte, Finset.union_empty] at hsub
  calc ((pathGraph n).neighborFinset ⟨n - 1, by omega⟩).card
      ≤ (if h0 : 0 < n - 1 then ({⟨n - 1 - 1, by omega⟩} : Finset (Fin n)) else ∅).card :=
        Finset.card_le_card hsub
    _ ≤ 1 := by split_ifs <;> simp

theorem pathGraph_sum_degrees_eq (n : ℕ) (hn : 2 ≤ n) :
    (∑ v : Fin n, (pathGraph n).degree v) = 2 * n - 2 := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have hconn : (pathGraph (m + 1)).Connected := pathGraph_connected m
  obtain ⟨T, hTle, hTtree⟩ := hconn.exists_isTree_le
  have hmge : m ≤ (pathGraph (m + 1)).edgeFinset.card := by
    have h1 : T.edgeFinset.card ≤ (pathGraph (m + 1)).edgeFinset.card :=
      Finset.card_le_card (edgeFinset_mono hTle)
    have h2 : T.edgeFinset.card + 1 = m + 1 := by
      simpa using hTtree.card_edgeFinset
    omega
  have hsum_ge : 2 * m ≤ ∑ v : Fin (m + 1), (pathGraph (m + 1)).degree v := by
    rw [sum_degrees_eq_twice_card_edges]
    omega
  -- For the upper bound, separate 0 and m from the rest of Fin (m + 1):
  let v0 : Fin (m + 1) := ⟨0, by omega⟩
  let vm : Fin (m + 1) := ⟨m, by omega⟩
  have hne : v0 ≠ vm := by
    intro h; simp only [v0, vm, Fin.ext_iff] at h; omega
  have hsplit1 := (Finset.add_sum_erase (Finset.univ : Finset (Fin (m + 1)))
    (fun v => (pathGraph (m + 1)).degree v) (Finset.mem_univ v0)).symm
  have hvm_mem : vm ∈ (Finset.univ : Finset (Fin (m + 1))).erase v0 :=
    Finset.mem_erase.mpr ⟨hne.symm, Finset.mem_univ vm⟩
  have hsplit2 := (Finset.add_sum_erase ((Finset.univ : Finset (Fin (m + 1))).erase v0)
    (fun v => (pathGraph (m + 1)).degree v) hvm_mem).symm
  have hrest_le : (∑ v ∈ ((Finset.univ : Finset (Fin (m + 1))).erase v0).erase vm,
      (pathGraph (m + 1)).degree v) ≤ 2 * (m - 1) := by
    calc (∑ v ∈ ((Finset.univ : Finset (Fin (m + 1))).erase v0).erase vm,
            (pathGraph (m + 1)).degree v)
        ≤ ∑ _v ∈ ((Finset.univ : Finset (Fin (m + 1))).erase v0).erase vm, 2 :=
          Finset.sum_le_sum (fun v _ => pathGraph_degree_le_two (m + 1) v)
      _ = (((Finset.univ : Finset (Fin (m + 1))).erase v0).erase vm).card * 2 := by
          rw [Finset.sum_const, smul_eq_mul]
      _ = 2 * (m - 1) := by
          rw [Finset.card_erase_of_mem hvm_mem, Finset.card_erase_of_mem (Finset.mem_univ v0),
              Finset.card_univ, Fintype.card_fin]
          omega
  have hd0 := pathGraph_degree_zero_le_one (m + 1) (by omega)
  have hdm := pathGraph_degree_last_le_one (m + 1) (by omega)
  change (pathGraph (m + 1)).degree v0 ≤ 1 at hd0
  change (pathGraph (m + 1)).degree vm ≤ 1 at hdm
  omega

theorem pathGraph_residue_eq (n : ℕ) (hn : 3 ≤ n) :
    (pathGraph n).residue = (n + 3) / 3 := by
  unfold residue
  have hmem : ∀ d ∈ (Finset.univ.val.map fun v => (pathGraph n).degree v), d = 1 ∨ d = 2 := by
    intro d hd
    rw [Multiset.mem_map] at hd
    obtain ⟨v, _, rfl⟩ := hd
    obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
    haveI : Nontrivial (Fin (m + 1)) := Fin.nontrivial_iff_two_le.mpr (by omega)
    have hpos : 0 < (pathGraph (m + 1)).degree v :=
      (pathGraph_connected m).preconnected.degree_pos_of_nontrivial v
    have hle2 := pathGraph_degree_le_two (m + 1) v
    omega
  have hcard : (Finset.univ.val.map fun v => (pathGraph n).degree v).card = n := by
    rw [Multiset.card_map, Finset.card_val, Finset.card_univ, Fintype.card_fin]
  have hsum : (Finset.univ.val.map fun v => (pathGraph n).degree v).sum = 2 * n - 2 := by
    change (∑ v : Fin n, (pathGraph n).degree v) = 2 * n - 2
    exact pathGraph_sum_degrees_eq n (by omega)
  have hsort := sort_eq_twosOnes_of_sum_eq_two_mul_card_sub_two
    (Finset.univ.val.map fun v => (pathGraph n).degree v) hmem
    (by rw [hcard]; exact hn) (by rw [hcard, hsum])
  rw [hcard] at hsort
  rw [hsort, residueAux_path_degSeq n hn]

theorem cycleGraph_residue_eq (n : ℕ) (hn : 3 ≤ n) :
    (cycleGraph n).residue = (n + 2) / 3 := by
  unfold residue
  have hdeg : ∀ v : Fin n, (cycleGraph n).degree v = 2 := by
    obtain ⟨k, rfl⟩ : ∃ k, n = k + 3 := ⟨n - 3, by omega⟩
    intro v; exact cycleGraph_degree_three_le
  have hmem : ∀ d ∈ (Finset.univ.val.map fun v => (cycleGraph n).degree v), d = 1 ∨ d = 2 := by
    intro d hd
    rw [Multiset.mem_map] at hd
    obtain ⟨v, _, rfl⟩ := hd
    right; exact hdeg v
  have hcard : (Finset.univ.val.map fun v => (cycleGraph n).degree v).card = n := by
    rw [Multiset.card_map, Finset.card_val, Finset.card_univ, Fintype.card_fin]
  have hsum : (Finset.univ.val.map fun v => (cycleGraph n).degree v).sum = 2 * n := by
    change (∑ v : Fin n, (cycleGraph n).degree v) = 2 * n
    simp only [hdeg, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    omega
  have hsort := sort_eq_twosOnes_of_sum_eq_two_mul_card
    (Finset.univ.val.map fun v => (cycleGraph n).degree v) hmem
    (by rw [hcard, hsum])
  rw [hcard] at hsort
  rw [hsort, residueAux_cycle_degSeq n hn]

#print axioms pathGraph_residue_eq
#print axioms cycleGraph_residue_eq

/-- For `pathGraph n`, the even-indexed vertices form an independent set of cardinality
`(n + 1) / 2`, so `(n + 1) / 2 ≤ (pathGraph n).indepNum`. -/
theorem pathGraph_indepNum_ge (n : ℕ) :
    (n + 1) / 2 ≤ (pathGraph n).indepNum := by
  set I : Finset (Fin n) := Finset.univ.filter (fun v : Fin n => v.val % 2 = 0)
  have hind : (pathGraph n).IsIndepSet (I : Set (Fin n)) := by
    intro u hu v hv _ hadj
    simp only [I, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hu hv
    rw [pathGraph_adj] at hadj
    omega
  have hcard : I.card = (n + 1) / 2 := by
    -- Induction on n using Fin.castSucc / Fin.last!
    -- Or map `k : Fin ((n + 1) / 2) ↦ ⟨2 * k, ...⟩` into `I`!
    let f : Fin ((n + 1) / 2) ↪ Fin n :=
      ⟨fun k => ⟨2 * k.val, by have := k.isLt; omega⟩,
       fun a b h => by simp only [Fin.ext_iff] at h ⊢; omega⟩
    have hmap : Finset.univ.map f = I := by
      ext v
      simp only [Finset.mem_map, Finset.mem_univ, true_and, I, Finset.mem_filter]
      refine ⟨?_, ?_⟩
      · rintro ⟨k, rfl⟩
        change (2 * k.val) % 2 = 0
        omega
      · intro hv
        refine ⟨⟨v.val / 2, by have := v.isLt; omega⟩, ?_⟩
        ext
        change 2 * (v.val / 2) = v.val
        omega
    rw [← hmap, Finset.card_map, Finset.card_univ, Fintype.card_fin]
  rw [← hcard]
  exact hind.card_le_indepNum

/-- For `cycleGraph n` with `n ≥ 1`, the even-indexed vertices `< n - 1` form an independent set
of cardinality `n / 2`, so `n / 2 ≤ (cycleGraph n).indepNum`. -/
theorem cycleGraph_indepNum_ge (n : ℕ) (hn : 3 ≤ n) :
    n / 2 ≤ (cycleGraph n).indepNum := by
  set I : Finset (Fin n) := Finset.univ.filter (fun v : Fin n => v.val % 2 = 0 ∧ v.val + 1 < n)
  have hind : (cycleGraph n).IsIndepSet (I : Set (Fin n)) := by
    intro u hu v hv _ hadj
    simp only [I, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hu hv
    rw [cycleGraph_adj'] at hadj
    have hu_lt := u.isLt
    have hv_lt := v.isLt
    rcases hadj with h1 | h2
    · by_cases hle : v ≤ u
      · rw [Fin.coe_sub_iff_le.mpr hle] at h1; omega
      · rw [Fin.coe_sub_iff_lt.mpr (not_le.mp hle)] at h1; omega
    · by_cases hle : u ≤ v
      · rw [Fin.coe_sub_iff_le.mpr hle] at h2; omega
      · rw [Fin.coe_sub_iff_lt.mpr (not_le.mp hle)] at h2; omega
  have hcard : I.card = n / 2 := by
    let f : Fin (n / 2) ↪ Fin n :=
      ⟨fun k => ⟨2 * k.val, by have := k.isLt; omega⟩,
       fun a b h => by simp only [Fin.ext_iff] at h ⊢; omega⟩
    have hmap : Finset.univ.map f = I := by
      ext v
      simp only [Finset.mem_map, Finset.mem_univ, true_and, I, Finset.mem_filter]
      refine ⟨?_, ?_⟩
      · rintro ⟨k, rfl⟩
        have hk := k.isLt
        change (2 * k.val) % 2 = 0 ∧ 2 * k.val + 1 < n
        omega
      · rintro ⟨hv0, hv1⟩
        refine ⟨⟨v.val / 2, by omega⟩, ?_⟩
        ext
        change 2 * (v.val / 2) = v.val
        omega
    rw [← hmap, Finset.card_map, Finset.card_univ, Fintype.card_fin]
  rw [← hcard]
  exact hind.card_le_indepNum

/-- If `(pathGraph n).residue = (pathGraph n).indepNum` and `3 ≤ n`, then `n = 3 ∨ n = 4 ∨ n = 6`. -/
theorem eq_three_or_four_or_six_of_pathGraph_residue_eq_indepNum
    (n : ℕ) (hn : 3 ≤ n) (heq : (pathGraph n).residue = (pathGraph n).indepNum) :
    n = 3 ∨ n = 4 ∨ n = 6 := by
  have hres := pathGraph_residue_eq n hn
  have hind := pathGraph_indepNum_ge n
  by_cases hn5 : n = 5
  · subst hn5
    -- For n = 5, residue is 2, whereas indepNum ≥ 3, contradiction!
    omega
  · omega

/-- If `(cycleGraph n).residue = (cycleGraph n).indepNum` and `3 ≤ n`, then
`n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 7`. -/
theorem eq_three_or_four_or_five_or_seven_of_cycleGraph_residue_eq_indepNum
    (n : ℕ) (hn : 3 ≤ n) (heq : (cycleGraph n).residue = (cycleGraph n).indepNum) :
    n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 7 := by
  have hres := cycleGraph_residue_eq n hn
  have hind := cycleGraph_indepNum_ge n hn
  by_cases hn6 : n = 6
  · subst hn6
    -- For n = 6, residue is 2, whereas indepNum ≥ 3, contradiction!
    omega
  · omega

#print axioms eq_three_or_four_or_six_of_pathGraph_residue_eq_indepNum
#print axioms eq_three_or_four_or_five_or_seven_of_cycleGraph_residue_eq_indepNum

end SimpleGraph
