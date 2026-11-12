import TxGraffitiC1FC.ResidueLemmas
import TxGraffitiC1FC.Graphical
import TxGraffitiC1FC.Majorization
import TxGraffitiC1FC.HavelHakimi

namespace SimpleGraph

open TxGraffitiC1FC

/-- `IsInc M L z` states that `M` and `L` decompose around a common prefix `P` and suffix `S`
with `z` in `M` replaced by `z + 1` in `L`, and with every element of `P` strictly greater than `z`
and every element of `S` at most `z`. -/
def IsInc (M L : List ℕ) (z : ℕ) : Prop :=
  ∃ P S : List ℕ,
    (∀ w ∈ P, z + 1 ≤ w) ∧
    (∀ w ∈ S, w ≤ z) ∧
    M = P ++ (z :: S) ∧
    L = P ++ ((z + 1) :: S)

/-- Multiset form of a single increment: `L` is obtained from `M` by replacing one copy of `z`
with `z + 1`. -/
def IsIncM (M L : List ℕ) (z : ℕ) : Prop :=
  ∃ C : Multiset ℕ, (M : Multiset ℕ) = z ::ₘ C ∧ (L : Multiset ℕ) = (z + 1) ::ₘ C

theorem coe_append_cons (P S : List ℕ) (a : ℕ) :
    ((P ++ (a :: S) : List ℕ) : Multiset ℕ) = a ::ₘ ((P ++ S : List ℕ) : Multiset ℕ) := by
  rw [← Multiset.coe_add, ← Multiset.coe_add, ← Multiset.cons_coe,
    add_comm (P : Multiset ℕ), Multiset.cons_add, add_comm]

theorem IsInc.to_isIncM {M L : List ℕ} {z : ℕ} (h : IsInc M L z) : IsIncM M L z := by
  rcases h with ⟨P, S, _, _, rfl, rfl⟩
  exact ⟨((P ++ S : List ℕ) : Multiset ℕ), coe_append_cons P S z, coe_append_cons P S (z + 1)⟩

theorem IsIncM.length_eq {M L : List ℕ} {z : ℕ} (h : IsIncM M L z) :
    M.length = L.length := by
  rcases h with ⟨C, hM, hL⟩
  have h1 : Multiset.card (M : Multiset ℕ) = Multiset.card (z ::ₘ C) := congrArg Multiset.card hM
  have h2 : Multiset.card (L : Multiset ℕ) = Multiset.card ((z + 1) ::ₘ C) := congrArg Multiset.card hL
  simp only [Multiset.coe_card, Multiset.card_cons] at h1 h2
  omega

theorem IsIncM.to_isInc {M L : List ℕ} {z : ℕ}
    (hM : M.Pairwise (· ≥ ·)) (hL : L.Pairwise (· ≥ ·)) (h : IsIncM M L z) :
    IsInc M L z := by
  rcases h with ⟨C, hMC, hLC⟩
  set P := (C.filter (fun w => z < w)).sort (· ≥ ·) with hPdef
  set S := (C.filter (fun w => ¬ z < w)).sort (· ≥ ·) with hSdef
  have hP : ∀ w ∈ P, z + 1 ≤ w := by
    intro w hw
    rw [hPdef, Multiset.mem_sort, Multiset.mem_filter] at hw
    exact hw.2
  have hS : ∀ w ∈ S, w ≤ z := by
    intro w hw
    rw [hSdef, Multiset.mem_sort, Multiset.mem_filter] at hw
    omega
  have hPS_C : ((P ++ S : List ℕ) : Multiset ℕ) = C := by
    rw [← Multiset.coe_add, hPdef, hSdef, Multiset.sort_eq, Multiset.sort_eq, Multiset.filter_add_not]
  have hM_sort : (P ++ (z :: S)).Pairwise (· ≥ ·) := by
    rw [List.pairwise_append]
    refine ⟨Multiset.pairwise_sort _ _, List.pairwise_cons.mpr ⟨hS, Multiset.pairwise_sort _ _⟩, ?_⟩
    intro a ha b hb
    have ha' : z + 1 ≤ a := hP a ha
    rcases List.mem_cons.mp hb with rfl | hb'
    · omega
    · have hb'' := hS b hb'; omega
  have hL_sort : (P ++ ((z + 1) :: S)).Pairwise (· ≥ ·) := by
    rw [List.pairwise_append]
    refine ⟨Multiset.pairwise_sort _ _, List.pairwise_cons.mpr ⟨fun b hb => by have := hS b hb; omega, Multiset.pairwise_sort _ _⟩, ?_⟩
    intro a ha b hb
    have ha' : z + 1 ≤ a := hP a ha
    rcases List.mem_cons.mp hb with rfl | hb'
    · omega
    · have hb'' := hS b hb'; omega
  have hM_eq : M = P ++ (z :: S) := by
    apply List.Perm.eq_of_sortedGE hM.sortedGE hM_sort.sortedGE
    rw [← Multiset.coe_eq_coe, hMC, coe_append_cons, hPS_C]
  have hL_eq : L = P ++ ((z + 1) :: S) := by
    apply List.Perm.eq_of_sortedGE hL.sortedGE hL_sort.sortedGE
    rw [← Multiset.coe_eq_coe, hLC, coe_append_cons, hPS_C]
  exact ⟨P, S, hP, hS, hM_eq, hL_eq⟩

/-- Raw decrement before sorting in `havelHakimiStep`: decrement the first `s` entries of `T` by 1
and keep the remaining entries unchanged. -/
def rawDec (s : ℕ) (T : List ℕ) : List ℕ :=
  (T.take s).map (· - 1) ++ T.drop s

theorem havelHakimiStep_cons_eq_mergeSort_rawDec (s : ℕ) (T : List ℕ) :
    havelHakimiStep (s :: T) = (rawDec s T).mergeSort (· ≥ ·) := by
  simp [havelHakimiStep, rawDec, List.splitAt_eq]

theorem coe_havelHakimiStep (s : ℕ) (T : List ℕ) :
    (havelHakimiStep (s :: T) : Multiset ℕ) = (rawDec s T : Multiset ℕ) := by
  rw [Multiset.coe_eq_coe, havelHakimiStep_cons_eq_mergeSort_rawDec]
  exact List.mergeSort_perm _ _

theorem rawDec_append_cons_of_le (s : ℕ) (P S : List ℕ) (a : ℕ) (hs : s ≤ P.length) :
    rawDec s (P ++ (a :: S)) = ((P.take s).map (· - 1) ++ P.drop s) ++ (a :: S) := by
  unfold rawDec
  rw [List.take_append_of_le_length hs, List.drop_append_of_le_length hs, List.append_assoc]

theorem rawDec_append_cons_of_lt (s : ℕ) (P S : List ℕ) (a : ℕ) (hlt : P.length < s) :
    rawDec s (P ++ (a :: S))
      = P.map (· - 1) ++ ((a - 1) :: ((S.take (s - P.length - 1)).map (· - 1) ++ S.drop (s - P.length - 1))) := by
  unfold rawDec
  have hle : P.length ≤ s := hlt.le
  have hsub : s - P.length = (s - P.length - 1) + 1 := by omega
  rw [List.take_append, List.drop_append, List.take_of_length_le hle,
    List.drop_eq_nil_of_le hle, List.nil_append]
  conv_lhs => rw [hsub, List.take_succ_cons, List.drop_succ_cons, List.map_append, List.map_cons,
    List.append_assoc, List.cons_append]

/-- **Identity 1**: Commuting `havelHakimiStep` with a single increment `IsInc M L z`. -/
theorem isIncM_havelHakimiStep (s : ℕ) (P S : List ℕ) (z : ℕ)
    (hz : P.length < s → 1 ≤ z) :
    IsIncM (havelHakimiStep (s :: (P ++ (z :: S))))
           (havelHakimiStep (s :: (P ++ ((z + 1) :: S))))
           (if P.length < s then z - 1 else z) := by
  rw [IsIncM]
  simp only [coe_havelHakimiStep]
  by_cases hlt : P.length < s
  · rw [if_pos hlt]
    have hz1 : 1 ≤ z := hz hlt
    have hzz : z - 1 + 1 = (z + 1) - 1 := by omega
    refine ⟨((P.map (· - 1) ++ ((S.take (s - P.length - 1)).map (· - 1) ++ S.drop (s - P.length - 1)) : List ℕ) : Multiset ℕ), ?_, ?_⟩
    · rw [rawDec_append_cons_of_lt s P S z hlt, coe_append_cons]
    · rw [hzz, rawDec_append_cons_of_lt s P S (z + 1) hlt, coe_append_cons]
  · rw [if_neg hlt]
    have hs : s ≤ P.length := by omega
    refine ⟨((((P.take s).map (· - 1) ++ P.drop s) ++ S : List ℕ) : Multiset ℕ), ?_, ?_⟩
    · rw [rawDec_append_cons_of_le s P S z hs, coe_append_cons]
    · rw [rawDec_append_cons_of_le s P S (z + 1) hs, coe_append_cons]

/-- **Identity 2**: Comparing `havelHakimiStep (s :: T)` and `havelHakimiStep ((s + 1) :: T)`. -/
theorem isIncM_havelHakimiStep_succ (s : ℕ) (T : List ℕ) (hs : s < T.length)
    (hw : 1 ≤ T[s]) :
    IsIncM (havelHakimiStep ((s + 1) :: T))
           (havelHakimiStep (s :: T))
           (T[s] - 1) := by
  rw [IsIncM]
  simp only [coe_havelHakimiStep]
  set P := T.take s
  set w := T[s]
  set S := T.drop (s + 1)
  have hT : T = P ++ (w :: S) := by
    simp [P, w, S]
  have hPlen : P.length = s := List.length_take_of_le hs.le
  have hle : s ≤ P.length := hPlen.symm.le
  have hlt : P.length < s + 1 := by omega
  have hww : w - 1 + 1 = w := by omega
  refine ⟨((P.map (· - 1) ++ S : List ℕ) : Multiset ℕ), ?_, ?_⟩
  · conv_lhs => rw [hT]
    rw [rawDec_append_cons_of_lt (s + 1) P S w hlt]
    have h0 : s + 1 - P.length - 1 = 0 := by omega
    rw [h0, List.take_zero, List.map_nil, List.drop_zero, List.nil_append, coe_append_cons]
  · conv_lhs => rw [hT]
    rw [rawDec_append_cons_of_le s P S w hle, ← hPlen,
      List.take_of_length_le (le_refl _), List.drop_eq_nil_of_le (le_refl _), List.append_nil,
      hww, coe_append_cons]

theorem IsIncM.eq_of_eq {M A B : List ℕ} {x y : ℕ}
    (hA : A.Pairwise (· ≥ ·)) (hB : B.Pairwise (· ≥ ·))
    (hy : IsIncM M A y) (hx : IsIncM M B x) (hxy : y = x) : A = B := by
  subst hxy
  rcases hy with ⟨C₁, hM₁, hA₁⟩
  rcases hx with ⟨C₂, hM₂, hB₂⟩
  have hC : C₁ = C₂ := by simpa using hM₁.symm.trans hM₂
  subst hC
  apply List.Perm.eq_of_sortedGE hA.sortedGE hB.sortedGE
  rw [← Multiset.coe_eq_coe, hA₁, hB₂]

theorem isIncM_comm {B M A : List ℕ} {v u : ℕ}
    (hBM : IsIncM B M v) (hMA : IsIncM M A u) (huv : u < v) :
    ∃ M' : List ℕ, M'.Pairwise (· ≥ ·) ∧ IsIncM B M' u ∧ IsIncM M' A v := by
  rcases hBM with ⟨C₁, hB, hM₁⟩
  rcases hMA with ⟨C₂, hM₂, hA⟩
  have h12 : (v + 1) ::ₘ C₁ = u ::ₘ C₂ := hM₁.symm.trans hM₂
  have hmem : u ∈ (v + 1) ::ₘ C₁ := by rw [h12]; exact Multiset.mem_cons_self u C₂
  have huC₁ : u ∈ C₁ := by
    rcases Multiset.mem_cons.mp hmem with h | h
    · omega
    · exact h
  set C₀ := C₁.erase u
  have hC₁ : C₁ = u ::ₘ C₀ := (Multiset.cons_erase huC₁).symm
  have hC₂ : C₂ = (v + 1) ::ₘ C₀ := by
    rw [hC₁, Multiset.cons_swap] at h12
    simpa using h12.symm
  set m' : Multiset ℕ := (u + 1) ::ₘ v ::ₘ C₀
  refine ⟨m'.sort (· ≥ ·), Multiset.pairwise_sort _ _, ?_, ?_⟩
  · refine ⟨v ::ₘ C₀, ?_, ?_⟩
    · rw [hB, hC₁, Multiset.cons_swap]
    · rw [Multiset.sort_eq]
  · refine ⟨(u + 1) ::ₘ C₀, ?_, ?_⟩
    · rw [Multiset.sort_eq, Multiset.cons_swap]
    · rw [hA, hC₂, Multiset.cons_swap]

theorem IsInc.le_head_left {m₀ : ℕ} {TM L : List ℕ} {z : ℕ}
    (hM : (m₀ :: TM).Pairwise (· ≥ ·)) (h : IsInc (m₀ :: TM) L z) : z ≤ m₀ := by
  rcases h with ⟨P, S, _, _, hMeq, _⟩
  have hzmem : z ∈ m₀ :: TM := by rw [hMeq]; simp
  rcases List.mem_cons.mp hzmem with rfl | hzt
  · exact le_refl _
  · exact (List.pairwise_cons.mp hM).1 z hzt

theorem IsInc.le_head_right {M : List ℕ} {l₀ : ℕ} {TL : List ℕ} {z : ℕ}
    (hL : (l₀ :: TL).Pairwise (· ≥ ·)) (h : IsInc M (l₀ :: TL) z) : z + 1 ≤ l₀ := by
  rcases h with ⟨P, S, _, _, _, hLeq⟩
  have hzmem : z + 1 ∈ l₀ :: TL := by rw [hLeq]; simp
  rcases List.mem_cons.mp hzmem with hhead | hzt
  · omega
  · exact (List.pairwise_cons.mp hL).1 (z + 1) hzt

theorem IsInc.head_cases {m₀ : ℕ} {TM L : List ℕ} {z : ℕ} (h : IsInc (m₀ :: TM) L z) :
    (z = m₀ ∧ L = (m₀ + 1) :: TM) ∨
    (z + 1 ≤ m₀ ∧ ∃ TL, L = m₀ :: TL ∧ IsInc TM TL z) := by
  rcases h with ⟨P, S, hP, hS, hMeq, hLeq⟩
  cases P with
  | nil =>
    simp only [List.nil_append, List.cons.injEq] at hMeq hLeq
    rcases hMeq with ⟨rfl, rfl⟩
    left
    exact ⟨rfl, hLeq⟩
  | cons p₀ P' =>
    simp only [List.cons_append, List.cons.injEq] at hMeq
    rcases hMeq with ⟨rfl, hTM⟩
    right
    have hm₀ : z + 1 ≤ m₀ := hP m₀ (List.mem_cons_self ..)
    refine ⟨hm₀, P' ++ ((z + 1) :: S), ?_, ?_⟩
    · simpa using hLeq
    · exact ⟨P', S, fun w hw => hP w (List.mem_cons_of_mem m₀ hw), hS, hTM, rfl⟩

/-! ## `countGE`: counting elements at least `v` -/

/-- Number of elements in `l` that are at least `v`. -/
def countGE (v : ℕ) (l : List ℕ) : ℕ :=
  (l.filter (fun w => decide (v ≤ w))).length

theorem countGE_nil (v : ℕ) : countGE v [] = 0 := rfl

theorem countGE_cons (v a : ℕ) (l : List ℕ) :
    countGE v (a :: l) = (if v ≤ a then 1 else 0) + countGE v l := by
  unfold countGE
  by_cases h : v ≤ a <;> simp [h]
  omega

theorem countGE_append (v : ℕ) (l₁ l₂ : List ℕ) :
    countGE v (l₁ ++ l₂) = countGE v l₁ + countGE v l₂ := by
  unfold countGE
  rw [List.filter_append, List.length_append]

theorem countGE_perm (v : ℕ) {l₁ l₂ : List ℕ} (hp : l₁.Perm l₂) :
    countGE v l₁ = countGE v l₂ := by
  unfold countGE
  exact (hp.filter _).length_eq

theorem countGE_havelHakimiStep (v s : ℕ) (T : List ℕ) :
    countGE v (havelHakimiStep (s :: T)) = countGE v (rawDec s T) := by
  rw [havelHakimiStep_cons_eq_mergeSort_rawDec]
  exact countGE_perm v (List.mergeSort_perm _ _)

theorem countGE_eq_length_of_forall (v : ℕ) (l : List ℕ) (h : ∀ w ∈ l, v ≤ w) :
    countGE v l = l.length := by
  unfold countGE
  rw [List.filter_eq_self.mpr]
  intro w hw
  simpa using h w hw

theorem countGE_pred_map_pred_ge (v : ℕ) : ∀ (l : List ℕ),
    countGE v l ≤ countGE (v - 1) (l.map (· - 1))
  | [] => by simp [countGE]
  | a :: t => by
    simp only [List.map_cons, countGE_cons]
    have ih := countGE_pred_map_pred_ge v t
    split_ifs <;> omega

theorem countGE_mono_val {v₁ v₂ : ℕ} (h : v₁ ≤ v₂) : ∀ (l : List ℕ),
    countGE v₂ l ≤ countGE v₁ l
  | [] => by simp [countGE]
  | a :: t => by
    simp only [countGE_cons]
    have ih := countGE_mono_val h t
    split_ifs <;> omega

theorem countGE_pred_rawDec_ge (v s : ℕ) (T : List ℕ) :
    countGE v T ≤ countGE (v - 1) (rawDec s T) := by
  unfold rawDec
  rw [countGE_append]
  have hsplit : countGE v T = countGE v (T.take s) + countGE v (T.drop s) := by
    conv_lhs => rw [← List.take_append_drop s T, countGE_append]
  have h1 := countGE_pred_map_pred_ge v (T.take s)
  have h2 := countGE_mono_val (Nat.sub_le v 1) (T.drop s)
  omega

theorem countGE_rawDec_append_cons_of_le (s : ℕ) (P S : List ℕ) (v : ℕ)
    (hP : ∀ w ∈ P, v + 1 ≤ w) (hs : s ≤ P.length) :
    countGE v (P ++ (v :: S)) ≤ countGE v (rawDec s (P ++ (v :: S))) := by
  rw [rawDec_append_cons_of_le s P S v hs]
  simp only [countGE_append, countGE_cons, le_refl, ↓reduceIte]
  have hP1 : countGE v P = P.length :=
    countGE_eq_length_of_forall v P (fun w hw => by have := hP w hw; omega)
  have hPt : countGE v ((P.take s).map (· - 1)) = (P.take s).length := by
    rw [countGE_eq_length_of_forall, List.length_map]
    intro w hw
    rw [List.mem_map] at hw
    rcases hw with ⟨u, hu, rfl⟩
    have huP := hP u (List.take_subset s P hu)
    omega
  have hPd : countGE v (P.drop s) = (P.drop s).length := by
    apply countGE_eq_length_of_forall
    intro w hw
    have hwP := hP w (List.drop_subset s P hw)
    omega
  have hlen : (P.take s).length + (P.drop s).length = P.length := by
    rw [← List.length_append, List.take_append_drop]
  omega

/-- **Universal `countGE` preservation under Identity 1.** -/
theorem countGE_havelHakimiStep_ge (s : ℕ) (P S : List ℕ) (v : ℕ)
    (hP : ∀ w ∈ P, v + 1 ≤ w) :
    countGE v (P ++ (v :: S)) ≤
      countGE (if P.length < s then v - 1 else v) (havelHakimiStep (s :: (P ++ (v :: S)))) := by
  rw [countGE_havelHakimiStep]
  by_cases hlt : P.length < s
  · rw [if_pos hlt]
    exact countGE_pred_rawDec_ge v s (P ++ (v :: S))
  · rw [if_neg hlt]
    exact countGE_rawDec_append_cons_of_le s P S v hP (by omega)

theorem le_head_of_mem_cons {b₀ : ℕ} {TB : List ℕ} (hB : (b₀ :: TB).Pairwise (· ≥ ·))
    {w : ℕ} (hw : w ∈ b₀ :: TB) : w ≤ b₀ := by
  rcases List.mem_cons.mp hw with rfl | hwt
  · exact le_refl _
  · exact (List.pairwise_cons.mp hB).1 w hwt

theorem pred_le_head_havelHakimiStep_of_mem {k b₀' : ℕ} {TB TB' : List ℕ}
    (hB' : havelHakimiStep (k :: TB) = b₀' :: TB') {w : ℕ} (hw : w ∈ TB) :
    w - 1 ≤ b₀' := by
  have hB'sort : (b₀' :: TB').Pairwise (· ≥ ·) := by
    rw [← hB']
    unfold havelHakimiStep
    simp only [List.splitAt_eq]
    exact List.pairwise_mergeSort' _ _
  have hperm : (b₀' :: TB').Perm (rawDec k TB) := by
    rw [← hB', havelHakimiStep_cons_eq_mergeSort_rawDec]
    exact List.mergeSort_perm _ _
  have hw_split : w ∈ TB.take k ++ TB.drop k := by
    rw [List.take_append_drop]
    exact hw
  rcases List.mem_append.mp hw_split with hwt | hwd
  · have hmem : w - 1 ∈ rawDec k TB := by
      unfold rawDec
      exact List.mem_append_left _ (List.mem_map_of_mem hwt)
    exact le_head_of_mem_cons hB'sort (hperm.mem_iff.mpr hmem)
  · have hmem : w ∈ rawDec k TB := by
      unfold rawDec
      exact List.mem_append_right _ hwd
    have hle := le_head_of_mem_cons hB'sort (hperm.mem_iff.mpr hmem)
    omega

theorem z_prime_le_of_forall_pred_le {s b₀' : ℕ} {P S : List ℕ} {z : ℕ}
    (hs : 1 ≤ s) (hP : ∀ w ∈ P, z + 1 ≤ w)
    (hpred : ∀ w ∈ P ++ (z :: S), w - 1 ≤ b₀') :
    (if P.length < s then z - 1 else z) ≤ b₀' := by
  by_cases hlt : P.length < s
  · rw [if_pos hlt]
    exact hpred z (by simp)
  · rw [if_neg hlt]
    cases P with
    | nil => simp at hlt; omega
    | cons p₀ P' =>
      have hp₀ : z + 1 ≤ p₀ := hP p₀ (List.mem_cons_self ..)
      have hpred₀ : p₀ - 1 ≤ b₀' := hpred p₀ (by simp)
      omega

theorem countGE_pos_head_ge {v b₀ : ℕ} {TB : List ℕ}
    (hB : (b₀ :: TB).Pairwise (· ≥ ·)) (hpos : 1 ≤ countGE v (b₀ :: TB)) :
    v ≤ b₀ := by
  unfold countGE at hpos
  obtain ⟨w, hw⟩ : ∃ w, w ∈ (b₀ :: TB).filter (fun w => decide (v ≤ w)) := by
    cases hf : (b₀ :: TB).filter (fun w => decide (v ≤ w)) with
    | nil => rw [hf] at hpos; simp at hpos
    | cons a l => exact ⟨a, List.mem_cons_self ..⟩
  rw [List.mem_filter, decide_eq_true_eq] at hw
  have hwb := le_head_of_mem_cons hB hw.1
  omega

theorem le_getElem_of_countGE {v : ℕ} : ∀ {T : List ℕ}, T.Pairwise (· ≥ ·) →
    ∀ {s : ℕ}, s < countGE v T → ∃ hs : s < T.length, v ≤ T[s]
  | [], _, s, hs => by simp [countGE] at hs
  | a :: T', hT, s, hs => by
    rw [countGE_cons] at hs
    obtain ⟨haT', hT'⟩ := List.pairwise_cons.mp hT
    by_cases hva : v ≤ a
    · rw [if_pos hva] at hs
      cases s with
      | zero => exact ⟨by simp, hva⟩
      | succ s' =>
        obtain ⟨hs', hv'⟩ := le_getElem_of_countGE hT' (by omega : s' < countGE v T')
        exact ⟨Nat.succ_lt_succ hs', hv'⟩
    · rw [if_neg hva] at hs
      have hzero : countGE v T' = 0 := by
        unfold countGE
        rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
        intro w hw
        have hwa := haT' w hw
        simp; omega
      omega

theorem countGE_ge_of_getElem {v : ℕ} : ∀ {T : List ℕ}, T.Pairwise (· ≥ ·) →
    ∀ {s : ℕ} (hs : s < T.length), v ≤ T[s] → s + 1 ≤ countGE v T
  | [], _, s, hs, _ => by simp at hs
  | a :: T', hT, s, hs, hv => by
    rw [countGE_cons]
    obtain ⟨haT', hT'⟩ := List.pairwise_cons.mp hT
    cases s with
    | zero =>
      simp only [List.getElem_cons_zero] at hv
      rw [if_pos hv]
      omega
    | succ s' =>
      simp only [List.getElem_cons_succ] at hv
      have hs' : s' < T'.length := Nat.lt_of_succ_lt_succ hs
      have hmem : T'[s'] ∈ T' := List.getElem_mem hs'
      have hva : v ≤ a := le_trans hv (haT' _ hmem)
      rw [if_pos hva]
      have ih := countGE_ge_of_getElem hT' hs' hv
      omega

theorem countGE_pos_of_mem {v w : ℕ} {l : List ℕ} (hw : w ∈ l) (hvw : v ≤ w) :
    1 ≤ countGE v l := by
  induction l with
  | nil => simp at hw
  | cons a t ih =>
    rw [countGE_cons]
    rcases List.mem_cons.mp hw with rfl | hwt
    · rw [if_pos hvw]; omega
    · have := ih hwt; split_ifs <;> omega

theorem getElem_le_of_mem_take {T : List ℕ} {k : ℕ} (hT : T.Pairwise (· ≥ ·))
    (hk : k < T.length) {u : ℕ} (hu : u ∈ T.take (k + 1)) : T[k] ≤ u := by
  induction T generalizing k with
  | nil => simp at hk
  | cons a T' ih =>
    obtain ⟨haT', hT'⟩ := List.pairwise_cons.mp hT
    cases k with
    | zero =>
      simp only [List.take_succ_cons, List.take_zero, List.mem_singleton] at hu
      subst hu; simp
    | succ k' =>
      rw [List.take_succ_cons, List.mem_cons] at hu
      have hk' : k' < T'.length := Nat.lt_of_succ_lt_succ hk
      simp only [List.getElem_cons_succ]
      rcases hu with rfl | hu'
      · exact haT' _ (List.getElem_mem hk')
      · exact ih hT' hk' hu'

theorem havelHakimiStep_eq_of_prefix_length_eq (t : ℕ) (P S : List ℕ) (v : ℕ)
    (ht : P.length = t) :
    havelHakimiStep ((t + 1) :: (P ++ ((v + 1) :: S))) =
      havelHakimiStep (t :: (P ++ (v :: S))) := by
  have hraw : rawDec (t + 1) (P ++ ((v + 1) :: S)) = rawDec t (P ++ (v :: S)) := by
    rw [rawDec_append_cons_of_lt (t + 1) P S (v + 1) (by omega),
        rawDec_append_cons_of_le t P S v (by omega)]
    have h0 : t + 1 - P.length - 1 = 0 := by omega
    rw [h0, ← ht, List.take_of_length_le (le_refl _), List.drop_eq_nil_of_le (le_refl _)]
    simp
  rw [havelHakimiStep_cons_eq_mergeSort_rawDec, havelHakimiStep_cons_eq_mergeSort_rawDec, hraw]

theorem countGE_havelHakimiStep_succ_ge_head_add_two {s b₀' : ℕ} {TB TB' : List ℕ}
    (hTB : TB.Pairwise (· ≥ ·)) (hTB_le : ∀ w ∈ TB, w ≤ s) (hs1 : 1 ≤ s)
    (hs : s < TB.length) (hw : 1 ≤ TB[s])
    (hB' : havelHakimiStep ((s + 1) :: TB) = b₀' :: TB') :
    b₀' + 2 ≤ countGE (TB[s] - 1) (b₀' :: TB') := by
  set v := TB[s] - 1
  have hv1 : v + 1 = TB[s] := by omega
  have hge1 : s + 1 ≤ countGE (v + 1) TB :=
    countGE_ge_of_getElem hTB hs (by omega)
  have hge2 : countGE (v + 1) TB ≤ countGE v (b₀' :: TB') := by
    rw [← hB', countGE_havelHakimiStep]
    simpa [v] using countGE_pred_rawDec_ge (v + 1) (s + 1) TB
  have hge : s + 1 ≤ countGE v (b₀' :: TB') := le_trans hge1 hge2
  have hB'sort : (b₀' :: TB').Pairwise (· ≥ ·) := by
    rw [← hB']; unfold havelHakimiStep; simp only [List.splitAt_eq]; exact List.pairwise_mergeSort' _ _
  have hvb₀' : v ≤ b₀' := countGE_pos_head_ge hB'sort (by omega)
  have hperm : (b₀' :: TB').Perm (rawDec (s + 1) TB) := by
    rw [← hB', havelHakimiStep_cons_eq_mergeSort_rawDec]; exact List.mergeSort_perm _ _
  have hb₀'_mem : b₀' ∈ rawDec (s + 1) TB := hperm.mem_iff.mp (List.mem_cons_self ..)
  unfold rawDec at hb₀'_mem
  rcases List.mem_append.mp hb₀'_mem with htake | hdrop
  · rw [List.mem_map] at htake
    rcases htake with ⟨u, hu, rfl⟩
    have hus : u ≤ s := hTB_le u (List.take_subset _ _ hu)
    omega
  · have hb₀'s : b₀' ≤ s := hTB_le b₀' (List.drop_subset _ _ hdrop)
    have htake_eq : countGE v ((TB.take (s + 1)).map (· - 1)) = s + 1 := by
      have hlen : ((TB.take (s + 1)).map (· - 1)).length = s + 1 := by
        rw [List.length_map, List.length_take_of_le (by omega)]
      rw [countGE_eq_length_of_forall, hlen]
      intro x hx
      rw [List.mem_map] at hx
      rcases hx with ⟨u, hu, rfl⟩
      have hu_ge : TB[s] ≤ u := getElem_le_of_mem_take hTB hs hu
      omega
    have hdrop_pos : 1 ≤ countGE v (TB.drop (s + 1)) :=
      countGE_pos_of_mem hdrop hvb₀'
    have htot : countGE v (b₀' :: TB') =
        countGE v ((TB.take (s + 1)).map (· - 1)) + countGE v (TB.drop (s + 1)) := by
      rw [← hB', countGE_havelHakimiStep, rawDec, countGE_append]
    omega

theorem countGE_havelHakimiStep_same_ge_head_add_two {t v b₀' : ℕ} {TB P S TB' : List ℕ}
    (hB : (t :: TB).Pairwise (· ≥ ·))
    (hTBeq : TB = P ++ (v :: S)) (hP : ∀ w ∈ P, v + 1 ≤ w) (hS : ∀ w ∈ S, w ≤ v)
    (hvt : v + 1 ≤ t) (hcnt : t + 2 ≤ countGE v (t :: TB))
    (hB' : havelHakimiStep (t :: TB) = b₀' :: TB') :
    b₀' + 2 ≤ countGE (if P.length < t then v - 1 else v) (b₀' :: TB') := by
  set v' := if P.length < t then v - 1 else v
  have hTB_le : ∀ w ∈ TB, w ≤ t := (List.pairwise_cons.mp hB).1
  have hcnt_TB : t + 1 ≤ countGE v TB := by
    rw [countGE_cons, if_pos (by omega)] at hcnt
    omega
  have hge_v' : countGE v TB ≤ countGE v' (b₀' :: TB') := by
    rw [← hB', hTBeq]
    exact countGE_havelHakimiStep_ge t P S v hP
  have hperm : (b₀' :: TB').Perm (rawDec t TB) := by
    rw [← hB', havelHakimiStep_cons_eq_mergeSort_rawDec]; exact List.mergeSort_perm _ _
  have hb₀'_mem : b₀' ∈ rawDec t TB := hperm.mem_iff.mp (List.mem_cons_self ..)
  unfold rawDec at hb₀'_mem
  rcases List.mem_append.mp hb₀'_mem with htake | hdrop
  · rw [List.mem_map] at htake
    rcases htake with ⟨u, hu, rfl⟩
    have hut : u ≤ t := hTB_le u (List.take_subset _ _ hu)
    omega
  · by_cases hPlt : P.length ≤ t
    · rw [hTBeq, List.drop_append] at hdrop
      have hPe : P.drop t = [] := List.drop_eq_nil_of_le hPlt
      rw [hPe, List.nil_append] at hdrop
      have hmem_vS : b₀' ∈ v :: S := List.drop_subset _ _ hdrop
      have hb₀'v : b₀' ≤ v := by
        rcases List.mem_cons.mp hmem_vS with rfl | hs
        · exact le_refl _
        · exact hS b₀' hs
      omega
    · have hcnt_P : countGE v TB ≥ P.length + 1 := by
        rw [hTBeq, countGE_append, countGE_cons, if_pos (le_refl _),
          countGE_eq_length_of_forall v P (fun w hw => by have := hP w hw; omega)]
        omega
      have hb₀'t : b₀' ≤ t := hTB_le b₀' (List.drop_subset _ _ hdrop)
      omega

theorem havelHakimiStep_pairwise' (l : List ℕ) : (havelHakimiStep l).Pairwise (· ≥ ·) := by
  cases l with
  | nil => simp [havelHakimiStep]
  | cons d rest =>
    unfold havelHakimiStep
    simp only [List.splitAt_eq]
    exact List.pairwise_mergeSort' _ _

theorem graphical_cons_take_pos {d : ℕ} {rest₀ : List ℕ}
    (hsort₀ : (d :: rest₀).Pairwise (· ≥ ·))
    (hg : Graphical ((d :: rest₀ : List ℕ) : Multiset ℕ)) :
    d ≤ rest₀.length ∧ ∀ x ∈ rest₀.take d, 1 ≤ x := by
  classical
  obtain ⟨V, hFin, hDec, G, hAdj, hGd⟩ := hg
  rw [Subsingleton.elim hAdj (Classical.decRel G.Adj)] at hGd
  clear hAdj
  have hcardV : Fintype.card V = rest₀.length + 1 := by
    have h1 : Fintype.card V = Multiset.card G.degreeMultiset := by
      rw [SimpleGraph.degreeMultiset, Multiset.card_map]; rfl
    rw [h1, hGd, Multiset.coe_card, List.length_cons]
  haveI : Nonempty V := Fintype.card_pos_iff.mp (by rw [hcardV]; omega)
  obtain ⟨v, -, hv₀⟩ :=
    Finset.exists_max_image Finset.univ (fun w => G.degree w) Finset.univ_nonempty
  have hv : ∀ w, G.degree w ≤ G.degree v := fun w => hv₀ w (Finset.mem_univ w)
  set l : List (ℕ × ℕ) :=
    (Finset.univ.erase v).toList.map (fun w => (G.degree w, if G.Adj w v then 1 else 0)) with hl
  have hbin_l : ∀ p ∈ l, p.2 ≤ 1 := by
    intro p hp; rw [hl, List.mem_map] at hp
    obtain ⟨w, _, rfl⟩ := hp; dsimp only; split_ifs <;> simp
  have hpair_l : ∀ p ∈ l, p.2 ≤ p.1 := by
    intro p hp; rw [hl, List.mem_map] at hp
    obtain ⟨w, _, rfl⟩ := hp; dsimp only
    split_ifs with hadj
    · exact (G.degree_pos_iff_exists_adj w).mpr ⟨v, hadj⟩
    · simp
  obtain ⟨rest, e, hsort, hbin, hlen, hesum, hrestperm, -, hForall⟩ :=
    pairsort_align l hbin_l hpair_l
  have hΔ : e.sum = G.degree v := by
    rw [hesum, hl, List.map_map]; exact sum_adj_indicator_eq_degree G v
  have hrest_eq : rest = ((Finset.univ.erase v).val.map (fun w => G.degree w)).sort (· ≥ ·) := by
    refine List.Perm.eq_of_sortedGE hsort.sortedGE (Multiset.pairwise_sort _ _).sortedGE ?_
    refine hrestperm.trans ?_
    rw [hl, List.map_map]
    exact (sort_map_perm_toList_map (Finset.univ.erase v) (fun w => G.degree w)).symm
  have hsG : (G.degreeMultiset).sort (· ≥ ·) = G.degree v :: rest := by
    rw [hrest_eq]; exact sortedDeg_cons G v hv
  have hcons_eq : G.degree v :: rest = d :: rest₀ := by
    rw [← hsG]
    apply List.Perm.eq_of_sortedGE (Multiset.pairwise_sort _ _).sortedGE hsort₀.sortedGE
    rw [← Multiset.coe_eq_coe, Multiset.sort_eq, hGd]
  injection hcons_eq with hd_eq hrest₀_eq
  have hk : e.sum ≤ rest.length := by rw [← hlen]; exact sum_le_length_of_le_one e hbin
  have hpos : ∀ x ∈ rest.take e.sum, 1 ≤ x := take_sum_pos rest e hsort hForall hbin
  rw [hΔ, hd_eq, hrest₀_eq] at hk hpos
  exact ⟨hk, hpos⟩

theorem pos_of_prefix_lt_of_take_pos {d : ℕ} {P S : List ℕ} {z : ℕ}
    (hpos : ∀ x ∈ (P ++ (z :: S)).take d, 1 ≤ x) (hlt : P.length < d) : 1 ≤ z := by
  have hsub : d - P.length = (d - P.length - 1) + 1 := by omega
  rw [List.take_append, hsub, List.take_succ_cons] at hpos
  exact hpos z (by simp)

theorem getElem_pos_of_take_pos {d s : ℕ} {TB : List ℕ}
    (hsd : s < d) (hd : d ≤ TB.length) (hpos : ∀ x ∈ TB.take d, 1 ≤ x) :
    1 ≤ TB[s]'(by omega) := by
  have hlt : s < (TB.take d).length := by rw [List.length_take_of_le hd]; exact hsd
  have hmem : (TB.take d)[s] ∈ TB.take d := List.getElem_mem hlt
  have heq : (TB.take d)[s] = TB[s]'(by omega) := List.getElem_take
  rw [heq] at hmem
  exact hpos _ hmem

theorem countGE_one_ge_of_take_pos {d : ℕ} {T : List ℕ}
    (hd : d ≤ T.length) (hpos : ∀ x ∈ T.take d, 1 ≤ x) :
    d ≤ countGE 1 T := by
  conv_rhs => rw [← List.take_append_drop d T, countGE_append]
  rw [countGE_eq_length_of_forall 1 (T.take d) hpos, List.length_take_of_le hd]
  omega

theorem countGE_one_le_of_isInc {M L : List ℕ} {z : ℕ} (h : IsInc M L z) :
    countGE 1 M ≤ countGE 1 L := by
  rcases h with ⟨P, S, _, _, rfl, rfl⟩
  simp only [countGE_append, countGE_cons]
  split_ifs <;> omega

theorem countGE_one_eq_of_isInc_pos {M L : List ℕ} {z : ℕ} (h : IsInc M L z) (hz : 1 ≤ z) :
    countGE 1 M = countGE 1 L := by
  rcases h with ⟨P, S, _, _, rfl, rfl⟩
  simp only [countGE_append, countGE_cons]
  rw [if_pos hz, if_pos (by omega)]

theorem pos_of_prefix_lt_of_countGE_one {d : ℕ} {P S : List ℕ} {z : ℕ}
    (hS : ∀ w ∈ S, w ≤ z) (hcnt : d ≤ countGE 1 (P ++ (z :: S))) (hlt : P.length < d) :
    1 ≤ z := by
  by_contra hz0
  have hz : z = 0 := by omega
  subst hz
  rw [countGE_append, countGE_cons, if_neg (by omega)] at hcnt
  have hS0 : countGE 1 S = 0 := by
    unfold countGE
    rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
    intro w hw
    have := hS w hw
    simp; omega
  have hP_le : countGE 1 P ≤ P.length := List.length_filter_le _ _
  omega

theorem getElem_mem_prefix_of_lt {P S : List ℕ} {v t : ℕ} (ht : t < P.length) :
    (P ++ (v :: S))[t]'(by simp; omega) ∈ P := by
  have heq : (P ++ (v :: S))[t]'(by simp; omega) = P[t] := List.getElem_append_left ht
  rw [heq]
  exact List.getElem_mem ht

theorem getElem_le_head_havelHakimiStep {t b₀' : ℕ} {TB TB' : List ℕ}
    (ht : t < TB.length) (hB' : havelHakimiStep (t :: TB) = b₀' :: TB') :
    TB[t] ≤ b₀' := by
  have hB'sort : (b₀' :: TB').Pairwise (· ≥ ·) := by
    rw [← hB']; exact havelHakimiStep_pairwise' _
  have hperm : (b₀' :: TB').Perm (rawDec t TB) := by
    rw [← hB', havelHakimiStep_cons_eq_mergeSort_rawDec]
    exact List.mergeSort_perm _ _
  have hmem_drop : TB[t] ∈ TB.drop t := by
    have hlt : 0 < (TB.drop t).length := by simp; omega
    have h0 : (TB.drop t)[0] ∈ TB.drop t := List.getElem_mem hlt
    simpa using h0
  have hmem_raw : TB[t] ∈ rawDec t TB := by
    unfold rawDec
    exact List.mem_append_right _ hmem_drop
  exact le_head_of_mem_cons hB'sort (hperm.mem_iff.mpr hmem_raw)

/-! ## The 2-state induction machine (`State1` and `State2`) -/

def State1 (A B : List ℕ) : Prop :=
  ∃ (M : List ℕ) (y x : ℕ),
    M.Pairwise (· ≥ ·) ∧ y ≤ x ∧ IsIncM M A y ∧ IsIncM M B x

def State2 (A B : List ℕ) : Prop :=
  ∃ (M : List ℕ) (v u : ℕ),
    M.Pairwise (· ≥ ·) ∧ v ≤ u ∧ IsIncM B M v ∧ IsIncM M A u ∧
    (∀ b₀ TB, B = b₀ :: TB → u ≤ b₀ ∧ b₀ + 2 ≤ countGE v (b₀ :: TB))

theorem head_pos_of_isIncM_right {X : List ℕ} {a₀ : ℕ} {TA : List ℕ} {z : ℕ}
    (hA : (a₀ :: TA).Pairwise (· ≥ ·)) (h : IsIncM X (a₀ :: TA) z) : 1 ≤ a₀ := by
  rcases h with ⟨C, _, hAC⟩
  have hmem : z + 1 ∈ (a₀ :: TA : Multiset ℕ) := by
    rw [hAC]; exact Multiset.mem_cons_self (z + 1) C
  rw [Multiset.mem_coe] at hmem
  have hle := le_head_of_mem_cons hA hmem
  omega

theorem step_state1 {a₀ b₀ : ℕ} {TA TB : List ℕ}
    (hA : (a₀ :: TA).Pairwise (· ≥ ·)) (hB : (b₀ :: TB).Pairwise (· ≥ ·))
    (hgA : Graphical ((a₀ :: TA : List ℕ) : Multiset ℕ))
    (hgB : Graphical ((b₀ :: TB : List ℕ) : Multiset ℕ))
    (hS1 : State1 (a₀ :: TA) (b₀ :: TB)) :
    (a₀ :: TA = b₀ :: TB) ∨
    State1 (havelHakimiStep (a₀ :: TA)) (havelHakimiStep (b₀ :: TB)) ∨
    State2 (havelHakimiStep (a₀ :: TA)) (havelHakimiStep (b₀ :: TB)) := by
  rcases hS1 with ⟨M, y, x, hM, hyx, hyM, hxM⟩
  by_cases hxy : y = x
  · left; exact IsIncM.eq_of_eq hA hB hyM hxM hxy
  · right
    have hyx_lt : y < x := by omega
    have ha₀1 : 1 ≤ a₀ := head_pos_of_isIncM_right hA hyM
    have hMlen : M.length = (a₀ :: TA).length := hyM.length_eq
    cases M with
    | nil => simp at hMlen
    | cons s TM =>
      have hy := hyM.to_isInc hM hA
      have hx := hxM.to_isInc hM hB
      have hxs : x ≤ s := hx.le_head_left hM
      rcases hy.head_cases with ⟨hys, _⟩ | ⟨hy1s, TA', hAeq, hyT⟩
      · omega
      · injection hAeq with hs_a₀ hTA'
        subst a₀; subst TA'
        rcases hx.head_cases with ⟨hxs_eq, hBeq⟩ | ⟨_, TB', hBeq, hxT⟩
        · -- Transition 1b: x = s, B = (s + 1) :: TM
          injection hBeq with hb₀_eq hTM_TB
          subst b₀; subst TM
          right
          obtain ⟨hdB, hposB⟩ := graphical_cons_take_pos hB hgB
          have hs_len : s < TB.length := by omega
          have hw : 1 ≤ TB[s] := getElem_pos_of_take_pos (by omega : s < s + 1) hdB hposB
          have hcntB : s + 1 ≤ countGE 1 TB := countGE_one_ge_of_take_pos hdB hposB
          rcases hyT with ⟨Py, Sy, hPy, hSy, hTBy, hTAy⟩
          have hy_pos : Py.length < s → 1 ≤ y :=
            pos_of_prefix_lt_of_countGE_one hSy (by rw [← hTBy]; omega)
          set v₀ := TB[s] - 1
          set u₀ := if Py.length < s then y - 1 else y
          have hv₀ : IsIncM (havelHakimiStep ((s + 1) :: TB)) (havelHakimiStep (s :: TB)) v₀ :=
            isIncM_havelHakimiStep_succ s TB hs_len hw
          have hu₀ : IsIncM (havelHakimiStep (s :: TB)) (havelHakimiStep (s :: TA)) u₀ := by
            have h := isIncM_havelHakimiStep s Py Sy y hy_pos
            rwa [← hTBy, ← hTAy] at h
          have hTB_sort : TB.Pairwise (· ≥ ·) := (List.pairwise_cons.mp hB).2
          have hTA_le : ∀ w ∈ TA, w ≤ s := (List.pairwise_cons.mp hA).1
          have hTB_le : ∀ w ∈ TB, w ≤ s := by
            intro w hw
            rw [hTBy, List.mem_append, List.mem_cons] at hw
            rcases hw with hwP | hwy | hwS
            · exact hTA_le w (by rw [hTAy]; exact List.mem_append_left _ hwP)
            · omega
            · exact hTA_le w (by rw [hTAy]; exact List.mem_append_right _ (List.mem_cons_of_mem _ hwS))
          have hB'_props : ∀ b₀' TB', havelHakimiStep ((s + 1) :: TB) = b₀' :: TB' →
              v₀ ≤ b₀' ∧ u₀ ≤ b₀' ∧ b₀' + 2 ≤ countGE v₀ (b₀' :: TB') := by
            intro b₀' TB' hB'
            have hpred : ∀ w ∈ TB, w - 1 ≤ b₀' :=
              fun w hw => pred_le_head_havelHakimiStep_of_mem hB' hw
            have hv₀_le : v₀ ≤ b₀' := hpred TB[s] (List.getElem_mem hs_len)
            have hu₀_le : u₀ ≤ b₀' :=
              z_prime_le_of_forall_pred_le ha₀1 hPy (fun w hw => hpred w (hTBy ▸ hw))
            have hcnt' : b₀' + 2 ≤ countGE v₀ (b₀' :: TB') :=
              countGE_havelHakimiStep_succ_ge_head_add_two hTB_sort hTB_le ha₀1 hs_len hw hB'
            exact ⟨hv₀_le, hu₀_le, hcnt'⟩
          by_cases hvu : v₀ ≤ u₀
          · exact ⟨havelHakimiStep (s :: TB), v₀, u₀, havelHakimiStep_pairwise' _, hvu, hv₀, hu₀,
              fun b₀' TB' hB' => ⟨(hB'_props b₀' TB' hB').2.1, (hB'_props b₀' TB' hB').2.2⟩⟩
          · obtain ⟨M_swap, hMs, hu₀', hv₀'⟩ := isIncM_comm hv₀ hu₀ (by omega)
            refine ⟨M_swap, u₀, v₀, hMs, by omega, hu₀', hv₀', fun b₀' TB' hB' => ?_⟩
            obtain ⟨hv₀_le, _, hcnt'⟩ := hB'_props b₀' TB' hB'
            have hmono := countGE_mono_val (by omega : u₀ ≤ v₀) (b₀' :: TB')
            exact ⟨hv₀_le, le_trans hcnt' hmono⟩
        · -- Transition 1a: B = s :: TB with IsInc TM TB x
          injection hBeq with hb₀_eq hTB'
          subst b₀; subst TB'
          left
          obtain ⟨hdB, hposB⟩ := graphical_cons_take_pos hB hgB
          have hcntB : s ≤ countGE 1 TB := countGE_one_ge_of_take_pos hdB hposB
          have hcntM : s ≤ countGE 1 TM := by
            rw [countGE_one_eq_of_isInc_pos hxT (by omega)]
            exact hcntB
          rcases hyT with ⟨Py, Sy, hPy, hSy, hTMy, hTAy⟩
          rcases hxT with ⟨Px, Sx, hPx, hSx, hTMx, hTBx⟩
          have hy_pos : Py.length < s → 1 ≤ y :=
            pos_of_prefix_lt_of_countGE_one hSy (hTMy ▸ hcntM)
          have hx_pos : Px.length < s → 1 ≤ x := fun _ => by omega
          have hy' := isIncM_havelHakimiStep s Py Sy y hy_pos
          rw [← hTMy, ← hTAy] at hy'
          have hx' := isIncM_havelHakimiStep s Px Sx x hx_pos
          rw [← hTMx, ← hTBx] at hx'
          refine ⟨havelHakimiStep (s :: TM), _, _, havelHakimiStep_pairwise' _, ?_, hy', hx'⟩
          split_ifs <;> omega

theorem state2_unpack {a₀ t v u : ℕ} {TA TB M : List ℕ}
    (hA : (a₀ :: TA).Pairwise (· ≥ ·)) (hB : (t :: TB).Pairwise (· ≥ ·))
    (hM : M.Pairwise (· ≥ ·)) (hvu : v ≤ u) (hut : u ≤ t)
    (hBM : IsIncM (t :: TB) M v) (hMA : IsIncM M (a₀ :: TA) u) :
    (a₀ = t + 1 ∧ IsInc TB TA v) ∨
    (a₀ = t ∧ ∃ M_T, M = t :: M_T ∧ v + 1 ≤ t ∧ IsInc TB M_T v ∧ IsInc M_T TA u) := by
  have hBM_i := hBM.to_isInc hB hM
  have hMA_i := hMA.to_isInc hM hA
  rcases hBM_i.head_cases with ⟨rfl, rfl⟩ | ⟨hvt, M_T, rfl, hBM_T⟩
  · have huv_eq : u = v := by omega
    subst huv_eq
    rcases hMA_i.head_cases with ⟨huv1, _⟩ | ⟨_, TA', hAeq, hTB_TA⟩
    · omega
    · injection hAeq with ha₀ hTA'
      subst a₀; subst TA'
      exact Or.inl ⟨rfl, hTB_TA⟩
  · rcases hMA_i.head_cases with ⟨rfl, hAeq⟩ | ⟨_, TA', hAeq, hMT_TA⟩
    · injection hAeq with ha₀ hTA'
      subst a₀; subst TA
      exact Or.inl ⟨rfl, hBM_T⟩
    · injection hAeq with ha₀ hTA'
      subst a₀; subst TA'
      exact Or.inr ⟨rfl, M_T, rfl, hvt, hBM_T, hMT_TA⟩

theorem step_state2 {a₀ t : ℕ} {TA TB : List ℕ}
    (hA : (a₀ :: TA).Pairwise (· ≥ ·)) (hB : (t :: TB).Pairwise (· ≥ ·))
    (hgB : Graphical ((t :: TB : List ℕ) : Multiset ℕ))
    (ht1 : 1 ≤ t)
    (hS2 : State2 (a₀ :: TA) (t :: TB)) :
    (havelHakimiStep (a₀ :: TA) = havelHakimiStep (t :: TB)) ∨
    State1 (havelHakimiStep (a₀ :: TA)) (havelHakimiStep (t :: TB)) ∨
    State2 (havelHakimiStep (a₀ :: TA)) (havelHakimiStep (t :: TB)) := by
  rcases hS2 with ⟨M, v, u, hM, hvu, hBM, hMA, hcond⟩
  obtain ⟨hut, hcnt⟩ := hcond t TB rfl
  obtain ⟨hdB, hposB⟩ := graphical_cons_take_pos hB hgB
  have hcnt1_TB : t ≤ countGE 1 TB := countGE_one_ge_of_take_pos hdB hposB
  have hTB_sort : TB.Pairwise (· ≥ ·) := (List.pairwise_cons.mp hB).2
  have hcnt_TB : t + 1 ≤ countGE v TB := by
    rw [countGE_cons, if_pos (by omega)] at hcnt
    omega
  obtain ⟨ht_len, hvt_el⟩ := le_getElem_of_countGE hTB_sort (by omega : t < countGE v TB)
  rcases state2_unpack hA hB hM hvu hut hBM hMA with ⟨ha₀, hTB_TA⟩ | ⟨ha₀, M_T, hM_eq, hvt, hBM_T, hMT_A⟩
  · -- Case 2a: a₀ = t + 1 and IsInc TB TA v
    subst a₀
    rcases hTB_TA with ⟨P, S, hP, hS, hTBv, hTAv⟩
    rcases lt_trichotomy P.length t with hlt | heq | hgt
    · -- P.length < t
      have hv1 : 1 ≤ v := pos_of_prefix_lt_of_countGE_one hS (hTBv ▸ hcnt1_TB) hlt
      have hw1 : 1 ≤ TB[t] := by omega
      have hv' := isIncM_havelHakimiStep (t + 1) P S v (fun _ => hv1)
      rw [if_pos (by omega : P.length < t + 1), ← hTBv, ← hTAv] at hv'
      have hw' := isIncM_havelHakimiStep_succ t TB ht_len hw1
      right; left
      exact ⟨havelHakimiStep ((t + 1) :: TB), v - 1, TB[t] - 1, havelHakimiStep_pairwise' _,
        by omega, hv', hw'⟩
    · -- P.length = t
      left
      rw [hTAv, hTBv]
      exact havelHakimiStep_eq_of_prefix_length_eq t P S v heq
    · -- t < P.length
      have hmemP : TB[t] ∈ P := by simp only [hTBv]; exact getElem_mem_prefix_of_lt hgt
      have hvt1 : v + 1 ≤ TB[t] := hP _ hmemP
      have hv' := isIncM_havelHakimiStep (t + 1) P S v (fun h => by omega)
      rw [if_neg (by omega : ¬ P.length < t + 1), ← hTBv, ← hTAv] at hv'
      have hw' := isIncM_havelHakimiStep_succ t TB ht_len (by omega)
      right; left
      exact ⟨havelHakimiStep ((t + 1) :: TB), v, TB[t] - 1, havelHakimiStep_pairwise' _,
        by omega, hv', hw'⟩
  · -- Case 2b: a₀ = t, M = t :: M_T, v + 1 ≤ t, IsInc TB M_T v, IsInc M_T TA u
    subst a₀; subst M
    right; right
    have hcnt1_MT : t ≤ countGE 1 M_T := le_trans hcnt1_TB (countGE_one_le_of_isInc hBM_T)
    rcases hBM_T with ⟨Pv, Sv, hPv, hSv, hTBv, hMTv⟩
    rcases hMT_A with ⟨Pu, Su, hPu, hSu, hMTu, hTAu⟩
    have hv_pos : Pv.length < t → 1 ≤ v :=
      pos_of_prefix_lt_of_countGE_one hSv (hTBv ▸ hcnt1_TB)
    have hu_pos : Pu.length < t → 1 ≤ u :=
      pos_of_prefix_lt_of_countGE_one hSu (hMTu ▸ hcnt1_MT)
    set v' := if Pv.length < t then v - 1 else v
    set u' := if Pu.length < t then u - 1 else u
    have hv' : IsIncM (havelHakimiStep (t :: TB)) (havelHakimiStep (t :: M_T)) v' := by
      have h := isIncM_havelHakimiStep t Pv Sv v hv_pos
      rwa [← hTBv, ← hMTv] at h
    have hu' : IsIncM (havelHakimiStep (t :: M_T)) (havelHakimiStep (t :: TA)) u' := by
      have h := isIncM_havelHakimiStep t Pu Su u hu_pos
      rwa [← hMTu, ← hTAu] at h
    have hB'_props : ∀ b₀' TB', havelHakimiStep (t :: TB) = b₀' :: TB' →
        v' ≤ b₀' ∧ u' ≤ b₀' ∧ b₀' + 2 ≤ countGE v' (b₀' :: TB') := by
      intro b₀' TB' hB'
      have hcnt' : b₀' + 2 ≤ countGE v' (b₀' :: TB') :=
        countGE_havelHakimiStep_same_ge_head_add_two hB hTBv hPv hSv hvt hcnt hB'
      have hv_le_b₀' : v ≤ b₀' :=
        le_trans hvt_el (getElem_le_head_havelHakimiStep ht_len hB')
      have hv'_le : v' ≤ b₀' := by dsimp [v']; split_ifs <;> omega
      have hpred_TB : ∀ w ∈ TB, w - 1 ≤ b₀' :=
        fun w hw => pred_le_head_havelHakimiStep_of_mem hB' hw
      have hpred_MT : ∀ w ∈ M_T, w - 1 ≤ b₀' := by
        intro w hw
        rw [hMTv, List.mem_append, List.mem_cons] at hw
        rcases hw with hwP | rfl | hwS
        · exact hpred_TB w (by rw [hTBv]; exact List.mem_append_left _ hwP)
        · omega
        · exact hpred_TB w (by rw [hTBv]; exact List.mem_append_right _ (List.mem_cons_of_mem _ hwS))
      have hu'_le : u' ≤ b₀' :=
        z_prime_le_of_forall_pred_le ht1 hPu (fun w hw => hpred_MT w (hMTu ▸ hw))
      exact ⟨hv'_le, hu'_le, hcnt'⟩
    by_cases hvu' : v' ≤ u'
    · exact ⟨havelHakimiStep (t :: M_T), v', u', havelHakimiStep_pairwise' _, hvu', hv', hu',
        fun b₀' TB' hB' => ⟨(hB'_props b₀' TB' hB').2.1, (hB'_props b₀' TB' hB').2.2⟩⟩
    · obtain ⟨M_swap, hMs, hu'', hv''⟩ := isIncM_comm hv' hu' (by omega)
      refine ⟨M_swap, u', v', hMs, by omega, hu'', hv'', fun b₀' TB' hB' => ?_⟩
      obtain ⟨hv'_le, _, hcnt'⟩ := hB'_props b₀' TB' hB'
      have hmono := countGE_mono_val (by omega : u' ≤ v') (b₀' :: TB')
      exact ⟨hv'_le, le_trans hcnt' hmono⟩

theorem State1.length_eq {A B : List ℕ} (h : State1 A B) : A.length = B.length := by
  rcases h with ⟨M, y, x, _, _, hy, hx⟩
  exact hy.length_eq.symm.trans hx.length_eq

theorem State2.length_eq {A B : List ℕ} (h : State2 A B) : A.length = B.length := by
  rcases h with ⟨M, v, u, _, _, hBM, hMA, _⟩
  exact hMA.length_eq.symm.trans hBM.length_eq.symm

theorem residueAux_le_length' (l : List ℕ) : residueAux l ≤ l.length := by
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

/-- **Favaron–Mahéo–Saclé (1991) Theorem 3.1**: if `(A, B)` is in `State1` (a single unit
transfer) or `State2` (the intermediate state of a head-decrementing unit transfer) and both
sequences are graphical and descending-sorted, then `residueAux A ≤ residueAux B`. -/
theorem residueAux_le_of_state12
    (hHH_graph : ∀ l : List ℕ, l.Pairwise (· ≥ ·) → Graphical (l : Multiset ℕ) →
      Graphical ((havelHakimiStep l : List ℕ) : Multiset ℕ)) :
    ∀ (n : ℕ) (A B : List ℕ),
      B.length = n →
      A.Pairwise (· ≥ ·) → B.Pairwise (· ≥ ·) →
      Graphical (A : Multiset ℕ) → Graphical (B : Multiset ℕ) →
      (State1 A B ∨ State2 A B) →
      residueAux A ≤ residueAux B
  | 0, A, B, hlenB, _, _, _, _, hS => by
    have hlenA : A.length = 0 := by
      rcases hS with h1 | h2
      · rw [h1.length_eq, hlenB]
      · rw [h2.length_eq, hlenB]
    rw [List.eq_nil_of_length_eq_zero hlenA, List.eq_nil_of_length_eq_zero hlenB]
  | n + 1, A, B, hlenB, hA, hB, hgA, hgB, hS => by
    have hlenA : A.length = n + 1 := by
      rcases hS with h1 | h2
      · rw [h1.length_eq, hlenB]
      · rw [h2.length_eq, hlenB]
    cases A with
    | nil => simp at hlenA
    | cons a₀ TA =>
      cases B with
      | nil => simp at hlenB
      | cons b₀ TB =>
        by_cases hb₀ : b₀ = 0
        · subst hb₀
          have hresB : residueAux (0 :: TB) = (a₀ :: TA).length := by
            simp only [residueAux, List.length_cons] at hlenA hlenB ⊢
            omega
          rw [hresB]
          exact residueAux_le_length' (a₀ :: TA)
        · have hb₀1 : 1 ≤ b₀ := Nat.pos_of_ne_zero hb₀
          have ha₀1 : 1 ≤ a₀ := by
            rcases hS with ⟨M, y, x, _, _, hy, _⟩ | ⟨M, v, u, _, _, _, hu, _⟩
            · exact head_pos_of_isIncM_right hA hy
            · exact head_pos_of_isIncM_right hA hu
          have ha₀ : a₀ ≠ 0 := by omega
          rw [residueAux_cons_ne_zero a₀ TA ha₀, residueAux_cons_ne_zero b₀ TB hb₀]
          have hlenB' : (havelHakimiStep (b₀ :: TB)).length = n := by
            rw [havelHakimiStep_length_cons]
            simp only [List.length_cons] at hlenB
            omega
          have hstep :
              (havelHakimiStep (a₀ :: TA) = havelHakimiStep (b₀ :: TB)) ∨
              (State1 (havelHakimiStep (a₀ :: TA)) (havelHakimiStep (b₀ :: TB)) ∨
               State2 (havelHakimiStep (a₀ :: TA)) (havelHakimiStep (b₀ :: TB))) := by
            rcases hS with h1 | h2
            · rcases step_state1 hA hB hgA hgB h1 with heq | h1' | h2'
              · left; rw [heq]
              · exact Or.inr (Or.inl h1')
              · exact Or.inr (Or.inr h2')
            · rcases step_state2 hA hB hgB hb₀1 h2 with heq | h1' | h2'
              · exact Or.inl heq
              · exact Or.inr (Or.inl h1')
              · exact Or.inr (Or.inr h2')
          rcases hstep with heq | hS'
          · rw [heq]
          · exact residueAux_le_of_state12 hHH_graph n _ _ hlenB'
              (havelHakimiStep_pairwise' _) (havelHakimiStep_pairwise' _)
              (hHH_graph _ hA hgA) (hHH_graph _ hB hgB) hS'

/-! ## Bridging the 2-switch chain to `residue_le_residue_induce_compl_type0` -/

open Classical in
theorem state1_induce_twoSwitch {V : Type} [Fintype V] [DecidableEq V]
    (G' : SimpleGraph V) (v₀ vi vj w : V)
    (hvi_ne : vi ≠ v₀) (hvi_nadj : ¬ G'.Adj v₀ vi) (hvj_adj : G'.Adj v₀ vj)
    (hle : G'.degree vj < G'.degree vi)
    (hwv : w ≠ v₀) (hwvi : w ≠ vi) (hwvj : w ≠ vj)
    (hadj_iw : G'.Adj vi w) (hnadj_jw : ¬ G'.Adj vj w) :
    State1 (((G'.twoSwitch v₀ vj vi w).induce ({v₀}ᶜ : Set V)).degreeMultiset.sort (· ≥ ·))
           (((G'.induce ({v₀}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·)) := by
  set G'' := G'.twoSwitch v₀ vj vi w
  have hv₀vi : v₀ ≠ vi := hvi_ne.symm
  have hvj_ne : vj ≠ v₀ := hvj_adj.ne'
  have hvjvi : vj ≠ vi := fun h => ne_of_lt hle (congrArg (fun z => G'.degree z) h)
  have hdeg_eq : ∀ u, G''.degree u = G'.degree u :=
    fun u => twoSwitch_degree G' v₀ vj vi w hvj_adj hadj_iw hvi_nadj hnadj_jw
      hv₀vi hwv.symm hvjvi hwvj.symm u
  have hvi_mem1 : vi ∈ (Finset.univ.erase v₀).val := by
    rw [Finset.mem_val, Finset.mem_erase]
    exact ⟨hvi_ne, Finset.mem_univ vi⟩
  have hvj_mem2 : vj ∈ ((Finset.univ.erase v₀).erase vi).val := by
    rw [Finset.mem_val, Finset.mem_erase, Finset.mem_erase]
    exact ⟨hvjvi, hvj_ne, Finset.mem_univ vj⟩
  set S₀ := ((Finset.univ.erase v₀).erase vi).erase vj
  have hval_split : (Finset.univ.erase v₀).val = vi ::ₘ vj ::ₘ S₀.val := by
    conv_lhs => rw [← Multiset.cons_erase hvi_mem1, ← Finset.erase_val]
    congr 1
    rw [← Multiset.cons_erase hvj_mem2, ← Finset.erase_val]
  let f (H : SimpleGraph V) (u : V) : ℕ := H.degree u - if H.Adj u v₀ then 1 else 0
  have hf_S₀ : S₀.val.map (f G'') = S₀.val.map (f G') := by
    apply Multiset.map_congr rfl
    intro u hu
    rw [Finset.mem_val, Finset.mem_erase, Finset.mem_erase, Finset.mem_erase] at hu
    obtain ⟨huvj, huvi, huv₀, _⟩ := hu
    have hadj_iff : G''.Adj u v₀ ↔ G'.Adj u v₀ := by
      rw [twoSwitch_adj]
      simp only [Sym2.eq_iff]
      aesop
    dsimp [f]
    rw [hdeg_eq u]
    by_cases h : G'.Adj u v₀
    · rw [if_pos h, if_pos (hadj_iff.mpr h)]
    · rw [if_neg h, if_neg (fun h' => h (hadj_iff.mp h'))]
  have hf_vi_G' : f G' vi = G'.degree vi := by
    dsimp [f]
    rw [if_neg (fun h => hvi_nadj (G'.adj_symm h)), Nat.sub_zero]
  have hf_vi_G'' : f G'' vi = G'.degree vi - 1 := by
    dsimp [f]
    have hadj : G''.Adj vi v₀ := by
      rw [twoSwitch_adj]; simp only [Sym2.eq_iff]; aesop
    rw [hdeg_eq vi, if_pos hadj]
  have hf_vj_G' : f G' vj = G'.degree vj - 1 := by
    dsimp [f]
    rw [if_pos (G'.adj_symm hvj_adj)]
  have hf_vj_G'' : f G'' vj = G'.degree vj := by
    dsimp [f]
    have hnadj : ¬ G''.Adj vj v₀ := by
      rw [twoSwitch_adj]; simp only [Sym2.eq_iff]; aesop
    rw [hdeg_eq vj, if_neg hnadj, Nat.sub_zero]
  have hdvj_pos : 1 ≤ G'.degree vj :=
    (G'.degree_pos_iff_exists_adj vj).mpr ⟨v₀, G'.adj_symm hvj_adj⟩
  set x := G'.degree vi - 1
  set y := G'.degree vj - 1
  set C := S₀.val.map (f G')
  have hx1 : x + 1 = G'.degree vi := by omega
  have hy1 : y + 1 = G'.degree vj := by omega
  have hG'_m : (G'.induce ({v₀}ᶜ : Set V)).degreeMultiset = (x + 1) ::ₘ y ::ₘ C := by
    rw [degreeMultiset_induce_compl G' v₀, hval_split, Multiset.map_cons, Multiset.map_cons]
    change f G' vi ::ₘ f G' vj ::ₘ C = (x + 1) ::ₘ y ::ₘ C
    rw [hf_vi_G', hf_vj_G', hx1]
  have hG''_m : (G''.induce ({v₀}ᶜ : Set V)).degreeMultiset = (y + 1) ::ₘ x ::ₘ C := by
    rw [degreeMultiset_induce_compl G'' v₀, hval_split, Multiset.map_cons, Multiset.map_cons]
    change f G'' vi ::ₘ f G'' vj ::ₘ S₀.val.map (f G'') = (y + 1) ::ₘ x ::ₘ C
    rw [hf_vi_G'', hf_vj_G'', hf_S₀, hy1, Multiset.cons_swap]
  set mM : Multiset ℕ := x ::ₘ y ::ₘ C
  refine ⟨mM.sort (· ≥ ·), y, x, Multiset.pairwise_sort _ _, by omega, ?_, ?_⟩
  · refine ⟨x ::ₘ C, ?_, ?_⟩
    · rw [Multiset.sort_eq, Multiset.cons_swap]
    · rw [Multiset.sort_eq, hG''_m]
  · refine ⟨y ::ₘ C, ?_, ?_⟩
    · rw [Multiset.sort_eq]
    · rw [Multiset.sort_eq, hG'_m]

open Classical in
theorem adjWeight_le_sum_univ {V : Type} [Fintype V] (G' : SimpleGraph V) (wt : V → ℕ) (v₀ : V) :
    adjWeight G' wt v₀ ≤ ∑ u : V, wt u := by
  unfold adjWeight
  exact Finset.sum_le_univ_sum_of_nonneg (fun _ => Nat.zero_le _)

open Classical in
/-- **Native proof of the `Type 0` residue-monotonicity crux via the 2-switch chain and FMS
Theorem 3.1 (`residueAux_le_of_state12`).** -/
theorem residue_le_residue_induce_compl_type0 {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [hAdj : DecidableRel G.Adj] {v₀ : V}
    (hv₀ : ∀ w, G.degree w ≤ G.degree v₀) (hpos : 0 < G.degree v₀) :
    G.residue ≤ (G.induce ({v₀}ᶜ : Set V)).residue := by
  cases Subsingleton.elim hAdj (Classical.decRel G.Adj)
  set UB := ∑ u : V, G.degree u
  suffices h : ∀ (k : ℕ) (G' : SimpleGraph V),
      (∀ x, G'.degree x = G.degree x) →
      UB - adjWeight G' (fun u => G.degree u) v₀ = k →
      G.residue ≤ (G'.induce ({v₀}ᶜ : Set V)).residue by
    exact h (UB - adjWeight G (fun u => G.degree u) v₀) G (fun _ => rfl) rfl
  intro k
  induction k using Nat.strong_induction_on with
  | _ k IH =>
    intro G' hdeg hk
    by_cases hinv : ∀ vi vj, vi ≠ v₀ → ¬ G'.Adj v₀ vi → G'.Adj v₀ vj → G'.degree vi ≤ G'.degree vj
    · -- Base case: no inversion at v₀ in G', so G' - v₀ realizes havelHakimiStep(sort(deg G))!
      have hmaxG' : ∀ w, G'.degree w ≤ G'.degree v₀ := by
        intro w; rw [hdeg w, hdeg v₀]; exact hv₀ w
      have hG'eq : G'.degreeMultiset = G.degreeMultiset := by
        unfold SimpleGraph.degreeMultiset
        exact Multiset.map_congr rfl (fun x _ => hdeg x)
      have hcore := induce_degreeMultiset_eq_havelHakimiStep G' v₀ hmaxG' hinv
      have hsort_ind : ((G'.induce ({v₀}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·)
          = havelHakimiStep ((G.degreeMultiset).sort (· ≥ ·)) := by
        apply List.Perm.eq_of_sortedGE (Multiset.pairwise_sort _ _).sortedGE
          (havelHakimiStep_pairwise' _).sortedGE
        rw [← Multiset.coe_eq_coe, Multiset.sort_eq, hcore, hG'eq]
      have hsc := sortedDeg_cons G v₀ hv₀
      have hGres : G.residue = residueAux (havelHakimiStep ((G.degreeMultiset).sort (· ≥ ·))) := by
        simp only [SimpleGraph.residue, SimpleGraph.degreeMultiset]
        rw [hsc, residueAux_cons_ne_zero _ _ hpos.ne']
      have hG'res : (G'.induce ({v₀}ᶜ : Set V)).residue
          = residueAux (((G'.induce ({v₀}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·)) := rfl
      rw [hGres, hG'res, hsort_ind]
    · -- Inductive step: an inversion exists at v₀; apply one West 2-switch!
      push Not at hinv
      obtain ⟨vi, vj, hvi_ne, hvi_nadj, hvj_adj, hle⟩ := hinv
      have hv₀vi : v₀ ≠ vi := hvi_ne.symm
      obtain ⟨w, hwv, hwvi, hwvj, hadj_iw, hnadj_jw⟩ :=
        exists_twoSwitch_partner G' v₀ vi vj hvj_adj hvi_nadj hv₀vi hle.le
      have hvjvi : vj ≠ vi := fun h => ne_of_lt hle (congrArg (fun z => G'.degree z) h)
      set G'' := G'.twoSwitch v₀ vj vi w
      have hdeg'' : ∀ x, G''.degree x = G.degree x :=
        fun x => (twoSwitch_degree G' v₀ vj vi w hvj_adj hadj_iw hvi_nadj hnadj_jw
          hv₀vi hwv.symm hvjvi hwvj.symm x).trans (hdeg x)
      have hlt_wt : adjWeight G' (fun u => G.degree u) v₀ < adjWeight G'' (fun u => G.degree u) v₀ :=
        adjWeight_lt_twoSwitch G' v₀ vj vi w (fun u => G.degree u)
          hvj_adj hvi_nadj hv₀vi hwv.symm
          (by show G.degree vj < G.degree vi; rw [← hdeg vj, ← hdeg vi]; exact hle)
      have hub'' : adjWeight G'' (fun u => G.degree u) v₀ ≤ UB :=
        adjWeight_le_sum_univ G'' (fun u => G.degree u) v₀
      have hk'' : UB - adjWeight G'' (fun u => G.degree u) v₀ < k := by omega
      have h1 : G.residue ≤ (G''.induce ({v₀}ᶜ : Set V)).residue :=
        IH _ hk'' G'' hdeg'' rfl
      have hS1 : State1 (((G''.induce ({v₀}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·))
                        (((G'.induce ({v₀}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·)) :=
        state1_induce_twoSwitch G' v₀ vi vj w hvi_ne hvi_nadj hvj_adj hle
          hwv hwvi hwvj hadj_iw hnadj_jw
      have h2 : (G''.induce ({v₀}ᶜ : Set V)).residue ≤ (G'.induce ({v₀}ᶜ : Set V)).residue := by
        unfold SimpleGraph.residue
        change residueAux (((G''.induce ({v₀}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·))
          ≤ residueAux (((G'.induce ({v₀}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·))
        refine residueAux_le_of_state12 havelHakimiStep_preserves_graphical'
          (((G'.induce ({v₀}ᶜ : Set V)).degreeMultiset).sort (· ≥ ·)).length
          _ _ rfl (Multiset.pairwise_sort _ _) (Multiset.pairwise_sort _ _) ?_ ?_ (Or.inl hS1)
        · rw [Multiset.sort_eq]; exact graphical_induce_compl G'' v₀
        · rw [Multiset.sort_eq]; exact graphical_induce_compl G' v₀
      exact le_trans h1 h2

/-! ## Universe-polymorphic transport (`{V : Type*}`) and full Conjecture 1 closure -/

theorem degree_induce_compl_singleton_univ {V : Type*} [Fintype V] [DecidableEq V]
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

theorem degreeMultiset_induce_compl_univ {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [DecidableRel G.Adj] (v : V) :
    (G.induce ({v}ᶜ : Set V)).degreeMultiset
      = (Finset.univ.erase v).val.map (fun w => G.degree w - if G.Adj w v then 1 else 0) := by
  unfold degreeMultiset
  rw [Multiset.map_congr rfl (fun u _ => degree_induce_compl_singleton_univ G v u)]
  have hres : (fun u : ({v}ᶜ : Set V) => G.degree (u : V) - (if G.Adj (u : V) v then 1 else 0))
      = Subtype.restrict (· ∈ ({v}ᶜ : Set V)) (fun w => G.degree w - if G.Adj w v then 1 else 0) :=
    rfl
  rw [hres, Finset.univ_val_map_subtype_restrict]
  have hfe : Finset.univ.filter (· ∈ ({v}ᶜ : Set V)) = Finset.univ.erase v := by
    ext w; simp
  rw [hfe]

open Classical in
/-- **Universe-polymorphic native replacement for `Favaron.lean`'s custom axiom
`residue_le_residue_induce_compl_of_maxDegree`.** Same signature and universe scope `{V : Type*}`,
zero custom axioms. -/
theorem residue_le_residue_induce_compl_of_maxDegree' {V : Type*} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) [hAdj : DecidableRel G.Adj] {v : V}
    (hv : ∀ w, G.degree w ≤ G.degree v) (hpos : 0 < G.degree v) :
    G.residue ≤ (G.induce ({v}ᶜ : Set V)).residue := by
  cases Subsingleton.elim hAdj (Classical.decRel G.Adj)
  obtain ⟨e⟩ := Fintype.truncEquivFin V
  let W := Fin (Fintype.card V)
  let G₀ : SimpleGraph W := SimpleGraph.comap e.symm G
  have hdeg₀ : ∀ i : W, G₀.degree i = G.degree (e.symm i) := by
    intro i
    rw [← card_neighborFinset_eq_degree, ← card_neighborFinset_eq_degree]
    have hmap : G₀.neighborFinset i = (G.neighborFinset (e.symm i)).map e.toEmbedding := by
      ext j
      simp only [mem_neighborFinset, Finset.mem_map_equiv]
      rfl
    rw [hmap, Finset.card_map]
  have huniv_map : (Finset.univ : Finset W).val = (Finset.univ : Finset V).val.map e := by
    have h : (Finset.univ : Finset W) = (Finset.univ : Finset V).map e.toEmbedding := by
      ext j; simp
    rw [h, Finset.map_val, Equiv.coe_toEmbedding]
  have hdm₀ : G₀.degreeMultiset = G.degreeMultiset := by
    unfold SimpleGraph.degreeMultiset
    rw [huniv_map, Multiset.map_map]
    apply Multiset.map_congr rfl
    intro x _
    dsimp only [Function.comp_apply]
    rw [hdeg₀ (e x)]
    exact congrArg (fun z => G.degree z) (e.symm_apply_apply x)
  have hres₀ : G₀.residue = G.residue := by
    simp only [SimpleGraph.residue]
    rw [← SimpleGraph.degreeMultiset, ← SimpleGraph.degreeMultiset, hdm₀]
  have herase_map : ((Finset.univ : Finset W).erase (e v)).val
      = ((Finset.univ : Finset V).erase v).val.map e := by
    have h : (Finset.univ : Finset W).erase (e v)
        = ((Finset.univ : Finset V).erase v).map e.toEmbedding := by
      ext j
      simp
    rw [h, Finset.map_val, Equiv.coe_toEmbedding]
  have hdm_ind : (G₀.induce ({e v}ᶜ : Set W)).degreeMultiset
      = (G.induce ({v}ᶜ : Set V)).degreeMultiset := by
    rw [degreeMultiset_induce_compl G₀ (e v), degreeMultiset_induce_compl_univ G v,
      herase_map, Multiset.map_map]
    apply Multiset.map_congr rfl
    intro x _
    dsimp only [Function.comp_apply]
    have hd : G₀.degree (e x) = G.degree x :=
      (hdeg₀ (e x)).trans (congrArg (fun z => G.degree z) (e.symm_apply_apply x))
    have hadj : G₀.Adj (e x) (e v) ↔ G.Adj x v := by
      change G.Adj (e.symm (e x)) (e.symm (e v)) ↔ G.Adj x v
      rw [e.symm_apply_apply, e.symm_apply_apply]
    rw [hd]
    by_cases h : G.Adj x v
    · rw [if_pos h, if_pos (hadj.mpr h)]
    · rw [if_neg h, if_neg (fun h' => h (hadj.mp h'))]
  have hres_ind : (G₀.induce ({e v}ᶜ : Set W)).residue
      = (G.induce ({v}ᶜ : Set V)).residue := by
    simp only [SimpleGraph.residue]
    rw [← SimpleGraph.degreeMultiset, ← SimpleGraph.degreeMultiset, hdm_ind]
  have hdv : G₀.degree (e v) = G.degree v :=
    (hdeg₀ (e v)).trans (congrArg (fun z => G.degree z) (e.symm_apply_apply v))
  have hv₀ : ∀ i : W, G₀.degree i ≤ G₀.degree (e v) := by
    intro i
    rw [hdeg₀ i, hdv]
    exact hv (e.symm i)
  have hpos₀ : 0 < G₀.degree (e v) := by
    rw [hdv]
    exact hpos
  rw [← hres₀, ← hres_ind]
  exact residue_le_residue_induce_compl_type0 G₀ hv₀ hpos₀

end SimpleGraph
