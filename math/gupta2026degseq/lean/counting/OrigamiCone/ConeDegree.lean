import OrigamiCone.Cone
import OrigamiCone.MinDegree

namespace OrigamiCone

variable {m n : ℕ}

def Dq (q : Cell m n) : Cell m n → ℤ := fun v => gdist q v

lemma Dq_isHeight (q : Cell m n) : IsHeight (Dq q) := by
  intro p p' hpp'
  rw [abs_eq (by norm_num : (0 : ℤ) ≤ 1)]
  simp only [Dq]
  rcases gdist_adj_step (q := q) hpp' with h | h
  · left; omega
  · right; omega

lemma cone_min_iff_Dq_max (q b v : Cell m n) :
    IsStrictLocalMin (cone q b) v ↔ IsStrictLocalMax (Dq q) v := by
  constructor
  · intro h u hu
    have := h u hu
    simp only [cone, Dq] at this ⊢
    omega
  · intro h u hu
    have := h u hu
    simp only [cone, Dq] at this ⊢
    omega

def kappa (q : Cell m n) : ℕ :=
  (Finset.univ.filter (IsStrictLocalMax (Dq q))).card

lemma cone_strictMax_singleton (q b : Cell m n) :
    Finset.univ.filter (IsStrictLocalMax (cone q b)) = {q} := by
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  exact ⟨fun hv => cone_unique_max q b v hv, fun h => h ▸ cone_max_at q b⟩

lemma cone_strictMin_card (q b : Cell m n) :
    (Finset.univ.filter (IsStrictLocalMin (cone q b))).card = kappa q := by
  unfold kappa
  congr 1
  apply Finset.filter_congr
  intro v _
  simp only [cone_min_iff_Dq_max]

theorem cone_degree_eq (hmn : 2 ≤ m * n) (q b : Cell m n) :
    (neighbors (cone q b)).ncard = 1 + kappa q := by
  rw [degree_eq_extrema (cone_isHeight q b) hmn]
  have hsplit : Finset.univ.filter (IsStrictLocalExtremum (cone q b))
      = Finset.univ.filter (IsStrictLocalMax (cone q b))
        ∪ Finset.univ.filter (IsStrictLocalMin (cone q b)) := by
    ext v
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union,
               IsStrictLocalExtremum]
  have hdisj : Disjoint (Finset.univ.filter (IsStrictLocalMax (cone q b)))
      (Finset.univ.filter (IsStrictLocalMin (cone q b))) := by
    rw [Finset.disjoint_left]
    intro v hvmax hvmin
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hvmax hvmin
    exact max_min_excl hmn hvmax hvmin
  rw [hsplit, Finset.card_union_of_disjoint hdisj, cone_strictMax_singleton,
      Finset.card_singleton, cone_strictMin_card]

end OrigamiCone
