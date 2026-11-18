import OrigamiCone.Degree3

namespace OrigamiCone

variable {m n : ℕ}

instance {k : ℕ} (a : Fin k) : Decidable (IsEndpoint a) :=
  show Decidable (a.val = 0 ∨ a.val + 1 = k) from inferInstance

instance {m n : ℕ} (q : Cell m n) : Decidable (IsBoundaryNonCorner q) :=
  show Decidable ((IsEndpoint q.1 ∧ ¬ IsEndpoint q.2)
                    ∨ (¬ IsEndpoint q.1 ∧ IsEndpoint q.2)) from inferInstance

lemma card_isEndpoint_fin (k : ℕ) (hk : 2 ≤ k) :
    (Finset.univ.filter (IsEndpoint : Fin k → Prop)).card = 2 := by
  have h0 : 0 < k := by omega
  have hk1 : k - 1 < k := by omega

  have hset : Finset.univ.filter (IsEndpoint : Fin k → Prop)
            = {⟨0, h0⟩, ⟨k - 1, hk1⟩} := by
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
               Finset.mem_singleton, IsEndpoint]
    constructor
    · rintro (h | h)
      · exact Or.inl (Fin.ext h)
      · refine Or.inr (Fin.ext ?_); show i.val = k - 1; omega
    · rintro (rfl | rfl)
      · exact Or.inl rfl
      · refine Or.inr ?_; show (k - 1) + 1 = k; omega
  rw [hset, Finset.card_insert_of_notMem (by
    simp only [Finset.mem_singleton]
    intro hc

    have hval : (0 : ℕ) = k - 1 := congrArg Fin.val hc
    omega), Finset.card_singleton]

lemma card_not_isEndpoint_fin (k : ℕ) (hk : 2 ≤ k) :
    (Finset.univ.filter (fun a : Fin k => ¬ IsEndpoint a)).card = k - 2 := by

  have huniv : (Finset.univ : Finset (Fin k)).card = k := by
    simp [Finset.card_univ, Fintype.card_fin]
  have hcomplement := Finset.card_filter_add_card_filter_not
    (s := (Finset.univ : Finset (Fin k))) (p := IsEndpoint)
  rw [card_isEndpoint_fin k hk, huniv] at hcomplement
  omega

private lemma card_filter_prod_and {α β : Type*} [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    (P : α → Prop) (Q : β → Prop) [DecidablePred P] [DecidablePred Q] :
    (Finset.univ.filter (fun q : α × β => P q.1 ∧ Q q.2)).card
      = (Finset.univ.filter P).card * (Finset.univ.filter Q).card := by
  rw [← Finset.card_product]
  congr 1
  ext v
  simp [Finset.mem_filter, Finset.mem_product]

theorem card_boundaryNonCorner (hm : 2 ≤ m) (hn : 2 ≤ n) :
    (Finset.univ.filter (IsBoundaryNonCorner : Cell m n → Prop)).card
      = 2 * (m + n - 4) := by

  have h_or : ∀ q : Cell m n,
    IsBoundaryNonCorner q ↔
      (IsEndpoint q.1 ∧ ¬ IsEndpoint q.2)
        ∨ (¬ IsEndpoint q.1 ∧ IsEndpoint q.2) := by
    intro q; rfl
  rw [show Finset.univ.filter (IsBoundaryNonCorner : Cell m n → Prop)
        = Finset.univ.filter
            (fun q : Cell m n =>
              (IsEndpoint q.1 ∧ ¬ IsEndpoint q.2)
                ∨ (¬ IsEndpoint q.1 ∧ IsEndpoint q.2)) from
      Finset.filter_congr (fun q _ => h_or q)]
  rw [Finset.filter_or]

  have hdisj : Disjoint
      (Finset.univ.filter
        (fun q : Cell m n => IsEndpoint q.1 ∧ ¬ IsEndpoint q.2))
      (Finset.univ.filter
        (fun q : Cell m n => ¬ IsEndpoint q.1 ∧ IsEndpoint q.2)) := by
    rw [Finset.disjoint_left]
    intro q h1 h2
    rw [Finset.mem_filter] at h1 h2
    exact h2.2.1 h1.2.1
  rw [Finset.card_union_of_disjoint hdisj]

  rw [card_filter_prod_and (α := Fin m) (β := Fin n)
        (fun a => IsEndpoint a) (fun b => ¬ IsEndpoint b)]
  rw [card_filter_prod_and (α := Fin m) (β := Fin n)
        (fun a => ¬ IsEndpoint a) (fun b => IsEndpoint b)]

  rw [card_isEndpoint_fin m hm, card_not_isEndpoint_fin n hn,
      card_not_isEndpoint_fin m hm, card_isEndpoint_fin n hn]

  omega

end OrigamiCone
