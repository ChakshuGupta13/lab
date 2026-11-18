import OrigamiCone.ConeDegree

namespace OrigamiCone

variable {m n : ℕ}

lemma adj_grid_cases {v u : Cell m n} (h : adj v u) :
    (u.1 = v.1 ∧ (u.2.val = v.2.val + 1 ∨ u.2.val + 1 = v.2.val)) ∨
    (u.2 = v.2 ∧ (u.1.val = v.1.val + 1 ∨ u.1.val + 1 = v.1.val)) := by
  have hn : ((v.1.val : ℤ) - u.1.val).natAbs + ((v.2.val : ℤ) - u.2.val).natAbs = 1 := by
    have h' := h; unfold adj gdist at h'; exact_mod_cast h'
  by_cases hr : u.1.val = v.1.val
  · left
    exact ⟨Fin.ext hr, by omega⟩
  · right
    exact ⟨Fin.ext (by omega), by omega⟩

def PathEnd {k : ℕ} (a i : Fin k) : Prop :=
  (i.val = 0 ∧ a.val ≠ 0) ∨ (i.val + 1 = k ∧ a.val + 1 ≠ k)

instance {k : ℕ} (a i : Fin k) : Decidable (PathEnd a i) := by
  unfold PathEnd; infer_instance

lemma Dq_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) (q v : Cell m n) :
    IsStrictLocalMax (Dq q) v ↔ (PathEnd q.1 v.1 ∧ PathEnd q.2 v.2) := by
  constructor
  ·
    intro hmax
    refine ⟨?_, ?_⟩
    ·
      by_cases h0 : v.1.val = 0
      · refine Or.inl ⟨h0, ?_⟩
        have hadj : adj v (⟨1, by omega⟩, v.2) := by simp only [adj, gdist]; omega
        have := hmax _ hadj
        simp only [Dq, gdist] at this
        omega
      · by_cases h1 : v.1.val + 1 = m
        · refine Or.inr ⟨h1, ?_⟩
          have hadj : adj v (⟨v.1.val - 1, by omega⟩, v.2) := by
            simp only [adj, gdist]; omega
          have := hmax _ hadj
          simp only [Dq, gdist] at this
          omega
        · exfalso
          have hadjL : adj v (⟨v.1.val - 1, by omega⟩, v.2) := by
            simp only [adj, gdist]; omega
          have hadjR : adj v (⟨v.1.val + 1, by omega⟩, v.2) := by
            simp only [adj, gdist]; omega
          have hL := hmax _ hadjL
          have hR := hmax _ hadjR
          simp only [Dq, gdist] at hL hR
          omega
    ·
      by_cases h0 : v.2.val = 0
      · refine Or.inl ⟨h0, ?_⟩
        have hadj : adj v (v.1, ⟨1, by omega⟩) := by simp only [adj, gdist]; omega
        have := hmax _ hadj
        simp only [Dq, gdist] at this
        omega
      · by_cases h1 : v.2.val + 1 = n
        · refine Or.inr ⟨h1, ?_⟩
          have hadj : adj v (v.1, ⟨v.2.val - 1, by omega⟩) := by
            simp only [adj, gdist]; omega
          have := hmax _ hadj
          simp only [Dq, gdist] at this
          omega
        · exfalso
          have hadjL : adj v (v.1, ⟨v.2.val - 1, by omega⟩) := by
            simp only [adj, gdist]; omega
          have hadjR : adj v (v.1, ⟨v.2.val + 1, by omega⟩) := by
            simp only [adj, gdist]; omega
          have hL := hmax _ hadjL
          have hR := hmax _ hadjR
          simp only [Dq, gdist] at hL hR
          omega
  ·
    rintro ⟨hrow, hcol⟩ u hu
    simp only [Dq, gdist]
    rcases adj_grid_cases hu with ⟨hueq, hdiff⟩ | ⟨hueq, hdiff⟩
    ·
      have h1 : u.1.val = v.1.val := by rw [hueq]
      rcases hcol with ⟨hv, ha⟩ | ⟨hv, ha⟩ <;> omega
    ·
      have h2 : u.2.val = v.2.val := by rw [hueq]
      rcases hrow with ⟨hv, ha⟩ | ⟨hv, ha⟩ <;> omega

lemma kappa_eq_mul (hm : 2 ≤ m) (hn : 2 ≤ n) (q : Cell m n) :
    kappa q = (Finset.univ.filter (PathEnd q.1)).card
              * (Finset.univ.filter (PathEnd q.2)).card := by
  unfold kappa
  rw [← Finset.card_product]
  congr 1
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_product,
             Dq_strictMax_iff hm hn]

lemma pathEnd_card {k : ℕ} (hk : 2 ≤ k) (a : Fin k) :
    (Finset.univ.filter (PathEnd a)).card
      = if a.val = 0 ∨ a.val + 1 = k then 1 else 2 := by
  have he0 : (0 : ℕ) < k := by omega
  have he1 : k - 1 < k := by omega
  have hv0 : (⟨0, he0⟩ : Fin k).val = 0 := rfl
  have hv1 : (⟨k - 1, he1⟩ : Fin k).val = k - 1 := rfl
  by_cases ha0 : a.val = 0
  · rw [if_pos (Or.inl ha0), Finset.card_eq_one]
    refine ⟨⟨k - 1, he1⟩, ?_⟩
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
               PathEnd]
    constructor
    · rintro (⟨_, ha⟩ | ⟨hi, _⟩)
      · exact absurd ha0 ha
      · exact Fin.ext (by omega)
    · rintro rfl
      exact Or.inr ⟨by omega, by omega⟩
  · by_cases ha1 : a.val + 1 = k
    · rw [if_pos (Or.inr ha1), Finset.card_eq_one]
      refine ⟨⟨0, he0⟩, ?_⟩
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
                 PathEnd]
      constructor
      · rintro (⟨hi, _⟩ | ⟨_, ha⟩)
        · exact Fin.ext (by omega)
        · exact absurd ha1 ha
      · rintro rfl
        exact Or.inl ⟨rfl, ha0⟩
    · rw [if_neg (by push_neg; exact ⟨ha0, ha1⟩)]
      have hset : Finset.univ.filter (PathEnd a) = {⟨0, he0⟩, ⟨k - 1, he1⟩} := by
        ext i
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
                   Finset.mem_singleton, PathEnd]
        constructor
        · rintro (⟨hi, _⟩ | ⟨hi, _⟩)
          · exact Or.inl (Fin.ext (by omega))
          · exact Or.inr (Fin.ext (by omega))
        · rintro (rfl | rfl)
          · exact Or.inl ⟨rfl, ha0⟩
          · exact Or.inr ⟨by omega, ha1⟩
      rw [hset, Finset.card_insert_of_notMem (by
        simp only [Finset.mem_singleton]
        intro hc
        have := congrArg Fin.val hc
        omega), Finset.card_singleton]

def IsEndpoint {k : ℕ} (a : Fin k) : Prop := a.val = 0 ∨ a.val + 1 = k

lemma pathEnd_card_endpoint {k : ℕ} (hk : 2 ≤ k) {a : Fin k} (h : IsEndpoint a) :
    (Finset.univ.filter (PathEnd a)).card = 1 := by
  unfold IsEndpoint at h
  rw [pathEnd_card hk, if_pos h]

lemma pathEnd_card_interior {k : ℕ} (hk : 2 ≤ k) {a : Fin k} (h : ¬ IsEndpoint a) :
    (Finset.univ.filter (PathEnd a)).card = 2 := by
  unfold IsEndpoint at h
  rw [pathEnd_card hk, if_neg h]

theorem kappa_mem (hm : 2 ≤ m) (hn : 2 ≤ n) (q : Cell m n) :
    kappa q = 1 ∨ kappa q = 2 ∨ kappa q = 4 := by
  rw [kappa_eq_mul hm hn]
  by_cases hr : IsEndpoint q.1 <;> by_cases hc : IsEndpoint q.2
  · rw [pathEnd_card_endpoint hm hr, pathEnd_card_endpoint hn hc]; left; rfl
  · rw [pathEnd_card_endpoint hm hr, pathEnd_card_interior hn hc]; right; left; rfl
  · rw [pathEnd_card_interior hm hr, pathEnd_card_endpoint hn hc]; right; left; rfl
  · rw [pathEnd_card_interior hm hr, pathEnd_card_interior hn hc]; right; right; rfl

theorem kappa_corner (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (hr : IsEndpoint q.1) (hc : IsEndpoint q.2) : kappa q = 1 := by
  rw [kappa_eq_mul hm hn, pathEnd_card_endpoint hm hr, pathEnd_card_endpoint hn hc]

theorem kappa_boundary (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (h : (IsEndpoint q.1 ∧ ¬ IsEndpoint q.2) ∨ (¬ IsEndpoint q.1 ∧ IsEndpoint q.2)) :
    kappa q = 2 := by
  rw [kappa_eq_mul hm hn]
  rcases h with ⟨hr, hc⟩ | ⟨hr, hc⟩
  · rw [pathEnd_card_endpoint hm hr, pathEnd_card_interior hn hc]
  · rw [pathEnd_card_interior hm hr, pathEnd_card_endpoint hn hc]

theorem kappa_interior (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (hr : ¬ IsEndpoint q.1) (hc : ¬ IsEndpoint q.2) : kappa q = 4 := by
  rw [kappa_eq_mul hm hn, pathEnd_card_interior hm hr, pathEnd_card_interior hn hc]

theorem cone_corner_degree_two (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (hr : IsEndpoint q.1) (hc : IsEndpoint q.2) (b : Cell m n) :
    (neighbors (cone q b)).ncard = 2 := by
  have hmn : 2 ≤ m * n := by calc 2 ≤ 2 * 2 := by norm_num
                                _ ≤ m * n := Nat.mul_le_mul hm hn
  rw [cone_degree_eq hmn, kappa_corner hm hn hr hc]

end OrigamiCone
