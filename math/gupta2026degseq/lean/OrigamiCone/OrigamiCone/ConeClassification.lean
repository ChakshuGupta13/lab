import OrigamiCone.ConeDegree

/-!
# Cone classification trichotomy κ ∈ {1,2,4}  (Corollary 2.3, second part)

Formalisation of the trichotomy in Corollary 2.3:

> Writing `q = (a,b)`, the distance `d(q,(i,j)) = |i-a| + |j-b|` is separable, and
> a grid neighbour changes a single coordinate, so `(i,j)` is a local maximum of
> `d(q,·)` iff `i` is a local maximum of `t ↦ |t-a|` on `{1,…,m}` and `j` is a
> local maximum of `t ↦ |t-b|` on `{1,…,n}`. The function `t ↦ |t-a|` on a path
> has local maxima at both endpoints when `1 < a < m`, and at the single far
> endpoint when `a ∈ {1,m}`. Combining the two coordinates gives `κ(q) ∈ {1,2,4}`.

Building on `ConeDegree` (`cone_degree_eq : degree = 1 + κ`), this module proves
`κ(q) = |R(a)| · |C(b)|` where `R(a)`, `C(b)` are the 1-D endpoint sets, computes
each 1-D count as `1` (apex an endpoint) or `2` (apex interior), and deduces the
trichotomy and the degree-2 attainment of a corner cone (`cone_corner_degree_two`,
which discharges the "degree 2 occurs" half of Lemma 3.1).

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Grid adjacency, enumerated.** Two adjacent cells either share their row and
differ by one in the column, or share their column and differ by one in the row.
-/
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

/-- The 1-D endpoint predicate: position `i` on a path of length `k` is a strict
local maximum of the distance to apex `a` exactly when `i` is an endpoint on the
far side of `a`. -/
def PathEnd {k : ℕ} (a i : Fin k) : Prop :=
  (i.val = 0 ∧ a.val ≠ 0) ∨ (i.val + 1 = k ∧ a.val + 1 ≠ k)

instance {k : ℕ} (a i : Fin k) : Decidable (PathEnd a i) := by
  unfold PathEnd; infer_instance

/-- **Separability of the distance maxima.** For `m,n ≥ 2`, a cell `v` is a strict
local maximum of `d(q,·)` exactly when each coordinate is a 1-D endpoint maximum.
-/
lemma Dq_strictMax_iff (hm : 2 ≤ m) (hn : 2 ≤ n) (q v : Cell m n) :
    IsStrictLocalMax (Dq q) v ↔ (PathEnd q.1 v.1 ∧ PathEnd q.2 v.2) := by
  constructor
  · -- max ⟹ endpoint in each coordinate
    intro hmax
    refine ⟨?_, ?_⟩
    · -- row coordinate
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
    · -- column coordinate (symmetric)
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
  · -- endpoint in each coordinate ⟹ max
    rintro ⟨hrow, hcol⟩ u hu
    simp only [Dq, gdist]
    rcases adj_grid_cases hu with ⟨hueq, hdiff⟩ | ⟨hueq, hdiff⟩
    · -- column neighbour: u.1 = v.1
      have h1 : u.1.val = v.1.val := by rw [hueq]
      rcases hcol with ⟨hv, ha⟩ | ⟨hv, ha⟩ <;> omega
    · -- row neighbour: u.2 = v.2
      have h2 : u.2.val = v.2.val := by rw [hueq]
      rcases hrow with ⟨hv, ha⟩ | ⟨hv, ha⟩ <;> omega

/-- **κ as a product.** For `m,n ≥ 2`, the maxima of `d(q,·)` form the product of
the two 1-D endpoint sets, so `κ(q)` is the product of their sizes. -/
lemma kappa_eq_mul (hm : 2 ≤ m) (hn : 2 ≤ n) (q : Cell m n) :
    kappa q = (Finset.univ.filter (PathEnd q.1)).card
              * (Finset.univ.filter (PathEnd q.2)).card := by
  unfold kappa
  rw [← Finset.card_product]
  congr 1
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_product,
             Dq_strictMax_iff hm hn]

/-- **1-D endpoint count.** On a path of length `k ≥ 2`, the distance to apex `a`
has one strict local maximum when `a` is an endpoint, and two when `a` is
interior. -/
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

/-- The apex `a` is an endpoint of its path. -/
def IsEndpoint {k : ℕ} (a : Fin k) : Prop := a.val = 0 ∨ a.val + 1 = k

/-- The 1-D count is `1` at an endpoint apex, `2` at an interior apex. -/
lemma pathEnd_card_endpoint {k : ℕ} (hk : 2 ≤ k) {a : Fin k} (h : IsEndpoint a) :
    (Finset.univ.filter (PathEnd a)).card = 1 := by
  unfold IsEndpoint at h
  rw [pathEnd_card hk, if_pos h]

lemma pathEnd_card_interior {k : ℕ} (hk : 2 ≤ k) {a : Fin k} (h : ¬ IsEndpoint a) :
    (Finset.univ.filter (PathEnd a)).card = 2 := by
  unfold IsEndpoint at h
  rw [pathEnd_card hk, if_neg h]

/-- **κ(q) ∈ {1,2,4}.** -/
theorem kappa_mem (hm : 2 ≤ m) (hn : 2 ≤ n) (q : Cell m n) :
    kappa q = 1 ∨ kappa q = 2 ∨ kappa q = 4 := by
  rw [kappa_eq_mul hm hn]
  by_cases hr : IsEndpoint q.1 <;> by_cases hc : IsEndpoint q.2
  · rw [pathEnd_card_endpoint hm hr, pathEnd_card_endpoint hn hc]; left; rfl
  · rw [pathEnd_card_endpoint hm hr, pathEnd_card_interior hn hc]; right; left; rfl
  · rw [pathEnd_card_interior hm hr, pathEnd_card_endpoint hn hc]; right; left; rfl
  · rw [pathEnd_card_interior hm hr, pathEnd_card_interior hn hc]; right; right; rfl

/-- **κ(q) = 1 for a corner.** -/
theorem kappa_corner (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (hr : IsEndpoint q.1) (hc : IsEndpoint q.2) : kappa q = 1 := by
  rw [kappa_eq_mul hm hn, pathEnd_card_endpoint hm hr, pathEnd_card_endpoint hn hc]

/-- **κ(q) = 2 for a non-corner boundary vertex.** -/
theorem kappa_boundary (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (h : (IsEndpoint q.1 ∧ ¬ IsEndpoint q.2) ∨ (¬ IsEndpoint q.1 ∧ IsEndpoint q.2)) :
    kappa q = 2 := by
  rw [kappa_eq_mul hm hn]
  rcases h with ⟨hr, hc⟩ | ⟨hr, hc⟩
  · rw [pathEnd_card_endpoint hm hr, pathEnd_card_interior hn hc]
  · rw [pathEnd_card_interior hm hr, pathEnd_card_endpoint hn hc]

/-- **κ(q) = 4 for an interior vertex.** -/
theorem kappa_interior (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (hr : ¬ IsEndpoint q.1) (hc : ¬ IsEndpoint q.2) : kappa q = 4 := by
  rw [kappa_eq_mul hm hn, pathEnd_card_interior hm hr, pathEnd_card_interior hn hc]

/-- **Degree-2 attainment (Lemma 3.1, second half).** A corner cone has degree
exactly 2; hence degree 2 occurs in `OFG(M_{m,n})` for `m,n ≥ 2`. -/
theorem cone_corner_degree_two (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (hr : IsEndpoint q.1) (hc : IsEndpoint q.2) (b : Cell m n) :
    (neighbors (cone q b)).ncard = 2 := by
  have hmn : 2 ≤ m * n := by calc 2 ≤ 2 * 2 := by norm_num
                                _ ≤ m * n := Nat.mul_le_mul hm hn
  rw [cone_degree_eq hmn, kappa_corner hm hn hr hc]

end OrigamiCone
