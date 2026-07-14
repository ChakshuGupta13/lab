import OrigamiCone.Cone
import OrigamiCone.MinDegree

/-!
# Cone degree = 1 + κ  (Corollary 2.3, first part)

Formalisation of the opening identity of the Cone Classification (Corollary 2.3):

> By the degree–extrema correspondence the degree of `h_q` equals the number of
> its local extrema; `q` is the unique maximum, and a vertex is a local minimum
> of `h_q` exactly when it is a local maximum of `d(q,·)`, so the degree is
> `1 + κ(q)`, where `κ(q)` is the number of strict local maxima of `d(q,·)`.

Here `κ(q)` (`kappa q`) is defined exactly as in the paper: the number of strict
local maxima of the distance function `D_q = d(q,·)`.  The trichotomy
`κ(q) ∈ {1,2,4}` (corner / non-corner-boundary / interior) and the global cone
counts are the *second* part of the corollary; this module establishes the
degree identity they build on.

**Quotient caveat** (see `DegreeExtrema`): `cone_degree_eq` is proved at `mn ≥ 2`
for the *unquotiented* height-flip graph, since it rests on `degree_eq_extrema`.
At `M_{1,2}` (`mn = 2`) the unquotiented cone has degree `1 + κ(q) = 2`, whereas
the paper's quotient OFG vertex has degree `1`; the two coincide for `mn ≥ 3`.
Every consumer works at `m, n ≥ 2` (so `mn ≥ 4 ≥ 3`), inside the agreement
regime, so the cone counts it feeds faithfully describe the paper's OFG.

Main result: `cone_degree_eq` — for `mn ≥ 2`, the cone `cone q b` has degree
`1 + κ(q)`.  No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- The distance function `D_q = d(q,·)`, whose strict local maxima the paper
counts as `κ(q)`. -/
def Dq (q : Cell m n) : Cell m n → ℤ := fun v => gdist q v

/-- `D_q` is a height function: distance to a fixed cell changes by exactly one
across each edge. -/
lemma Dq_isHeight (q : Cell m n) : IsHeight (Dq q) := by
  intro p p' hpp'
  rw [abs_eq (by norm_num : (0 : ℤ) ≤ 1)]
  simp only [Dq]
  rcases gdist_adj_step (q := q) hpp' with h | h
  · left; omega
  · right; omega

/-- A vertex is a strict local minimum of the cone `h_q` exactly when it is a
strict local maximum of the distance function `D_q`. -/
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

/-- **κ(q)**: the number of strict local maxima of the distance function
`d(q,·)`, exactly as in the paper. -/
def kappa (q : Cell m n) : ℕ :=
  (Finset.univ.filter (IsStrictLocalMax (Dq q))).card

/-- The strict local maxima of the cone `h_q` are exactly `{q}`. -/
lemma cone_strictMax_singleton (q b : Cell m n) :
    Finset.univ.filter (IsStrictLocalMax (cone q b)) = {q} := by
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  exact ⟨fun hv => cone_unique_max q b v hv, fun h => h ▸ cone_max_at q b⟩

/-- The strict local minima of the cone `h_q` are exactly the strict local maxima
of `d(q,·)`, so they number `κ(q)`. -/
lemma cone_strictMin_card (q b : Cell m n) :
    (Finset.univ.filter (IsStrictLocalMin (cone q b))).card = kappa q := by
  unfold kappa
  congr 1
  apply Finset.filter_congr
  intro v _
  simp only [cone_min_iff_Dq_max]

/-- **Corollary 2.3 (degree identity).** For `mn ≥ 2`, the cone `cone q b` has
degree `1 + κ(q)`: its unique strict local maximum is `q`, and its strict local
minima are the `κ(q)` strict local maxima of `d(q,·)`. -/
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
