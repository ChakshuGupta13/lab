import OrigamiCone.Degree3

/-!
# Non-corner boundary cell count: `2(m + n − 4)`

The lattice count underlying **Theorem 3.3** (degree-3 count `4(m + n − 4)`):

  `|\{q : Cell\,m\,n \mid \mathrm{IsBoundaryNonCorner}\,q\}| = 2(m + n − 4)`,

for every `m, n ≥ 2` (the formula is `0` at `m = n = 2`, correctly reflecting a
`2 × 2` grid that has only corners, no non-corner-boundary cells).

The paper's `thm:deg3` proof identifies this cardinality as the bottleneck step:

> *"The non-corner boundary vertices of $G_{m,n}$ number
>  $(2m + 2n - 4) - 4 = 2(m + n − 4)$."*

This module formalises that count from the existing `IsBoundaryNonCorner` and
`IsEndpoint` primitives (in `Degree3.lean` / `ConeClassification.lean`).  The
remaining bridge — converting `2(m + n − 4)` lattice cells into `4(m + n − 4)`
OFG-vertex count via the shift quotient and colour-inversion bijection — is
deferred.

Proof structure:
1. `card_isEndpoint_fin` — `|\{a : Fin k \mid \mathrm{IsEndpoint}\,a\}| = 2`
   for `k ≥ 2` (the two cells `0` and `k − 1`).
2. `card_not_isEndpoint_fin` — `|\{a : Fin k \mid \neg \mathrm{IsEndpoint}\,a\}|
   = k − 2` (complement count).
3. `card_filter_prod_and` — generic lemma converting a product-set filter on a
   conjunction into a product of single-axis filter cardinalities.
4. `card_boundaryNonCorner` — combine via the disjoint union decomposition
   `IsBoundaryNonCorner = A ∨ B` with `A ⊓ B = ∅`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-! ## Decidability instances for the predicates -/

/-- `IsEndpoint a` (`a.val = 0 ∨ a.val + 1 = k`) is decidable. -/
instance {k : ℕ} (a : Fin k) : Decidable (IsEndpoint a) :=
  show Decidable (a.val = 0 ∨ a.val + 1 = k) from inferInstance

/-- `IsBoundaryNonCorner q` is decidable (conjunctions of decidable
`IsEndpoint` predicates). -/
instance {m n : ℕ} (q : Cell m n) : Decidable (IsBoundaryNonCorner q) :=
  show Decidable ((IsEndpoint q.1 ∧ ¬ IsEndpoint q.2)
                    ∨ (¬ IsEndpoint q.1 ∧ IsEndpoint q.2)) from inferInstance

/-! ## 1-D endpoint counts -/

/-- For `k ≥ 2`, the number of endpoints of `Fin k` is `2` (the two cells
indexed `0` and `k − 1`). -/
lemma card_isEndpoint_fin (k : ℕ) (hk : 2 ≤ k) :
    (Finset.univ.filter (IsEndpoint : Fin k → Prop)).card = 2 := by
  have h0 : 0 < k := by omega
  have hk1 : k - 1 < k := by omega
  -- The filter equals `{⟨0, _⟩, ⟨k - 1, _⟩}`, which has cardinality 2 since the
  -- two elements are distinct for `k ≥ 2`.
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
    -- `Fin.val` reduces `⟨0, _⟩.val` to `0` and `⟨k - 1, _⟩.val` to `k - 1`.
    have hval : (0 : ℕ) = k - 1 := congrArg Fin.val hc
    omega), Finset.card_singleton]

/-- For `k ≥ 2`, the number of non-endpoint (interior) cells of `Fin k` is
`k − 2` (complement of the two endpoints). -/
lemma card_not_isEndpoint_fin (k : ℕ) (hk : 2 ≤ k) :
    (Finset.univ.filter (fun a : Fin k => ¬ IsEndpoint a)).card = k - 2 := by
  -- Use the complement identity:
  --   `(filter p).card + (filter ¬p).card = univ.card`.
  have huniv : (Finset.univ : Finset (Fin k)).card = k := by
    simp [Finset.card_univ, Fintype.card_fin]
  have hcomplement := Finset.card_filter_add_card_filter_not
    (s := (Finset.univ : Finset (Fin k))) (p := IsEndpoint)
  rw [card_isEndpoint_fin k hk, huniv] at hcomplement
  omega

/-! ## Generic product-filter cardinality -/

/-- For a `α × β` filter by a conjunction `P q.1 ∧ Q q.2`, the cardinality is
the product of the two 1-D filter cardinalities.  Same pattern as `kappa_eq_mul`
(`ConeClassification.lean`) but for arbitrary 1-D predicates. -/
private lemma card_filter_prod_and {α β : Type*} [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    (P : α → Prop) (Q : β → Prop) [DecidablePred P] [DecidablePred Q] :
    (Finset.univ.filter (fun q : α × β => P q.1 ∧ Q q.2)).card
      = (Finset.univ.filter P).card * (Finset.univ.filter Q).card := by
  rw [← Finset.card_product]
  congr 1
  ext v
  simp [Finset.mem_filter, Finset.mem_product]

/-! ## The main count -/

/-- **Lattice count of non-corner boundary cells** in `Cell m n`.  For
`m, n ≥ 2`, the number of cells `q : Cell m n` satisfying `IsBoundaryNonCorner q`
is `2(m + n − 4)`.

This is the cardinality the paper's proof of `thm:deg3` invokes verbatim:
"*the non-corner boundary vertices of $G_{m,n}$ number $2(m + n − 4)$*"
(combined with the colour-inversion factor of 2 it yields the headline
`4(m + n − 4)`). -/
theorem card_boundaryNonCorner (hm : 2 ≤ m) (hn : 2 ≤ n) :
    (Finset.univ.filter (IsBoundaryNonCorner : Cell m n → Prop)).card
      = 2 * (m + n - 4) := by
  -- Step 1: rewrite the filter as a disjunction split on the two clauses of
  -- `IsBoundaryNonCorner`.
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
  -- Step 2: the two halves are disjoint (one requires `IsEndpoint q.1`,
  -- the other requires `¬ IsEndpoint q.1`).
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
  -- Step 3: convert each half to a product of 1-D filter counts.
  rw [card_filter_prod_and (α := Fin m) (β := Fin n)
        (fun a => IsEndpoint a) (fun b => ¬ IsEndpoint b)]
  rw [card_filter_prod_and (α := Fin m) (β := Fin n)
        (fun a => ¬ IsEndpoint a) (fun b => IsEndpoint b)]
  -- Step 4: substitute the 1-D counts.
  rw [card_isEndpoint_fin m hm, card_not_isEndpoint_fin n hn,
      card_not_isEndpoint_fin m hm, card_isEndpoint_fin n hn]
  -- Step 5: `2 * (n - 2) + (m - 2) * 2 = 2 * (m + n - 4)`, given `m, n ≥ 2`.
  omega

end OrigamiCone
