import OrigamiCone.ConeClassification

/-!
# Degree-2 characterization (Theorem 3.2)

Formalisation of the characterization underlying **Theorem 3.2**:

> For `m,n ≥ 2`, the degree-2 vertices of `OFG(M_{m,n})` are exactly the four
> corner gradients `h_ε(i,j) = ε₁(i-1) + ε₂(j-1)`.

A corner gradient is precisely a distance cone whose apex is a corner: at corner
`q` the cone `coneC q C v = C - d(q,v)` is linear (a gradient), with its unique
maximum at `q` and unique minimum at the opposite corner.  This module proves the
shift-invariant content of the theorem:

* `degree_two_iff_corner` — a height function has degree 2 **iff** it is a cone
  `coneC q C` with `q` a corner (both coordinates endpoints).

The "exactly four" cardinality is the grid-enumeration layer on top (four corners
give four normalized gradients); the characterization is the mathematical core
and is what the Cone Lemma + Cone Classification deliver.

`coneC q C` generalizes the cone of `OrigamiCone.Cone` to an arbitrary integer
constant `C` (the cone there fixes `C = d(q, basepoint)`); this is needed because
a degree-2 height function appears with whatever additive constant it carries.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- A cone with an explicit integer constant: `coneC q C v = C - d(q,v)`.
`OrigamiCone.cone q b = coneC q (d(q,b))`. -/
def coneC (q : Cell m n) (C : ℤ) : Cell m n → ℤ := fun v => C - gdist q v

lemma cone_eq_coneC (q b : Cell m n) : cone q b = coneC q (gdist q b) := rfl

/-- `coneC q C` is a height function. -/
lemma coneC_isHeight (q : Cell m n) (C : ℤ) : IsHeight (coneC q C) := by
  intro p p' hpp'
  rw [abs_eq (by norm_num : (0 : ℤ) ≤ 1)]
  simp only [coneC]
  rcases gdist_adj_step (q := q) hpp' with h | h
  · right; omega
  · left; omega

/-- `q` is a strict local maximum of `coneC q C`. -/
lemma coneC_max_at (q : Cell m n) (C : ℤ) : IsStrictLocalMax (coneC q C) q := by
  intro u hu
  have hu1 : gdist q u = 1 := hu
  have h0 : gdist q q = 0 := gdist_self q
  simp only [coneC]
  omega

/-- `q` is the only strict local maximum of `coneC q C`. -/
lemma coneC_unique_max (q : Cell m n) (C : ℤ) :
    ∀ q', IsStrictLocalMax (coneC q C) q' → q' = q := by
  intro q' hq'
  by_contra hne
  obtain ⟨u, hadj, hd⟩ := exists_step_toward (show q' ≠ q from hne)
  have hval := hq' u hadj
  simp only [coneC] at hval
  have hc1 : gdist q u = gdist u q := gdist_comm q u
  have hc2 : gdist q q' = gdist q' q := gdist_comm q q'
  omega

/-- A strict local minimum of `coneC q C` is exactly a strict local maximum of
the distance function `d(q,·)`. -/
lemma coneC_min_iff_Dq_max (q : Cell m n) (C : ℤ) (v : Cell m n) :
    IsStrictLocalMin (coneC q C) v ↔ IsStrictLocalMax (Dq q) v := by
  constructor
  · intro h u hu
    have := h u hu
    simp only [coneC, Dq] at this ⊢
    omega
  · intro h u hu
    have := h u hu
    simp only [coneC, Dq] at this ⊢
    omega

/-- **General extrema split.** For `mn ≥ 2`, the number of strict local extrema of
a height function is the number of its strict local maxima plus the number of its
strict local minima (they are disjoint, since no cell is both). -/
lemma extrema_card_split (hmn : 2 ≤ m * n) {h : Cell m n → ℤ} :
    (Finset.univ.filter (IsStrictLocalExtremum h)).card
      = (Finset.univ.filter (IsStrictLocalMax h)).card
        + (Finset.univ.filter (IsStrictLocalMin h)).card := by
  have hsplit : Finset.univ.filter (IsStrictLocalExtremum h)
      = Finset.univ.filter (IsStrictLocalMax h)
        ∪ Finset.univ.filter (IsStrictLocalMin h) := by
    ext v
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union,
               IsStrictLocalExtremum]
  have hdisj : Disjoint (Finset.univ.filter (IsStrictLocalMax h))
      (Finset.univ.filter (IsStrictLocalMin h)) := by
    rw [Finset.disjoint_left]
    intro v hvmax hvmin
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hvmax hvmin
    exact max_min_excl hmn hvmax hvmin
  rw [hsplit, Finset.card_union_of_disjoint hdisj]

/-- **Degree of a `coneC`.** For `mn ≥ 2`, the cone `coneC q C` has degree
`1 + κ(q)`. -/
lemma coneC_degree (hmn : 2 ≤ m * n) (q : Cell m n) (C : ℤ) :
    (neighbors (coneC q C)).ncard = 1 + kappa q := by
  rw [degree_eq_extrema (coneC_isHeight q C) hmn, extrema_card_split hmn]
  have hmax : (Finset.univ.filter (IsStrictLocalMax (coneC q C))).card = 1 := by
    rw [show Finset.univ.filter (IsStrictLocalMax (coneC q C)) = {q} from ?_,
        Finset.card_singleton]
    ext v
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    exact ⟨fun hv => coneC_unique_max q C v hv, fun h => h ▸ coneC_max_at q C⟩
  have hmin : (Finset.univ.filter (IsStrictLocalMin (coneC q C))).card = kappa q := by
    unfold kappa
    congr 1
    apply Finset.filter_congr
    intro v _
    simp only [coneC_min_iff_Dq_max]
  rw [hmax, hmin]

/-- A 1-D filter of size one forces the apex to an endpoint. -/
lemma pathEnd_card_one_imp_endpoint {k : ℕ} (hk : 2 ≤ k) {a : Fin k}
    (h : (Finset.univ.filter (PathEnd a)).card = 1) : IsEndpoint a := by
  by_contra hne
  rw [pathEnd_card_interior hk hne] at h
  omega

/-- **κ(q) = 1 forces a corner.** -/
lemma kappa_one_imp_corner (hm : 2 ≤ m) (hn : 2 ≤ n) {q : Cell m n}
    (h : kappa q = 1) : IsEndpoint q.1 ∧ IsEndpoint q.2 := by
  rw [kappa_eq_mul hm hn] at h
  have hr1 : (Finset.univ.filter (PathEnd q.1)).card = 1 :=
    Nat.eq_one_of_mul_eq_one_right h
  have hc1 : (Finset.univ.filter (PathEnd q.2)).card = 1 :=
    Nat.eq_one_of_mul_eq_one_left h
  exact ⟨pathEnd_card_one_imp_endpoint hm hr1, pathEnd_card_one_imp_endpoint hn hc1⟩

/-- **Theorem 3.2 (degree-2 characterization).** For `m,n ≥ 2`, a height function
`h` has degree 2 iff it is a cone `coneC q C` whose apex `q` is a corner (both
coordinates endpoints). These are exactly the corner gradients. -/
theorem degree_two_iff_corner (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h) :
    (neighbors h).ncard = 2
      ↔ ∃ (q : Cell m n) (C : ℤ), IsEndpoint q.1 ∧ IsEndpoint q.2 ∧ h = coneC q C := by
  have hmn : 2 ≤ m * n := le_trans (by norm_num) (Nat.mul_le_mul hm hn)
  have hm0 : 0 < m := by omega
  have hn0 : 0 < n := by omega
  haveI : Nonempty (Cell m n) := ⟨(⟨0, hm0⟩, ⟨0, hn0⟩)⟩
  constructor
  · intro hdeg0
    -- degree 2 ⟹ #max + #min = 2
    have hsum : (Finset.univ.filter (IsStrictLocalMax h)).card
        + (Finset.univ.filter (IsStrictLocalMin h)).card = 2 := by
      rw [← extrema_card_split hmn, ← degree_eq_extrema hh hmn]; exact hdeg0
    obtain ⟨wm, hwm⟩ := exists_strictLocalMax hh
    obtain ⟨wn, hwn⟩ := exists_strictLocalMin hh
    have hmax_pos : 0 < (Finset.univ.filter (IsStrictLocalMax h)).card :=
      Finset.card_pos.mpr ⟨wm, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hwm⟩⟩
    have hmin_pos : 0 < (Finset.univ.filter (IsStrictLocalMin h)).card :=
      Finset.card_pos.mpr ⟨wn, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hwn⟩⟩
    have hmax1 : (Finset.univ.filter (IsStrictLocalMax h)).card = 1 := by omega
    obtain ⟨q, hq⟩ := Finset.card_eq_one.mp hmax1
    have huniq : ∀ q', IsStrictLocalMax h q' → q' = q := by
      intro q' hq'
      have hmem : q' ∈ Finset.univ.filter (IsStrictLocalMax h) :=
        Finset.mem_filter.mpr ⟨Finset.mem_univ _, hq'⟩
      rw [hq, Finset.mem_singleton] at hmem
      exact hmem
    have hapex := cone_max hh huniq
    have happ : h = coneC q (h q) := by
      funext v; simp only [coneC]; exact hapex v
    have hdeg2 : (neighbors h).ncard = 1 + kappa q := by
      rw [happ]; exact coneC_degree hmn q (h q)
    rw [hdeg0] at hdeg2
    have hk1 : kappa q = 1 := by omega
    obtain ⟨hr, hc⟩ := kappa_one_imp_corner hm hn hk1
    exact ⟨q, h q, hr, hc, happ⟩
  · rintro ⟨q, C, hr, hc, rfl⟩
    rw [coneC_degree hmn q C, kappa_corner hm hn hr hc]

end OrigamiCone
