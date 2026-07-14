import OrigamiCone.Basic

/-!
# Degree–extrema correspondence (Lemma 2.1)

Formalisation of **Lemma 2.1** of the paper, in the *unquotiented* height-flip
model: for `mn ≥ 2`, the degree of a height function (its number of single-cell
flip neighbours) equals the number of strict local extrema of `h`.

We work in the height-function model the paper itself uses after the §2
reductions (the Ginepro–Hull bijection plus the bipartite height lift, both
cited): a vertex is a height function `h`, and two height functions are
**adjacent** when one is obtained from the other by changing the value at a
single cell (`OFGAdj`).  We do *not* reformalise the origami crease-pattern
definition; that is the content of the cited bijection `(eq:iso)`.

The mathematical content of Lemma 2.1 is then a finite bijection: flipping a
strict local maximum lowers it by `2` (turning it into a strict local minimum),
flipping a strict local minimum raises it by `2`, and the map
`extremum ↦ flipped height function` is a bijection from the set of strict local
extrema onto the neighbour set.  Hence the degree (the size of the neighbour set)
equals the number of strict local extrema.

The hypothesis `mn ≥ 2` guarantees every cell has a neighbour, so no cell is
simultaneously a strict local maximum and minimum (`max_min_excl`).

**Quotient caveat.** This `OFGAdj` graph is the height-flip reconfiguration
graph *before* the global colour rotation `γ ↦ γ + 1` is quotiented out; the
paper's OFG is the quotient `R₃(G)/(ℤ/3ℤ)` (`eq:iso`).  The two have equal vertex
degrees for `mn ≥ 3`, but they diverge at the single grid `M_{1,2}` (`mn = 2`):
there the unquotiented model has degree `2` (what is proved here), while the
paper's OFG has degree `1`, because the rotation identifies the two flips of the
lone edge.  The paper therefore states Lemma 2.1 for `mn ≥ 3`.  The theorem below
is proved at the weaker `mn ≥ 2` as a true statement about the unquotiented
model; every result that consumes it does so at `m, n ≥ 2` (hence `mn ≥ 4 ≥ 3`),
inside the regime where the two graphs coincide, so the downstream
degree-sequence theorems faithfully describe the paper's OFG.

Main result: `degree_eq_extrema`.  No `sorry`; check `#print axioms
degree_eq_extrema`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- Adjacency is symmetric. -/
lemma adj_comm {p q : Cell m n} : adj p q ↔ adj q p := by
  unfold adj; rw [gdist_comm]

instance decAdj (p q : Cell m n) : Decidable (adj p q) := by
  unfold adj; infer_instance

instance decStrictLocalMax (h : Cell m n → ℤ) (v : Cell m n) :
    Decidable (IsStrictLocalMax h v) := by unfold IsStrictLocalMax; infer_instance

instance decStrictLocalMin (h : Cell m n → ℤ) (v : Cell m n) :
    Decidable (IsStrictLocalMin h v) := by unfold IsStrictLocalMin; infer_instance

/-- A vertex is a strict local extremum if it is a strict local maximum or a
strict local minimum. -/
def IsStrictLocalExtremum (h : Cell m n → ℤ) (v : Cell m n) : Prop :=
  IsStrictLocalMax h v ∨ IsStrictLocalMin h v

instance decStrictLocalExtremum (h : Cell m n → ℤ) (v : Cell m n) :
    Decidable (IsStrictLocalExtremum h v) := by
  unfold IsStrictLocalExtremum; infer_instance

/-- When the grid has at least two cells, every cell has a neighbour. -/
lemma exists_neighbor (hmn : 2 ≤ m * n) (v : Cell m n) : ∃ u, adj v u := by
  haveI : Nontrivial (Cell m n) := by
    rw [← Fintype.one_lt_card_iff_nontrivial]
    simpa [Fintype.card_prod, Fintype.card_fin] using hmn
  obtain ⟨w, hw⟩ := exists_ne v
  obtain ⟨u, hu, _⟩ := exists_step_toward (Ne.symm hw)
  exact ⟨u, hu⟩

/-- A cell cannot be both a strict local maximum and a strict local minimum once
it has a neighbour. -/
lemma max_min_excl (hmn : 2 ≤ m * n) {h : Cell m n → ℤ} {v : Cell m n}
    (hmax : IsStrictLocalMax h v) (hmin : IsStrictLocalMin h v) : False := by
  obtain ⟨u, hu⟩ := exists_neighbor hmn v
  have e1 := hmax u hu
  have e2 := hmin u hu
  omega

/-- Flipping a strict local maximum: lower its value by two. -/
def flipMax (h : Cell m n → ℤ) (v : Cell m n) : Cell m n → ℤ :=
  fun w => if w = v then h v - 2 else h w

/-- Flipping a strict local minimum: raise its value by two. -/
def flipMin (h : Cell m n → ℤ) (v : Cell m n) : Cell m n → ℤ :=
  fun w => if w = v then h v + 2 else h w

/-- The flip at a strict local extremum: lower a maximum, raise a minimum. -/
def flipAt (h : Cell m n → ℤ) (v : Cell m n) : Cell m n → ℤ :=
  if IsStrictLocalMax h v then flipMax h v else flipMin h v

@[simp] lemma flipMax_apply_self {h : Cell m n → ℤ} {v : Cell m n} :
    flipMax h v v = h v - 2 := by simp [flipMax]

lemma flipMax_apply_ne {h : Cell m n → ℤ} {v w : Cell m n} (hw : w ≠ v) :
    flipMax h v w = h w := by simp [flipMax, hw]

@[simp] lemma flipMin_apply_self {h : Cell m n → ℤ} {v : Cell m n} :
    flipMin h v v = h v + 2 := by simp [flipMin]

lemma flipMin_apply_ne {h : Cell m n → ℤ} {v w : Cell m n} (hw : w ≠ v) :
    flipMin h v w = h w := by simp [flipMin, hw]

lemma flipAt_apply_ne {h : Cell m n → ℤ} {v w : Cell m n} (hw : w ≠ v) :
    flipAt h v w = h w := by
  unfold flipAt
  split_ifs
  · exact flipMax_apply_ne hw
  · exact flipMin_apply_ne hw

/-- Changing the value at a single cell `v` to `c` keeps a height function, as
long as the new value still differs by one from every neighbour of `v`. -/
lemma isHeight_update {h : Cell m n → ℤ} (hh : IsHeight h) {v : Cell m n} {c : ℤ}
    (hc : ∀ u, adj v u → |c - h u| = 1) :
    IsHeight (fun w => if w = v then c else h w) := by
  intro p q hpq
  by_cases hpv : p = v
  · by_cases hqv : q = v
    · exfalso; rw [hpv, hqv] at hpq; simp [adj, gdist_self] at hpq
    · subst hpv
      simp only [if_neg hqv]
      exact hc q hpq
  · by_cases hqv : q = v
    · subst hqv
      simp only [if_neg hpv]
      rw [abs_sub_comm]
      exact hc p (adj_comm.mp hpq)
    · simp only [if_neg hpv, if_neg hqv]
      exact hh p q hpq

lemma isHeight_flipMax {h : Cell m n → ℤ} (hh : IsHeight h) {v : Cell m n}
    (hv : IsStrictLocalMax h v) : IsHeight (flipMax h v) := by
  unfold flipMax
  apply isHeight_update hh
  intro u hu
  rw [hv u hu, show h v - 2 - (h v - 1) = -1 from by ring]
  norm_num

lemma isHeight_flipMin {h : Cell m n → ℤ} (hh : IsHeight h) {v : Cell m n}
    (hv : IsStrictLocalMin h v) : IsHeight (flipMin h v) := by
  unfold flipMin
  apply isHeight_update hh
  intro u hu
  rw [hv u hu, show h v + 2 - (h v + 1) = 1 from by ring]
  norm_num

/-- OFG adjacency in the height-function model: `h'` differs from the height
function `h` at exactly one cell and is itself a height function. -/
def OFGAdj (h h' : Cell m n → ℤ) : Prop :=
  IsHeight h' ∧ ∃ v, (∀ w, w ≠ v → h w = h' w) ∧ h v ≠ h' v

/-- The neighbour set of `h` in the flip graph. -/
def neighbors (h : Cell m n → ℤ) : Set (Cell m n → ℤ) := {h' | OFGAdj h h'}

/-- **Flipping an extremum yields a neighbour.** -/
lemma ofgAdj_flipAt (hmn : 2 ≤ m * n) {h : Cell m n → ℤ} (hh : IsHeight h)
    {v : Cell m n} (hv : IsStrictLocalExtremum h v) : OFGAdj h (flipAt h v) := by
  rcases hv with hmax | hmin
  · have hfa : flipAt h v = flipMax h v := by simp [flipAt, hmax]
    refine ⟨?_, v, ?_, ?_⟩
    · rw [hfa]; exact isHeight_flipMax hh hmax
    · intro w hw; rw [hfa]; exact (flipMax_apply_ne hw).symm
    · rw [hfa, flipMax_apply_self]; omega
  · have hnotmax : ¬ IsStrictLocalMax h v := fun hm => max_min_excl hmn hm hmin
    have hfa : flipAt h v = flipMin h v := by simp [flipAt, hnotmax]
    refine ⟨?_, v, ?_, ?_⟩
    · rw [hfa]; exact isHeight_flipMin hh hmin
    · intro w hw; rw [hfa]; exact (flipMin_apply_ne hw).symm
    · rw [hfa, flipMin_apply_self]; omega

/-- **Every neighbour is the flip of a unique extremum.** -/
lemma neighbor_is_flipAt (hmn : 2 ≤ m * n) {h : Cell m n → ℤ} (hh : IsHeight h)
    {h' : Cell m n → ℤ} (hAdj : OFGAdj h h') :
    ∃ v, IsStrictLocalExtremum h v ∧ flipAt h v = h' := by
  obtain ⟨hh', v, hagree, hvne⟩ := hAdj
  obtain ⟨u₀, hu₀⟩ := exists_neighbor hmn v
  have hu₀ne : u₀ ≠ v := by rintro rfl; simp [adj, gdist_self] at hu₀
  have hagr0 : h u₀ = h' u₀ := hagree u₀ hu₀ne
  have hb1 : |h v - h u₀| = 1 := hh v u₀ hu₀
  have hb2 : |h' v - h u₀| = 1 := by rw [hagr0]; exact hh' v u₀ hu₀
  have hcase : h' v = h v + 2 ∨ h' v = h v - 2 := by
    rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 hb1 with c1 | c1 <;>
    rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 hb2 with c2 | c2 <;> omega
  rcases hcase with hc | hc
  · -- `h' v = h v + 2`: `v` is a strict local minimum.
    have hmin : IsStrictLocalMin h v := by
      intro u hu
      have hunev : u ≠ v := by rintro rfl; simp [adj, gdist_self] at hu
      have a1 : |h v - h u| = 1 := hh v u hu
      have a2 : |h' v - h' u| = 1 := hh' v u hu
      have hagu : h u = h' u := hagree u hunev
      rw [← hagu, hc] at a2
      rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 a1 with b1 | b1 <;>
      rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 a2 with b2 | b2 <;> omega
    have hnotmax : ¬ IsStrictLocalMax h v := fun hm => max_min_excl hmn hm hmin
    refine ⟨v, Or.inr hmin, ?_⟩
    have hfa : flipAt h v = flipMin h v := by simp [flipAt, hnotmax]
    rw [hfa]; funext w
    by_cases hwv : w = v
    · subst hwv; rw [flipMin_apply_self]; omega
    · rw [flipMin_apply_ne hwv]; exact hagree w hwv
  · -- `h' v = h v - 2`: `v` is a strict local maximum.
    have hmax : IsStrictLocalMax h v := by
      intro u hu
      have hunev : u ≠ v := by rintro rfl; simp [adj, gdist_self] at hu
      have a1 : |h v - h u| = 1 := hh v u hu
      have a2 : |h' v - h' u| = 1 := hh' v u hu
      have hagu : h u = h' u := hagree u hunev
      rw [← hagu, hc] at a2
      rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 a1 with b1 | b1 <;>
      rcases (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 a2 with b2 | b2 <;> omega
    refine ⟨v, Or.inl hmax, ?_⟩
    have hfa : flipAt h v = flipMax h v := by simp [flipAt, hmax]
    rw [hfa]; funext w
    by_cases hwv : w = v
    · subst hwv; rw [flipMax_apply_self]; omega
    · rw [flipMax_apply_ne hwv]; exact hagree w hwv

/-- The flip changes the value at its own cell. -/
lemma flipAt_self_ne (hmn : 2 ≤ m * n) {h : Cell m n → ℤ} {v : Cell m n}
    (hv : IsStrictLocalExtremum h v) : flipAt h v v ≠ h v := by
  rcases hv with hmax | hmin
  · have hfa : flipAt h v = flipMax h v := by simp [flipAt, hmax]
    rw [hfa, flipMax_apply_self]; omega
  · have hnotmax : ¬ IsStrictLocalMax h v := fun hm => max_min_excl hmn hm hmin
    have hfa : flipAt h v = flipMin h v := by simp [flipAt, hnotmax]
    rw [hfa, flipMin_apply_self]; omega

/-- The flip map is injective on the set of strict local extrema. -/
lemma flipAt_injOn (hmn : 2 ≤ m * n) {h : Cell m n → ℤ} :
    Set.InjOn (flipAt h) {v | IsStrictLocalExtremum h v} := by
  intro v1 hv1 v2 _ heq
  by_contra hne
  have d1 : flipAt h v1 v1 ≠ h v1 := flipAt_self_ne hmn hv1
  have e1 : flipAt h v1 v1 = h v1 := by
    rw [congrFun heq v1]; exact flipAt_apply_ne hne
  exact d1 e1

/-- **Lemma 2.1 (Degree–extrema correspondence).** For `mn ≥ 2`, the degree of a
height function `h` (the number of its flip-graph neighbours) equals the number
of strict local extrema of `h`. -/
theorem degree_eq_extrema {h : Cell m n → ℤ} (hh : IsHeight h) (hmn : 2 ≤ m * n) :
    (neighbors h).ncard
      = (Finset.univ.filter (IsStrictLocalExtremum h)).card := by
  have hset : neighbors h
      = ↑((Finset.univ.filter (IsStrictLocalExtremum h)).image (flipAt h)) := by
    ext h'
    constructor
    · intro hAdj
      obtain ⟨v, hv, hfl⟩ := neighbor_is_flipAt hmn hh hAdj
      rw [Finset.coe_image, Set.mem_image]
      refine ⟨v, ?_, hfl⟩
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and]
      exact hv
    · intro hmem
      rw [Finset.coe_image, Set.mem_image] at hmem
      obtain ⟨v, hv, rfl⟩ := hmem
      rw [Finset.mem_coe, Finset.mem_filter] at hv
      exact ofgAdj_flipAt hmn hh hv.2
  rw [hset, Set.ncard_coe_finset]
  apply Finset.card_image_of_injOn
  intro v1 hv1 v2 hv2 heq
  have m1 : v1 ∈ {v | IsStrictLocalExtremum h v} := by simpa using hv1
  have m2 : v2 ∈ {v | IsStrictLocalExtremum h v} := by simpa using hv2
  exact flipAt_injOn hmn m1 m2 heq

end OrigamiCone
