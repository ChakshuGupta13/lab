import Mathlib

/-!
# The Cone Lemma for grid height functions

Formalisation of **Lemma 2.2 (Cone Lemma)** of the paper
*"The origami flip graph of the m × n Miura-ori: degree sequence and diameter
via height functions"*.

A *height function* on the `m × n` grid is an integer-valued function whose value
changes by **exactly one** across every grid edge — that is, across every pair of
cells at Manhattan distance `1`.  The Cone Lemma states:

> If `h` has a **unique** strict local maximum `q`, then `h v = h q - d(q,v)` for
> every cell `v`; dually, a unique strict local minimum `p` gives
> `h v = h p + d(p,v)`.

Here the grid is `Fin m × Fin n` and `d` is the Manhattan (grid) distance.

Finiteness of the grid is essential and is used exactly once, in
`unique_max_global`: it forces the unique *local* maximum to be the *global*
maximum, so the height deficit `h q - h v` is a nonnegative integer on which the
cone direction is proved by induction.

The two theorems `cone_max` and `cone_min` are the formal content of Lemma 2.2.
No `sorry`; check with `#print axioms cone_max`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- Cells of the `m × n` grid. -/
abbrev Cell (m n : ℕ) := Fin m × Fin n

/-- Manhattan (grid) distance between two cells, as an integer.  Defined through
`Int.natAbs` so that `omega` can reason about it directly. -/
def gdist (p q : Cell m n) : ℤ :=
  ((((p.1.val : ℤ) - q.1.val).natAbs + ((p.2.val : ℤ) - q.2.val).natAbs : ℕ) : ℤ)

lemma gdist_nonneg (p q : Cell m n) : 0 ≤ gdist p q := by unfold gdist; omega

lemma gdist_self (p : Cell m n) : gdist p p = 0 := by unfold gdist; omega

lemma gdist_comm (p q : Cell m n) : gdist p q = gdist q p := by unfold gdist; omega

lemma gdist_triangle (a b c : Cell m n) : gdist a c ≤ gdist a b + gdist b c := by
  unfold gdist; omega

lemma gdist_eq_zero {p q : Cell m n} : gdist p q = 0 ↔ p = q := by
  constructor
  · intro h
    unfold gdist at h
    have e1 : p.1.val = q.1.val := by omega
    have e2 : p.2.val = q.2.val := by omega
    exact Prod.ext_iff.2 ⟨Fin.ext e1, Fin.ext e2⟩
  · intro h; subst h; exact gdist_self p

/-- Distinct cells are at distance at least one. -/
lemma gdist_pos {p q : Cell m n} (hpq : p ≠ q) : 1 ≤ gdist p q := by
  have h0 := gdist_nonneg p q
  have hne : gdist p q ≠ 0 := fun h => hpq (gdist_eq_zero.1 h)
  omega

/-- Adjacency in the grid graph: cells at Manhattan distance one. -/
def adj (p q : Cell m n) : Prop := gdist p q = 1

/-- A height function: integer-valued, changing by exactly one across each edge. -/
def IsHeight (h : Cell m n → ℤ) : Prop := ∀ p q, adj p q → |h p - h q| = 1

/-- `q` is a strict local maximum: every neighbour is exactly one lower. -/
def IsStrictLocalMax (h : Cell m n → ℤ) (q : Cell m n) : Prop :=
  ∀ u, adj q u → h u = h q - 1

/-- `p` is a strict local minimum: every neighbour is exactly one higher. -/
def IsStrictLocalMin (h : Cell m n → ℤ) (p : Cell m n) : Prop :=
  ∀ u, adj p u → h u = h p + 1

/-- **Step toward a target.** From any cell `p ≠ q` there is a neighbour of `p`
strictly closer to `q`: one step that decreases the Manhattan distance by one.
This is the only place the bounded grid geometry is used directly. -/
lemma exists_step_toward {p q : Cell m n} (hpq : p ≠ q) :
    ∃ p', adj p p' ∧ gdist p' q = gdist p q - 1 := by
  have hcoord : p.1.val ≠ q.1.val ∨ p.2.val ≠ q.2.val := by
    by_contra h
    push_neg at h
    exact hpq (Prod.ext_iff.2 ⟨Fin.ext h.1, Fin.ext h.2⟩)
  rcases hcoord with hrow | hcol
  · rcases lt_or_gt_of_ne hrow with hlt | hgt
    · -- `p.1 < q.1`: increase the row by one.
      have hb : p.1.val + 1 < m := by have := q.1.isLt; omega
      refine ⟨(⟨p.1.val + 1, hb⟩, p.2), ?_, ?_⟩
      · simp only [adj, gdist]; omega
      · simp only [gdist]; omega
    · -- `p.1 > q.1`: decrease the row by one.
      have hb : p.1.val - 1 < m := by have := p.1.isLt; omega
      refine ⟨(⟨p.1.val - 1, hb⟩, p.2), ?_, ?_⟩
      · simp only [adj, gdist]; omega
      · simp only [gdist]; omega
  · rcases lt_or_gt_of_ne hcol with hlt | hgt
    · -- `p.2 < q.2`: increase the column by one.
      have hb : p.2.val + 1 < n := by have := q.2.isLt; omega
      refine ⟨(p.1, ⟨p.2.val + 1, hb⟩), ?_, ?_⟩
      · simp only [adj, gdist]; omega
      · simp only [gdist]; omega
    · -- `p.2 > q.2`: decrease the column by one.
      have hb : p.2.val - 1 < n := by have := p.2.isLt; omega
      refine ⟨(p.1, ⟨p.2.val - 1, hb⟩), ?_, ?_⟩
      · simp only [adj, gdist]; omega
      · simp only [gdist]; omega

/-- **One-Lipschitz bound** (the "shortest path" direction of the Cone Lemma).
A height function changes by at most the grid distance: `|h a - h b| ≤ d(a,b)`. -/
lemma height_lipschitz {h : Cell m n → ℤ} (hh : IsHeight h) :
    ∀ (k : ℕ) (a b : Cell m n), gdist a b ≤ k → |h a - h b| ≤ gdist a b := by
  intro k
  induction k with
  | zero =>
    intro a b hk
    have hz : gdist a b = 0 := le_antisymm hk (gdist_nonneg a b)
    have hab : a = b := gdist_eq_zero.1 hz
    subst hab; simp [gdist_self]
  | succ k ih =>
    intro a b hk
    by_cases hab : a = b
    · subst hab; simp [gdist_self]
    · obtain ⟨a', ha', hd⟩ := exists_step_toward hab
      have hb' : gdist a' b ≤ k := by rw [hd]; omega
      have hih := ih a' b hb'
      have h1 : |h a - h a'| = 1 := hh a a' ha'
      have htri : |h a - h b| ≤ |h a - h a'| + |h a' - h b| := abs_sub_le _ _ _
      rw [hd] at hih
      linarith

/-- **The unique strict local maximum is global.**  The only step using
finiteness of the grid. -/
lemma unique_max_global {h : Cell m n → ℤ} (hh : IsHeight h) {q : Cell m n}
    (huniq : ∀ q', IsStrictLocalMax h q' → q' = q) : ∀ v, h v ≤ h q := by
  haveI : Nonempty (Cell m n) := ⟨q⟩
  obtain ⟨w, hw⟩ := Finite.exists_max h
  have hwmax : IsStrictLocalMax h w := by
    intro u hu
    have hle := hw u
    have h1 := hh w u hu
    rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega
  have hwq : w = q := huniq w hwmax
  subst hwq
  exact hw

/-- **The ascending direction of the Cone Lemma.** Given a unique strict local
maximum `q` (hence global), an ascending path from any cell reaches `q`, so
`d(q,v) ≤ h q - h v`.  Induction on the height deficit `(h q - h v).toNat`. -/
lemma cone_ascend {h : Cell m n → ℤ} (hh : IsHeight h) {q : Cell m n}
    (huniq : ∀ q', IsStrictLocalMax h q' → q' = q) (hglob : ∀ v, h v ≤ h q) :
    ∀ (k : ℕ) (v : Cell m n), (h q - h v).toNat ≤ k → gdist q v ≤ h q - h v := by
  intro k
  induction k with
  | zero =>
    intro v hk
    have heq : h v = h q := by
      have h2 := hglob v
      have hz : (h q - h v).toNat = 0 := Nat.le_zero.1 hk
      omega
    have hvmax : IsStrictLocalMax h v := by
      intro u hu
      have hle := hglob u
      have h1 := hh v u hu
      rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega
    have hvq : v = q := huniq v hvmax
    subst hvq
    simp [gdist_self]
  | succ k ih =>
    intro v hk
    by_cases hvq : v = q
    · subst hvq; simp [gdist_self]
    · have hvlt : h v < h q := by
        rcases lt_or_eq_of_le (hglob v) with hlt | heq
        · exact hlt
        · exfalso
          have hvmax : IsStrictLocalMax h v := by
            intro u hu
            have hle := hglob u
            have h1 := hh v u hu
            rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega
          exact hvq (huniq v hvmax)
      -- `v` is not the (unique) maximum, so it has a strictly higher neighbour.
      have hnotmax : ¬ ∀ u, adj v u → h u = h v - 1 :=
        fun hvmax => hvq (huniq v hvmax)
      push_neg at hnotmax
      obtain ⟨u, hadj, hune⟩ := hnotmax
      have h1 := hh v u hadj
      have hup : h u = h v + 1 := by
        rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2
        · omega
        · omega
      have hdefu : (h q - h u).toNat ≤ k := by
        have hglu := hglob u
        omega
      have hihu := ih u hdefu
      have htri := gdist_triangle q u v
      have huv : gdist u v = 1 := by rw [gdist_comm]; exact hadj
      have hstep : gdist q v ≤ (h q - h u) + 1 := by rw [huv] at htri; linarith
      linarith

/-- **Cone Lemma, maximum case** (Lemma 2.2, first statement).
A height function with a unique strict local maximum `q` is the distance cone at
`q`: `h v = h q - d(q,v)` for every cell `v`. -/
theorem cone_max {h : Cell m n → ℤ} (hh : IsHeight h) {q : Cell m n}
    (huniq : ∀ q', IsStrictLocalMax h q' → q' = q) :
    ∀ v, h v = h q - gdist q v := by
  have hglob := unique_max_global hh huniq
  intro v
  have hle : gdist q v ≤ h q - h v :=
    cone_ascend hh huniq hglob (h q - h v).toNat v le_rfl
  have hlip : |h q - h v| ≤ gdist q v :=
    height_lipschitz hh (gdist q v).toNat q v
      (le_of_eq (Int.toNat_of_nonneg (gdist_nonneg q v)).symm)
  have hnn : 0 ≤ h q - h v := by have := hglob v; omega
  rw [abs_of_nonneg hnn] at hlip
  linarith

/-- **Cone Lemma, minimum case** (Lemma 2.2, dual statement).
A height function with a unique strict local minimum `p` is the dual cone:
`h v = h p + d(p,v)`.  Proved by applying `cone_max` to `-h`. -/
theorem cone_min {h : Cell m n → ℤ} (hh : IsHeight h) {p : Cell m n}
    (huniq : ∀ p', IsStrictLocalMin h p' → p' = p) :
    ∀ v, h v = h p + gdist p v := by
  have hh' : IsHeight (fun v => -h v) := by
    intro a b hab
    show |(-h a) - (-h b)| = 1
    rw [show (-h a) - (-h b) = -(h a - h b) by ring, abs_neg]
    exact hh a b hab
  have huniq' : ∀ q', IsStrictLocalMax (fun v => -h v) q' → q' = p := by
    intro q' hq'
    apply huniq
    intro u hu
    have hval : -h u = -h q' - 1 := hq' u hu
    show h u = h q' + 1
    linarith
  have hcone := cone_max hh' huniq'
  intro v
  have hv : -h v = -h p - gdist p v := hcone v
  linarith

end OrigamiCone
