import Mathlib.Data.Fin.Tuple.Sort
import OrigamiCone.Basic

/-!
# Rearrangement preserves `1`-Lipschitz on paths (fact (i) of `prop:monotone`)

Paper fact (i) of the **Monotone reduction** (Section 4,
`prop:monotone`): the increasing rearrangement of a `1`-Lipschitz integer
sequence on a path is itself `1`-Lipschitz; moreover its consecutive
differences lie in `{0, 1}`.

The proof is via a **discrete intermediate value theorem**: a
`1`-Lipschitz integer sequence takes every integer value between its
extremes.  If the sorted sequence had a jump `g (k+1) − g k ≥ 2`, the
integer `g k + 1` would be in the value-range of the original sequence
but absent from the sorted (= same-multiset) sequence — contradiction.

Results:
* `PathLipschitz1` — predicate "consecutive entries differ by at most one".
* `PathLipschitz1.ivt` — discrete IVT.
* `sort_pathLipschitz1` — sorting preserves `PathLipschitz1`.

No `sorry`.
-/

namespace OrigamiCone

variable {L : ℕ}

/-- A `Fin L → ℤ` sequence is **path-1-Lipschitz** if adjacent entries
differ by at most one.  Vacuous for `L = 0` and `L = 1`. -/
def PathLipschitz1 (x : Fin L → ℤ) : Prop :=
  ∀ i j : Fin L, i.val + 1 = j.val → |x i - x j| ≤ 1

/-- Negation preserves path-1-Lipschitz (used to reduce the IVT to the
directional case). -/
private lemma PathLipschitz1.neg {x : Fin L → ℤ} (hx : PathLipschitz1 x) :
    PathLipschitz1 (fun i => -x i) := by
  intro i j hij
  have h := hx i j hij
  show |-x i - -x j| ≤ 1
  rw [show -x i - -x j = -(x i - x j) by ring, abs_neg]
  exact h

/-- **Discrete IVT, directional version.**  Strong induction on
`i_hi.val − i_lo.val`: at each step take one path step from `i_lo`, then
either the value crosses `v` (use the current index) or it does not (use
the inductive hypothesis on the shorter remaining segment). -/
private lemma PathLipschitz1.ivt_dir {x : Fin L → ℤ} (hx : PathLipschitz1 x) :
    ∀ (k : ℕ) (i_lo i_hi : Fin L),
      i_hi.val = i_lo.val + k →
      ∀ (v : ℤ), x i_lo ≤ v → v ≤ x i_hi → ∃ i, x i = v := by
  intro k
  induction k with
  | zero =>
    intro i_lo i_hi h_eq v hlo hhi
    have h : i_hi = i_lo := Fin.ext (by omega)
    subst h
    -- `subst` removed `i_lo`, so `hlo : x i_hi ≤ v`, `hhi : v ≤ x i_hi`.
    exact ⟨i_hi, le_antisymm hlo hhi⟩
  | succ k ih =>
    intro i_lo i_hi h_eq v hlo hhi
    have h_lo_lt : i_lo.val + 1 < L := by
      have := i_hi.isLt
      omega
    let i₁ : Fin L := ⟨i_lo.val + 1, h_lo_lt⟩
    have h_adj : i_lo.val + 1 = i₁.val := rfl
    have habs : |x i_lo - x i₁| ≤ 1 := hx i_lo i₁ h_adj
    by_cases hcase : x i₁ ≤ v
    · have h_eq' : i_hi.val = i₁.val + k := by
        show i_hi.val = (i_lo.val + 1) + k
        omega
      exact ih i₁ i_hi h_eq' v hcase hhi
    · push_neg at hcase
      -- `x i₁ > v` and `|x i_lo - x i₁| ≤ 1` give `x i₁ ≤ x i_lo + 1`,
      -- so `v < x i_lo + 1`, i.e. `v ≤ x i_lo`; combined with `hlo`,
      -- `v = x i_lo`.
      rcases abs_le.mp habs with ⟨h_left, _⟩
      refine ⟨i_lo, ?_⟩
      linarith

/-- **Discrete IVT (general direction).**  A `1`-Lipschitz integer
sequence takes every integer value bracketed by two of its entries. -/
theorem PathLipschitz1.ivt {x : Fin L → ℤ} (hx : PathLipschitz1 x)
    {p q : Fin L} {v : ℤ} (h1 : x p ≤ v) (h2 : v ≤ x q) :
    ∃ i, x i = v := by
  rcases le_total p.val q.val with h | h
  · exact PathLipschitz1.ivt_dir hx (q.val - p.val) p q (by omega) v h1 h2
  · -- `q ≤ p` (in index) but `x p ≤ v ≤ x q` (in value): apply IVT_dir to
    -- `-x` with bounds `(q, p)` and value `-v`.
    have h_neg : PathLipschitz1 (fun i => -x i) := hx.neg
    have h_neg_dir :=
      PathLipschitz1.ivt_dir h_neg (p.val - q.val) q p (by omega) (-v)
        (by show -x q ≤ -v; linarith) (by show -v ≤ -x p; linarith)
    obtain ⟨i, hi⟩ := h_neg_dir
    refine ⟨i, ?_⟩
    linarith

/-- **Fact (i) of `prop:monotone`.**  The increasing rearrangement of a
path-`1`-Lipschitz integer sequence is itself path-`1`-Lipschitz.

Proof: the rearrangement `g = x ∘ Tuple.sort x` is monotone, so adjacent
differences are non-negative.  Suppose `g (k+1) − g k ≥ 2`.  Then the
value `v = g k + 1` is strictly between `g k` and `g (k+1)`.  By the
discrete IVT applied to the original sequence (bracketed by
`x (sort x k) = g k` and `x (sort x (k+1)) = g (k+1)`), some original
index hits `v`; pushed back through the sort permutation it gives a
position `l` of `g` with `g l = v`.  Monotonicity forces `k < l < k+1`,
impossible. -/
theorem sort_pathLipschitz1 {x : Fin L → ℤ} (hx : PathLipschitz1 x) :
    PathLipschitz1 (x ∘ Tuple.sort x) := by
  set g := x ∘ Tuple.sort x with hg_def
  intro i j hij
  have hmono : Monotone g := Tuple.monotone_sort x
  have hij_le : i ≤ j := by rw [Fin.le_def]; omega
  have hg_le : g i ≤ g j := hmono hij_le
  -- `|g i - g j| = g j - g i` (monotone), and we show `g j - g i ≤ 1`.
  rw [abs_sub_comm, abs_of_nonneg (by linarith : (0 : ℤ) ≤ g j - g i)]
  by_contra h_gt_1
  push_neg at h_gt_1
  -- `h_gt_1 : 1 < g j - g i`, i.e. `g i + 2 ≤ g j`.
  -- Take `v := g i + 1`.  Then `g i < v < g j`.
  set v := g i + 1 with hv_def
  -- IVT on the original sequence: `∃ k, x k = v`.
  have hlo : x (Tuple.sort x i) ≤ v := by
    show x (Tuple.sort x i) ≤ g i + 1
    have : x (Tuple.sort x i) = g i := rfl
    omega
  have hhi : v ≤ x (Tuple.sort x j) := by
    show g i + 1 ≤ x (Tuple.sort x j)
    have : x (Tuple.sort x j) = g j := rfl
    omega
  obtain ⟨k, hk⟩ := PathLipschitz1.ivt hx hlo hhi
  -- Push back through the sort permutation: `l := (sort x).symm k`
  -- satisfies `g l = v`.
  set l : Fin L := (Tuple.sort x).symm k with hl_def
  have h_g_l : g l = v := by
    show (x ∘ Tuple.sort x) l = v
    show x (Tuple.sort x l) = v
    have h_sort_l : Tuple.sort x l = k := by
      show Tuple.sort x ((Tuple.sort x).symm k) = k
      exact Equiv.apply_symm_apply _ _
    rw [h_sort_l]; exact hk
  -- Monotonicity squeezes `l` strictly between `i` and `j`, but
  -- `i.val + 1 = j.val` leaves no room.
  have h_l_gt_i : i < l := by
    rcases Nat.lt_or_ge i.val l.val with hlt | hge
    · exact (Fin.lt_def).mpr hlt
    · exfalso
      have h_l_le_i : l ≤ i := (Fin.le_def).mpr hge
      have : g l ≤ g i := hmono h_l_le_i
      omega
  have h_l_lt_j : l < j := by
    rcases Nat.lt_or_ge l.val j.val with hlt | hge
    · exact (Fin.lt_def).mpr hlt
    · exfalso
      have h_j_le_l : j ≤ l := (Fin.le_def).mpr hge
      have : g j ≤ g l := hmono h_j_le_l
      omega
  rw [Fin.lt_def] at h_l_gt_i h_l_lt_j
  omega

end OrigamiCone
