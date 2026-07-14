import OrigamiCone.SequelFrozen
import OrigamiCone.SequelCascade
import Mathlib.Tactic.LinearCombination

/-!
# Full Frozen Classification on a bounded grid (`lem:frozen`, both directions)

This module assembles the two directions of the paper's Frozen Classification
Lemma (`Lemma 8.3`, `lem:frozen`) into a single bidirectional statement,
BOUNDED to a grid of height `m`.  A triple `(u, v, w)` of proper 3-colouring
columns of the vertical path `P_m` has middle column `v` carrying no strict
local extremum at any row `i < m` iff the triple is **frozen on `[0, m)`**:
there is a nonzero slope `k ∈ ZMod 3` with `u i = v i - k` and `w i = v i + k`
for every row `i < m`.

The bound `i < m` matters: `SequelFrozen.IsFrozen` is unrestricted (`∀ i : ℕ`),
which is the "algebraic" form; the columns of a height function on `G_{m,n}`
are only defined for `i < m`, so downstream (`SequelActiveBound`,
`SequelContraction`, `SequelUniformOnsetProof`) needs the in-range form.  We
introduce the bounded predicate `IsFrozenOn u v w m` here rather than modify
the existing `IsFrozen`.

## Exports

* `IsFrozenOn u v w m` — bounded frozen predicate.
* `isExtremum_iff_offset_IsExtremum` — bridge from `SequelFrozen.isExtremum`
  to `SequelCascade.IsExtremum` under the offset substitution.
* `const_slope_of_rainbow_at` — helper: `SequelFrozen.rainbow_imp_const_slope`
  restated with the rainbow hypothesis at just two rows.
* `frozenOn_imp_extremumFree` — `⟸` direction.
* `extremumFree_imp_frozenOn` — `⟹` direction (real assembly).
* `frozenOn_iff_extremumFree` — bidirectional statement of `lem:frozen`.

`linear_combination` (rather than `linarith`) handles the ZMod 3 arithmetic:
`ZMod 3` is a commutative ring but not ordered, so `linarith` fails.
-/

namespace OrigamiCone.Sequel

/-- **Bounded frozen predicate**: `(u, v, w)` is frozen on `[0, m)` if there
is a nonzero slope `k ∈ ZMod 3` with `u i = v i - k` and `w i = v i + k` for
every row `i < m`. -/
def IsFrozenOn (u v w : ℕ → ZMod 3) (m : ℕ) : Prop :=
  ∃ k : ZMod 3, k ≠ 0 ∧
    (∀ i, i < m → u i = v i - k) ∧ (∀ i, i < m → w i = v i + k)

/-- **Bridge**: `SequelFrozen.isExtremum` (on `u, v, w`) coincides with
`SequelCascade.IsExtremum` (on the offset functions
`α i = u i - v i`, `γ i = w i - v i`, `s i = v (i+1) - v i`). -/
theorem isExtremum_iff_offset_IsExtremum
    (u v w : ℕ → ZMod 3) (m i : ℕ) :
    isExtremum u v w m i ↔
      IsExtremum (fun j => u j - v j) (fun j => w j - v j)
        (fun j => v (j + 1) - v j) m i := by
  unfold isExtremum IsExtremum
  refine ⟨fun ⟨h1, h2, h3⟩ => ⟨?_, ?_, ?_⟩, fun ⟨h1, h2, h3⟩ => ⟨?_, ?_, ?_⟩⟩
  · -- α i = γ i, i.e., u i - v i = w i - v i (from u i = w i)
    simp only; linear_combination h1
  · -- 0 < i → s (i - 1) = -α i, i.e., v i - v (i-1) = -(u i - v i)
    intro hpos
    simp only
    have hpre : (i - 1 + 1) = i := by omega
    rw [hpre]
    linear_combination -(h2 hpos)
  · -- i + 1 < m → s i = α i, i.e., v (i+1) - v i = u i - v i
    intro hbot
    simp only
    linear_combination h3 hbot
  · -- α i = γ i → u i = w i
    simp only at h1
    linear_combination h1
  · -- 0 < i → s (i - 1) = -α i → v (i - 1) = u i
    intro hpos
    have h := h2 hpos
    simp only at h
    have hpre : (i - 1 + 1) = i := by omega
    rw [hpre] at h
    linear_combination -h
  · -- i + 1 < m → s i = α i → v (i + 1) = u i
    intro hbot
    have h := h3 hbot
    simp only at h
    linear_combination h

/-- **`SequelFrozen.rainbow_imp_const_slope` restated with rainbow at only the
two involved rows**.  Same proof body — the original theorem's `hrain` is
used at exactly `i` and `i+1`.  This form sidesteps the `∀ j`-quantification,
which is problematic when the columns are only semantically defined for
`j < m`. -/
theorem const_slope_of_rainbow_at (u v w : ℕ → ZMod 3) (i : ℕ)
    (hsv : v (i + 1) ≠ v i) (hu : u (i + 1) ≠ u i) (hw : w (i + 1) ≠ w i)
    (hrain_i : w i - v i = -(u i - v i))
    (hrain_i1 : w (i + 1) - v (i + 1) = -(u (i + 1) - v (i + 1))) :
    u (i + 1) - v (i + 1) = u i - v i := by
  set sv := v (i + 1) - v i with hsvdef
  set a := u i - v i with hadef
  set x := u (i + 1) - v (i + 1) with hxdef
  have hs : sv ≠ 0 := sub_ne_zero.mpr hsv
  have h1 : x ≠ a - sv := by
    intro h; apply hu
    have hux : u (i + 1) = x + v (i + 1) := by rw [hxdef]; ring
    rw [hux, h, hsvdef, hadef]; ring
  have h2 : x ≠ a + sv := by
    intro h; apply hw
    have hwi1 : w (i + 1) = -(u (i + 1) - v (i + 1)) + v (i + 1) := by
      rw [← hrain_i1]; ring
    have hwi : w i = -(u i - v i) + v i := by rw [← hrain_i]; ring
    rw [hwi1, hwi, ← hxdef, ← hadef, h, hsvdef]; ring
  have key : ∀ p q r : ZMod 3, r ≠ 0 → p ≠ q - r → p ≠ q + r → p = q := by decide
  exact key x a sv hs h1 h2

/-- **`⟸` direction of `lem:frozen`**: a triple frozen on `[0, m)` carries no
strict local extremum at any row `i < m`.  Direct: from `u i = v i - k` and
`w i = v i + k` with `k ≠ 0`, the horizontal neighbours `u i, w i` are distinct
(as `k ≠ -k` in `ZMod 3` for `k ≠ 0`), so the first clause of the extremum
predicate fails. -/
theorem frozenOn_imp_extremumFree
    (u v w : ℕ → ZMod 3) (m : ℕ) (hf : IsFrozenOn u v w m) :
    ∀ i, i < m → ¬ isExtremum u v w m i := by
  obtain ⟨k, hk, hu, hw⟩ := hf
  intro i hi hext
  have heq : u i = w i := hext.1
  rw [hu i hi, hw i hi] at heq
  have hne : ∀ x y : ZMod 3, y ≠ 0 → x - y ≠ x + y := by decide
  exact hne (v i) k hk heq

/-- **`⟹` direction of `lem:frozen`**: given a triple `(u, v, w)` of proper
3-colouring columns of `P_m` (`m ≥ 1`) whose middle column `v` carries no
strict local extremum at any row `i < m`, the triple is frozen on `[0, m)`.

Proof:
* Translate to the offset functions `α = u − v`, `γ = w − v`, `s = v(·+1) − v`.
* Apply `SequelCascade.cascade` → rainbow at every row (`α i ≠ γ i` for
  `i < m`).
* Rainbow + nonzero offsets in ZMod 3 forces `γ i = −α i` at every row.
* Apply `const_slope_of_rainbow_at` at each step → `α (i+1) = α i` for
  `i + 1 < m`.
* Induct → `α i = α 0` for every `i < m`.
* Package with `k := −α 0`.
-/
theorem extremumFree_imp_frozenOn
    (u v w : ℕ → ZMod 3) (m : ℕ) (hm : 1 ≤ m)
    (hu_horz : ∀ i, i < m → u i ≠ v i)
    (hw_horz : ∀ i, i < m → w i ≠ v i)
    (hv_prop : ∀ i, i + 1 < m → v (i + 1) ≠ v i)
    (hu_prop : ∀ i, i + 1 < m → u (i + 1) ≠ u i)
    (hw_prop : ∀ i, i + 1 < m → w (i + 1) ≠ w i)
    (hNoExt : ∀ i, i < m → ¬ isExtremum u v w m i) :
    IsFrozenOn u v w m := by
  -- Offset functions.
  set α : ℕ → ZMod 3 := fun j => u j - v j with hα_def
  set γ : ℕ → ZMod 3 := fun j => w j - v j with hγ_def
  set s : ℕ → ZMod 3 := fun j => v (j + 1) - v j with hs_def
  -- Nonzero-offset hypotheses.
  have hα_ne : ∀ i, i < m → α i ≠ 0 := fun i hi =>
    sub_ne_zero.mpr (hu_horz i hi)
  have hγ_ne : ∀ i, i < m → γ i ≠ 0 := fun i hi =>
    sub_ne_zero.mpr (hw_horz i hi)
  have hs_ne : ∀ i, i + 1 < m → s i ≠ 0 := fun i hi =>
    sub_ne_zero.mpr (hv_prop i hi)
  -- Properness on the step: α (i+1) ≠ α i - s i, dually for γ.
  have hpu_off : ∀ i, i + 1 < m → α (i + 1) ≠ α i - s i := by
    intro i hi h
    apply hu_prop i hi
    -- α (i+1) = α i - s i is (u(i+1) - v(i+1)) = (u i - v i) - (v(i+1) - v i),
    -- which forces u(i+1) = u i by ring arithmetic.
    linear_combination h
  have hpw_off : ∀ i, i + 1 < m → γ (i + 1) ≠ γ i - s i := by
    intro i hi h
    apply hw_prop i hi
    linear_combination h
  -- Extremum-free ⟹ IsExtremum (offset form) fails at every row.
  have hNoExt_off : ∀ j, j < m → ¬ IsExtremum α γ s m j := by
    intro j hj hExt
    apply hNoExt j hj
    rw [isExtremum_iff_offset_IsExtremum]
    exact hExt
  -- Apply cascade → rainbow at every row.
  have hrainbow : ∀ i, i < m → α i ≠ γ i :=
    cascade α γ s m hα_ne hγ_ne hs_ne hpu_off hpw_off hNoExt_off
  -- Rainbow + nonzero forces γ i = -α i in ZMod 3.
  have hγ_eq_neg : ∀ i, i < m → γ i = -α i := by
    intro i hi
    have h1 := hα_ne i hi
    have h2 := hγ_ne i hi
    have h3 := hrainbow i hi
    have key : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ b → b = -a := by decide
    exact key (α i) (γ i) h1 h2 h3
  -- Rainbow form: w i - v i = -(u i - v i) for i < m.
  have hrain_form : ∀ i, i < m → w i - v i = -(u i - v i) := fun i hi =>
    hγ_eq_neg i hi
  -- Constant-slope step at each i with i + 1 < m.
  have hα_step : ∀ i, i + 1 < m → α (i + 1) = α i := by
    intro i hi
    have hi1 : i < m := by omega
    exact const_slope_of_rainbow_at u v w i (hv_prop i hi)
      (hu_prop i hi) (hw_prop i hi) (hrain_form i hi1) (hrain_form (i + 1) hi)
  -- Iterate: α i = α 0 for every i < m.
  have hα_const : ∀ i, i < m → α i = α 0 := by
    intro i
    induction i with
    | zero => intro _; rfl
    | succ k IH =>
      intro hk1
      have hk : k < m := by omega
      have hstep : α (k + 1) = α k := hα_step k hk1
      rw [hstep, IH hk]
  -- Package as IsFrozenOn with k := -α 0.
  refine ⟨-α 0, ?_, ?_, ?_⟩
  · exact neg_ne_zero.mpr (hα_ne 0 hm)
  · intro i hi
    -- Goal: u i = v i - (-α 0), i.e., u i = v i + α 0.
    have h_alpha : α i = α 0 := hα_const i hi
    -- α i = u i - v i, so u i - v i = α 0.
    linear_combination h_alpha
  · intro i hi
    -- Goal: w i = v i + (-α 0), i.e., w i = v i - α 0.
    have h_alpha : α i = α 0 := hα_const i hi
    have h_gamma : γ i = -α i := hγ_eq_neg i hi
    -- γ i = w i - v i, and γ i = -α i = -α 0.
    linear_combination h_gamma - h_alpha

/-- **Full Frozen Classification** (paper's `Lemma 8.3`, `lem:frozen`,
bidirectional).  A triple `(u, v, w)` of proper 3-colouring columns of `P_m`
(`m ≥ 1`) has middle column carrying no strict local extremum at any row
`i < m` iff the triple is frozen on `[0, m)`. -/
theorem frozenOn_iff_extremumFree
    (u v w : ℕ → ZMod 3) (m : ℕ) (hm : 1 ≤ m)
    (hu_horz : ∀ i, i < m → u i ≠ v i)
    (hw_horz : ∀ i, i < m → w i ≠ v i)
    (hv_prop : ∀ i, i + 1 < m → v (i + 1) ≠ v i)
    (hu_prop : ∀ i, i + 1 < m → u (i + 1) ≠ u i)
    (hw_prop : ∀ i, i + 1 < m → w (i + 1) ≠ w i) :
    (∀ i, i < m → ¬ isExtremum u v w m i) ↔ IsFrozenOn u v w m := by
  refine ⟨?_, frozenOn_imp_extremumFree u v w m⟩
  intro hNoExt
  exact extremumFree_imp_frozenOn u v w m hm hu_horz hw_horz
    hv_prop hu_prop hw_prop hNoExt

end OrigamiCone.Sequel
