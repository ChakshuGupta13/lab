import Mathlib

/-!
# Sequel meta-theorem: the cascade of `lem:frozen`

Standalone formalisation of the **cascade** half of the Frozen Classification of
the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

The `⟹` direction of `lem:frozen` says: if the middle column `v` of a triple
`(u, v, w)` of three consecutive height-function columns carries no strict local
extremum, then there is a slope `k ∈ {1, 2}` with `u = v - k` and `w = v + k`
everywhere (frozen). The paper proves this in two steps:

> **Cascade.** Extremum-free ⟹ every row is **rainbow** (`α_i ≠ γ_i`, with
> `α_i = u_i - v_i` and `γ_i = w_i - v_i`), via a maximal-aligned-block argument.
>
> **Constant slope.** Rainbow ⟹ the offset `α_i = u_i - v_i` is constant, hence
> `α ≡ -k` and `γ ≡ k = 3 - α` realise the frozen form. This is the algebraic
> engine already formalised in `SequelFrozen.rainbow_imp_const_slope`.

This module proves the **cascade** — the harder, finite-induction half. Working
in `ℤ/3` with abstract offset functions `α, γ, s : ℕ → ZMod 3` (where
`α i = u i - v i`, `γ i = w i - v i`, `s i = v(i+1) - v i`) and the paper's
extremum predicate (aligned rows with the right slope boundaries):

* `cascade_step` : the inductive step — aligned at `j` with value `r`,
  `s(j-1) = -r` (or `j = 0`), and non-extremum at `j` together force
  `s j = -r` AND aligned at `j+1` with the same value `r`. This is what the paper
  calls "the down-rule plus non-extremum propagation";
* `cascade_extend` : finite induction — given an aligned row `p` with value `r`
  and the base condition (`s(p-1) = -r` when `p > 0`), every row from `p` to
  `m-1` is aligned with value `r`, and every slope from `s(p)` through
  `s(m-2)` equals `-r`;
* `up_rule_base` : the up-rule contrapositive supplying the base case when
  `p > 0` — if row `p-1` is not aligned and row `p` is aligned with value `r`,
  then `s(p-1) = -r`;
* `cascade` (the `⟹` cascade of `lem:frozen`, **complete**): given the column
  hypotheses (nonzero offsets, properness, no extremum at any row), every row is
  rainbow (`α i ≠ γ i` for `0 ≤ i < m`).

The conclusion `cascade` matches in shape the rainbow hypothesis of
`SequelFrozen.rainbow_imp_const_slope` (which converts the rainbow column to
constant slope, completing the `⟹` of `lem:frozen`). Both modules use compatible
abstract column types `ℕ → ZMod 3`; the bridge from `α i ≠ γ i` (the cascade's
conclusion) to `γ i = -α i` (the form `rainbow_imp_const_slope` consumes) is a
one-line `ZMod 3` decide on the nonzero offsets, not literally definitional.

The hypotheses of the cascade are stated abstractly on `α, γ, s` (rather than on
`u, v, w`) so the proof avoids the `u`/`v`/`w` coupling and uses only the offset
algebra. The translation from `u, v, w` is straightforward (`α := u - v`,
`γ := w - v`, `s := v(·+1) - v`); properness of `u` and `w` (as proper
`3`-colourings of the path) reads as `α(i+1) ≠ α i - s i` and
`γ(i+1) ≠ γ i - s i`, and `α i, γ i, s i ≠ 0` records the proper-`3`-colouring
condition row by row.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.cascade`.
-/

namespace OrigamiCone.Sequel

/-- In `ℤ/3`, every element satisfies `x + x = -x`. -/
theorem zmod3_double (x : ZMod 3) : x + x = -x := by revert x; decide

/-- In `ℤ/3`, every element satisfies `-x - x = x`. -/
theorem zmod3_neg_sub (x : ZMod 3) : -x - x = x := by revert x; decide

/-- **Extremum predicate**, expressed on the offset functions `α, γ, s`. Row `j`
of the middle column carries a strict local extremum iff (1) it is **aligned**
(`α j = γ j`), (2) the up-neighbour parity guard `s(j-1) = -α j` holds when `j` is
not the top row, and (3) the down-neighbour parity guard `s j = α j` holds when
`j` is not the bottom row. This matches `SequelFrozen.isExtremum` under the
substitution `α = u - v`, `γ = w - v`, `s = v(·+1) - v`. -/
def IsExtremum (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m j : ℕ) : Prop :=
  α j = γ j ∧ (0 < j → s (j-1) = -α j) ∧ (j + 1 < m → s j = α j)

/-- **Aligned-with-value-`r`** predicate: row `j` is aligned with value `r`, and
when `j > 0` the entering slope `s(j-1)` equals `-r`. This is the loop invariant
maintained by `cascade_extend`. -/
def AtR (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (r : ZMod 3) (j : ℕ) : Prop :=
  α j = r ∧ γ j = r ∧ (0 < j → s (j-1) = -r)

/-- **Inductive step of the cascade**. From an aligned row `j` with value `r` and
the entering-slope condition `s(j-1) = -r` (or `j = 0`), under non-extremum at `j`
and properness on the step `j → j+1`, the leaving slope `s j` equals `-r` and
row `j+1` is aligned with the same value `r`. -/
theorem cascade_step
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m : ℕ)
    (hα : ∀ i, i < m → α i ≠ 0)
    (hγ : ∀ i, i < m → γ i ≠ 0)
    (hs : ∀ i, i + 1 < m → s i ≠ 0)
    (hpu : ∀ i, i + 1 < m → α (i+1) ≠ α i - s i)
    (hpw : ∀ i, i + 1 < m → γ (i+1) ≠ γ i - s i)
    (hNoExt : ∀ j, j < m → ¬ IsExtremum α γ s m j)
    (r : ZMod 3) (hr : r ≠ 0)
    (j : ℕ) (hj : j + 1 < m)
    (hAt : AtR α γ s r j) :
    AtR α γ s r (j + 1) := by
  obtain ⟨hαj, hγj, hsPrev⟩ := hAt
  have hjLt : j < m := by omega
  have hne := hNoExt j hjLt
  -- s j ≠ r (otherwise IsExtremum at j would hold).
  have hsj_ne_r : s j ≠ r := by
    intro h
    apply hne
    refine ⟨hαj.trans hγj.symm, ?_, ?_⟩
    · intro hjp; rw [hαj]; exact hsPrev hjp
    · intro _; rw [hαj]; exact h
  have hsj_ne_0 : s j ≠ 0 := hs _ hj
  have key₁ : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ b → a = -b := by decide
  have hsj : s j = -r := key₁ _ _ hsj_ne_0 hr hsj_ne_r
  -- α(j+1) = r and γ(j+1) = r via the down-rule.
  have hα1_ne_0 : α (j+1) ≠ 0 := hα _ hj
  have hγ1_ne_0 : γ (j+1) ≠ 0 := hγ _ hj
  have hα1_ne : α (j+1) ≠ -r := by
    intro h; apply hpu _ hj
    rw [h, hsj, hαj, sub_neg_eq_add, zmod3_double]
  have hγ1_ne : γ (j+1) ≠ -r := by
    intro h; apply hpw _ hj
    rw [h, hsj, hγj, sub_neg_eq_add, zmod3_double]
  have key₂ : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ -b → a = b := by decide
  refine ⟨key₂ _ _ hα1_ne_0 hr hα1_ne, key₂ _ _ hγ1_ne_0 hr hγ1_ne, ?_⟩
  intro _
  show s (j + 1 - 1) = -r
  have hsub : j + 1 - 1 = j := by omega
  rw [hsub]; exact hsj

/-- **Cascade extension**. Given a base alignment at row `p` with value `r`, the
cascade extends to every row from `p` to `m-1` (all aligned with `r`, with the
entering slope condition maintained). -/
theorem cascade_extend
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m : ℕ)
    (hα : ∀ i, i < m → α i ≠ 0)
    (hγ : ∀ i, i < m → γ i ≠ 0)
    (hs : ∀ i, i + 1 < m → s i ≠ 0)
    (hpu : ∀ i, i + 1 < m → α (i+1) ≠ α i - s i)
    (hpw : ∀ i, i + 1 < m → γ (i+1) ≠ γ i - s i)
    (hNoExt : ∀ j, j < m → ¬ IsExtremum α γ s m j)
    (r : ZMod 3) (hr : r ≠ 0)
    (p : ℕ) (hAtP : AtR α γ s r p) :
    ∀ k, p + k < m → AtR α γ s r (p + k) := by
  intro k hk
  induction k with
  | zero => simpa using hAtP
  | succ k IH =>
    have hk' : p + k + 1 < m := hk
    have hpk : p + k < m := by omega
    have ih := IH (by omega)
    exact cascade_step α γ s m hα hγ hs hpu hpw hNoExt r hr (p + k) hk' ih

/-- **Up-rule base** (the contrapositive supplying `s(p-1) = -r` from the
minimality of `p`). If row `p-1` is **not** aligned but row `p` is aligned with
value `r`, then the entering slope `s(p-1)` equals `-r`. -/
theorem up_rule_base
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (p : ℕ)
    (hαp1 : α (p-1) ≠ 0) (hγp1 : γ (p-1) ≠ 0)
    (hsp1_ne_0 : s (p-1) ≠ 0)
    (hpu : α p ≠ α (p-1) - s (p-1))
    (hpw : γ p ≠ γ (p-1) - s (p-1))
    (r : ZMod 3) (hr : r ≠ 0)
    (hAlP : α p = r ∧ γ p = r)
    (hNotAlPrev : α (p-1) ≠ γ (p-1)) :
    s (p-1) = -r := by
  -- Suppose s(p-1) = r. The up-rule would force α(p-1) = γ(p-1) = r, contradiction.
  have hsp1_ne_r : s (p-1) ≠ r := by
    intro h
    have hα_ne_neg : α (p-1) ≠ -r := by
      intro hα; apply hpu; rw [hα, h, hAlP.1, zmod3_neg_sub]
    have hγ_ne_neg : γ (p-1) ≠ -r := by
      intro hγ; apply hpw; rw [hγ, h, hAlP.2, zmod3_neg_sub]
    have key : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ -b → a = b := by decide
    exact hNotAlPrev ((key _ _ hαp1 hr hα_ne_neg).trans (key _ _ hγp1 hr hγ_ne_neg).symm)
  have key₁ : ∀ a b : ZMod 3, a ≠ 0 → b ≠ 0 → a ≠ b → a = -b := by decide
  exact key₁ _ _ hsp1_ne_0 hr hsp1_ne_r

/-- Local helper: from the cascade reaching row `m-1` with value `r`, derive
`IsExtremum` at row `m-1`, contradicting `hNoExt`. Requires `m ≥ 1`. -/
private theorem bottom_contradiction
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m : ℕ) (hm : 1 ≤ m)
    (r : ZMod 3)
    (hNoExt : ∀ j, j < m → ¬ IsExtremum α γ s m j)
    (hAt : AtR α γ s r (m - 1)) : False := by
  apply hNoExt (m - 1) (by omega)
  obtain ⟨hα, hγ, hsPrev⟩ := hAt
  refine ⟨hα.trans hγ.symm, ?_, ?_⟩
  · intro hpos
    have hsub : m - 1 - 1 = m - 2 := by omega
    rw [hα]; rw [show (m - 1 : ℕ) - 1 = m - 1 - 1 from rfl, hsub]
    have := hsPrev hpos
    rw [hsub] at this
    exact this
  · intro h; omega

/-- **Cascade** (the `⟹` direction of `lem:frozen`, complete). Under the column
hypotheses (nonzero offsets and slopes, properness of `u` and `w`, no extremum at
any row of the middle column), every row is **rainbow**: `α i ≠ γ i` for
`0 ≤ i < m`. Composed with `SequelFrozen.rainbow_imp_const_slope`, this completes
the `⟹` direction of the Frozen Classification. -/
theorem cascade
    (α γ : ℕ → ZMod 3) (s : ℕ → ZMod 3) (m : ℕ)
    (hα : ∀ i, i < m → α i ≠ 0)
    (hγ : ∀ i, i < m → γ i ≠ 0)
    (hs : ∀ i, i + 1 < m → s i ≠ 0)
    (hpu : ∀ i, i + 1 < m → α (i+1) ≠ α i - s i)
    (hpw : ∀ i, i + 1 < m → γ (i+1) ≠ γ i - s i)
    (hNoExt : ∀ j, j < m → ¬ IsExtremum α γ s m j) :
    ∀ i, i < m → α i ≠ γ i := by
  intro i hi hAl
  classical
  -- Take the minimal aligned row using Nat.find.
  let P : ℕ → Prop := fun j => j < m ∧ α j = γ j
  have hexists : ∃ j, P j := ⟨i, hi, hAl⟩
  set p := Nat.find hexists with hp_def
  obtain ⟨hp_lt, hp_al⟩ : P p := Nat.find_spec hexists
  have hp_min : ∀ q, q < p → ¬ P q := fun q hq => Nat.find_min hexists hq
  -- Establish AtR α γ s α(p) p.
  set r := α p with hr_def
  have hr_ne : r ≠ 0 := hα p hp_lt
  have hAlP : α p = r ∧ γ p = r := ⟨rfl, hp_al.symm⟩
  have hAtP : AtR α γ s r p := by
    refine ⟨rfl, hp_al.symm, ?_⟩
    intro hpos
    -- p > 0: row p-1 is not aligned (by minimality).
    have hNotAlPrev : α (p - 1) ≠ γ (p - 1) := by
      intro hEq; apply hp_min (p - 1) (by omega); exact ⟨by omega, hEq⟩
    have hp1_lt : p - 1 < m := by omega
    have hp_step : p - 1 + 1 < m := by omega
    have hp_eq : p - 1 + 1 = p := by omega
    have hpu_p : α p ≠ α (p - 1) - s (p - 1) := by
      have := hpu (p - 1) hp_step
      rw [hp_eq] at this; exact this
    have hpw_p : γ p ≠ γ (p - 1) - s (p - 1) := by
      have := hpw (p - 1) hp_step
      rw [hp_eq] at this; exact this
    exact up_rule_base α γ s p (hα _ hp1_lt) (hγ _ hp1_lt) (hs _ hp_step)
      hpu_p hpw_p r hr_ne hAlP hNotAlPrev
  -- Extend the cascade from p to m-1.
  have hk : p + (m - 1 - p) < m := by omega
  have hAtBottom : AtR α γ s r (p + (m - 1 - p)) :=
    cascade_extend α γ s m hα hγ hs hpu hpw hNoExt r hr_ne p hAtP _ hk
  have hsum : p + (m - 1 - p) = m - 1 := by omega
  rw [hsum] at hAtBottom
  exact bottom_contradiction α γ s m (by omega) r hNoExt hAtBottom

end OrigamiCone.Sequel
