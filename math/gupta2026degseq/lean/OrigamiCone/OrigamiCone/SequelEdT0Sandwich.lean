import Mathlib
import OrigamiCone.SequelTransferInst
import OrigamiCone.SequelEdPolyFit

/-!
# Sequel: T₀ sandwich is eventually constant (Task E.γ.b concrete)

Concrete instantiation of the polynomial-fit pipeline at the sequel paper's
specific transfer matrix `T₀` from `SequelTransferInst`.  The paper's
`lem:quotient` (§8) concludes that on the trivial-character block, `T₀`
acts as multiplication by the frozen indicator — so on any ρ-invariant
function `v`, the iterate `T₀^n · v = χ_frozen · v` for every `n ≥ 1`.
This module packages that iteration into the "sandwich is eventually
constant" conclusion:

```
u ⬝ᵥ (T₀ m ℚ)^n *ᵥ v = ∑ s, u s * (if frozenPair s then v s else 0)
                                              (for all n ≥ 1)
```

for any ρ-invariant `u, v : Col m × Col m → ℚ`.  As a corollary, the
sandwich sequence agrees on `{n ≥ 1}` with the constant polynomial
`C (∑ s, u s * χv v s)` — degree 0, onset 1.

## Role in Task E

This is the concrete `[x^0]`-slice case of Task E.γ.b's charpoly-factoring
step: for the specific `T₀`, the "sandwich is diagonal `{0,1}`-idempotent"
structure gives a degree-0 polynomial witness immediately, without going
through the full charpoly-factorisation machinery.  The paper's
`lem:quotient` for `d = 0` reduces to `E_0(m, n) = 0` (constant), and this
module supplies that abstract shape at the sandwich level.

For `d ≥ 1`, the analogous conclusion requires the `[x^d]`-slice extension
of Task E.γ.b via the `RseqMat T A d` Leibniz chain (deferred; the current
Lean chain only handles `T + x·A` linear-in-x, but `transferMatrix m` has
arbitrary-degree x-entries).

## Theorems

* `χv (v : Col m × Col m → ℚ)` — frozen indicator applied to `v`:
  `fun s => if frozenPair s then v s else 0`.
* `χv_ρ_invariant`, `χv_idempotent` — structural properties of `χv`.
* `T0_mulVec_eq_χv` — `T₀ *ᵥ v = χv v` for ρ-invariant `v` (direct wrapper
  of `SequelTransferInst.T0_quotient_action`).
* `T0_mulVec_χv` — `T₀ *ᵥ (χv v) = χv v` (idempotency step).
* `T0_pow_mulVec` — `T₀^n *ᵥ v = χv v` for every `n ≥ 1` and ρ-invariant `v`.
  Proved by induction on `n`.
* **`T0_sandwich_eventually_constant`** — the sandwich
  `u ⬝ᵥ T₀^n *ᵥ v = ∑ s, u s * χv v s` for every `n ≥ 1` and ρ-invariant
  `u, v`.
* **`T0_sandwich_polynomial_witness`** — the sandwich sequence
  `n ↦ u ⬝ᵥ T₀^n *ᵥ v` agrees on `{n ≥ 1}` with the constant polynomial
  `C (∑ s, u s * χv v s) : Polynomial ℚ`.

## Discipline

Imports:
* `Mathlib` — background.
* `OrigamiCone.SequelTransferInst` — for `T₀`, `ρcol`, `frozenPair`,
  `T0_quotient_action`.
* `OrigamiCone.SequelEdPolyFit` — for the polynomial-witness shape
  (imported but not composed algebraically here; the constant-poly witness
  is direct).

Both imports are Sequel modules but do NOT collide (no `Env`, `d2`, etc.
overlap between `SequelTransferInst` and `SequelEdPolyFit`; the latter is
standalone Mathlib-only apart from its own definitions).

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]`.
Check with `#print axioms OrigamiCone.Sequel.T0_sandwich_polynomial_witness`.
-/

namespace OrigamiCone.Sequel

open Polynomial Finset Matrix

variable {m : ℕ}

/-- Frozen indicator applied to a function. -/
noncomputable def χv (v : Col m × Col m → ℚ) : Col m × Col m → ℚ :=
  fun s => if frozenPair s then v s else 0

/-- `χv v` is ρ-invariant when `v` is. -/
lemma χv_ρ_invariant (v : Col m × Col m → ℚ)
    (hv : ∀ s, v (ρcol s) = v s) :
    ∀ s, χv v (ρcol s) = χv v s := by
  intro s
  unfold χv
  have hfp := frozenPair_ρcol s
  rw [hv]
  by_cases h : frozenPair s
  · rw [if_pos h, if_pos (hfp.mpr h)]
  · rw [if_neg h, if_neg (fun hh => h (hfp.mp hh))]

/-- `χv` is idempotent: `χv (χv v) = χv v`. -/
lemma χv_idempotent (v : Col m × Col m → ℚ) : χv (χv v) = χv v := by
  funext s
  unfold χv
  by_cases h : frozenPair s
  · rw [if_pos h, if_pos h]
  · rw [if_neg h, if_neg h]

/-- `T₀ *ᵥ v = χv v` when `v` is ρ-invariant.  Direct wrapper of
`SequelTransferInst.T0_quotient_action`. -/
lemma T0_mulVec_eq_χv (v : Col m × Col m → ℚ)
    (hv : ∀ s, v (ρcol s) = v s) :
    (T0 m ℚ) *ᵥ v = χv v := by
  funext s
  have := T0_quotient_action v hv s
  unfold χv
  exact this

/-- `T₀ *ᵥ (χv v) = χv v` — idempotency at the T₀ level. -/
lemma T0_mulVec_χv (v : Col m × Col m → ℚ)
    (hv : ∀ s, v (ρcol s) = v s) : (T0 m ℚ) *ᵥ (χv v) = χv v := by
  have hχ : ∀ s, χv v (ρcol s) = χv v s := χv_ρ_invariant v hv
  rw [T0_mulVec_eq_χv (χv v) hχ, χv_idempotent]

/-- `T₀^n *ᵥ v = χv v` for every `n ≥ 1` and ρ-invariant `v`. -/
lemma T0_pow_mulVec (v : Col m × Col m → ℚ)
    (hv : ∀ s, v (ρcol s) = v s)
    (n : ℕ) (hn : 1 ≤ n) :
    (T0 m ℚ)^n *ᵥ v = χv v := by
  induction n with
  | zero => omega
  | succ n IH =>
    by_cases hn' : n = 0
    · subst hn'
      rw [pow_one]
      exact T0_mulVec_eq_χv v hv
    · have hn'' : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr hn'
      have IH' := IH hn''
      show ((T0 m ℚ) ^ (n + 1)) *ᵥ v = χv v
      rw [pow_succ', ← Matrix.mulVec_mulVec, IH', T0_mulVec_χv v hv]

/-- **T₀ sandwich is eventually constant.**  For any ρ-invariant boundary
vectors `u, v : Col m × Col m → ℚ`, and every `n ≥ 1`,
`u ⬝ᵥ (T₀ m ℚ)^n *ᵥ v` is CONSTANT (equal to `∑ s, u s * χv v s`, or
equivalently `∑ s frozen, u s * v s`).

This is the concrete `[x^0]`-slice conclusion of `lem:quotient` for the
sequel's transfer matrix.  As a corollary, the sandwich sequence admits a
degree-0 polynomial witness on `{n ≥ 1}`. -/
theorem T0_sandwich_eventually_constant (u v : Col m × Col m → ℚ)
    (hv : ∀ s, v (ρcol s) = v s)
    (n : ℕ) (hn : 1 ≤ n) :
    u ⬝ᵥ ((T0 m ℚ)^n *ᵥ v) = ∑ s, u s * χv v s := by
  rw [T0_pow_mulVec v hv n hn]
  rfl

/-- **Polynomial witness for the T₀ sandwich** (`d = 0` case of
`Ed_thm_poly_of_perAxis`'s `hrow` for the sequel's specific matrix).
The witness is the constant polynomial
`C (∑ s, u s * χv v s) : Polynomial ℚ`, of natDegree 0, agreeing with the
sandwich on `{n ≥ 1}`. -/
theorem T0_sandwich_polynomial_witness (u v : Col m × Col m → ℚ)
    (hv : ∀ s, v (ρcol s) = v s) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ 0 ∧
      ∀ n : ℕ, 1 ≤ n → u ⬝ᵥ ((T0 m ℚ)^n *ᵥ v) = p.eval (n : ℚ) := by
  refine ⟨Polynomial.C (∑ s, u s * χv v s), ?_, ?_⟩
  · exact Polynomial.natDegree_C _ |>.le
  · intro n hn
    rw [T0_sandwich_eventually_constant u v hv n hn, Polynomial.eval_C]

end OrigamiCone.Sequel
