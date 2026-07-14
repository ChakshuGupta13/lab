import OrigamiCone.D2nCLeq
import OrigamiCone.D2nSumIdentity

/-!
# `D(2, n) = ⌈n²/2⌉` (Section 4)

The next paper closed form (after `D(m, m) = (m³ − m)/3`):

  `D(2, n) := \min_K \sum_{v \in \mathrm{Cell}\,2\,n} |\mathrm{acell}\,v - K|
            = \lceil n^2 / 2 \rceil = (n^2 + 1) / 2 \quad (\text{in } \mathbb{Z})`,

unconditionally for every `n : ℕ` (the statement is vacuous at `n = 0`:
the grid is empty, the median dispersion is `0`, and `(0 + 1)/2 = 0`
in truncating integer division).

The proof assembles three previously formalised pieces:
1. `isMedianMin_sum_min` (`Median.lean`) — the median characterisation
   `disp(φ) = Σ_ℓ min(c_ℓ, N − c_ℓ)`.
2. `cLeq_acell_two` (`D2nCLeq.lean`) — the unified `cLeq = 2ℓ + 1` formula on
   the `2 × n` antidiagonal (combining triangle at `ℓ = 0` with middle band at
   `ℓ ∈ [1, n − 1]`).
3. `min_2n_sum` (`D2nSumIdentity.lean`) — the parity-cased sum identity
   `Σ min(2ℓ + 1, 2n − 2ℓ − 1) = (n² + 1)/2`.

The level range for `Cell 2 n` is `[0, 2 + n − 2)` = `[0, n)`.  Per-level
reduction via `cLeq_acell_two` converts the median sum into the parity-cased
form, which evaluates to `(n² + 1)/2` (equivalently `⌈n²/2⌉` for both
parities).

Results:
* `D_2n` — `IsMedianMin acell ((n² + 1)/2)` on `Cell 2 n` for `n ≥ 1`.

No `sorry`.
-/

namespace OrigamiCone

variable {n : ℕ}

/-- Range bound for `acell` on `Cell 2 n` (private helper).  For every `v`,
`0 ≤ acell v ≤ n`.  Vacuous when `Cell 2 n = ∅` (i.e. `n = 0`). -/
private lemma acell_range_2n (v : Cell 2 n) :
    (0 : ℤ) ≤ acell v ∧ acell v ≤ (n : ℤ) := by
  have h1 : v.1.val < 2 := v.1.isLt
  have h2 : v.2.val < n := v.2.isLt
  unfold acell
  refine ⟨by positivity, ?_⟩
  have h1' : (v.1.val : ℤ) ≤ 1 := by
    have : (v.1.val : ℤ) + 1 ≤ 2 := by exact_mod_cast h1
    linarith
  have h2' : (v.2.val : ℤ) ≤ (n : ℤ) - 1 := by
    have : (v.2.val : ℤ) + 1 ≤ (n : ℤ) := by exact_mod_cast h2
    linarith
  linarith

/-- **Median sum on the `2 × n` antidiagonal**, evaluated.  The
`Σ_ℓ min(c_ℓ, 2n − c_ℓ)` evaluation over `[0, n)` equals `(n² + 1)/2`
(in `ℤ`).  Vacuously true when `n = 0` (empty Ico).  The arithmetic core
of `D_2n`. -/
private lemma medianSum_acell_2n :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) (n : ℤ),
        min (cLeq (acell (m := 2) (n := n)) ℓ)
            (((2 * n : ℕ) : ℤ) - cLeq (acell (m := 2) (n := n)) ℓ)
      = ((n * n + 1 : ℕ) : ℤ) / 2 := by
  -- Step 1: per-level reduction via cLeq_acell_two.
  rw [show ∑ ℓ ∈ Finset.Ico (0 : ℤ) (n : ℤ),
          min (cLeq (acell (m := 2) (n := n)) ℓ)
              (((2 * n : ℕ) : ℤ) - cLeq (acell (m := 2) (n := n)) ℓ)
        = ∑ ℓ ∈ Finset.Ico (0 : ℤ) (n : ℤ),
            min (2 * ℓ + 1) (2 * (n : ℤ) - 2 * ℓ - 1) from ?_]
  · -- Step 2: apply the parity-cased sum identity.
    exact min_2n_sum n
  -- Reindex the sum: each ℓ ∈ Ico 0 n is a nonneg integer < n; cast via toNat.
  rw [sum_Ico_int_eq_sum_range (n : ℤ) (Int.natCast_nonneg _)]
  rw [Int.toNat_natCast]
  rw [sum_Ico_int_eq_sum_range (n : ℤ) (Int.natCast_nonneg _)]
  rw [Int.toNat_natCast]
  apply Finset.sum_congr rfl
  intro ℓ hℓ
  rw [Finset.mem_range] at hℓ
  rw [cLeq_acell_two ℓ hℓ]
  -- Goal: min (2·ℓ + 1) (2n − (2·ℓ + 1)) = min (2·ℓ + 1) (2n − 2·ℓ − 1).
  congr 1
  push_cast; ring

/-! ## The main theorem -/

/-- **`D(2, n) = ⌈n²/2⌉`** (Section 4 paper closed form).

The minimised dispersion of the antidiagonal `acell : Cell 2 n → ℤ` on the
`2 × n` grid equals `(n² + 1)/2`, the integer reformulation of `⌈n²/2⌉`
(since for both parities of `n`, `⌈n²/2⌉ = (n² + 1)/2` in truncating
integer division).  Holds unconditionally for all `n : ℕ`; at `n = 0`
both sides are `0` (the grid is empty so the median dispersion is
trivially `0`).

Proof: combine `isMedianMin_sum_min` (the median characterisation,
`Median.lean`) with `medianSum_acell_2n` (the closed-form evaluation, this
file).  The latter routes through `cLeq_acell_two` (`D2nCLeq.lean`, the
unified `cLeq = 2ℓ + 1` formula) and `min_2n_sum` (`D2nSumIdentity.lean`,
the parity-cased sum identity). -/
theorem D_2n (n : ℕ) :
    IsMedianMin (acell (m := 2) (n := n)) (((n * n + 1 : ℕ) : ℤ) / 2) := by
  have hLU : (0 : ℤ) ≤ (n : ℤ) := Int.natCast_nonneg _
  have hφ : ∀ v : Cell 2 n, (0 : ℤ) ≤ acell v ∧ acell v ≤ (n : ℤ) :=
    acell_range_2n
  have hmid := isMedianMin_sum_min
    (acell (m := 2) (n := n)) 0 (n : ℤ) hLU hφ
  have hcard : (Fintype.card (Cell 2 n) : ℤ) = ((2 * n : ℕ) : ℤ) := by
    simp [Cell, Fintype.card_prod, Fintype.card_fin]
  rw [hcard] at hmid
  rw [medianSum_acell_2n] at hmid
  exact hmid

end OrigamiCone
