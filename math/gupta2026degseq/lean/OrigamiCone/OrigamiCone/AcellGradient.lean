import OrigamiCone.AcellHeight
import OrigamiCone.Degree2

/-!
# `acell` is the corner gradient `coneC (top, top) (m + n − 2)`

The antidiagonal `acell v = v.1.val + v.2.val` is exactly the **cone at the
top-right corner** `(m − 1, n − 1)` with peak value `m + n − 2`:

  `acell = coneC ⟨m − 1, n − 1⟩ (m + n − 2)`        (for `m, n ≥ 1`).

This identification realizes `acell` as the paper's canonical degree-2
height function `h₊₊(i, j) = (i − 1) + (j − 1)` (paper's 1-based indexing;
identical up to translation under our 0-based indexing).

Consequences delivered here:
* `acell` has a **unique strict local maximum** at `(m − 1, n − 1)`
  (immediate from `coneC_unique_max`).

Why this matters: the paper's `thm:diam` lower bound exhibits the two
opposite corner gradients `h₊₊` and `h₋₋ = −h₊₊` as witnesses whose OFG
distance equals `D(m, n)`.  This module identifies `h₊₊` with `acell` in
the cone-based machinery (`Degree2.lean`), bridging the antidiagonal
appearing in the median-dispersion side (`Diameter.lean`,
`DiameterLower.lean`) with the cone classification on the degree side.

Results:
* `acell_eq_coneC_top` — the formula `acell = coneC ⟨m − 1, n − 1⟩ (m + n − 2)`.
* `acell_unique_max` — `(m − 1, n − 1)` is the unique strict local maximum.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-! ## The corner-gradient identity -/

/-- **The antidiagonal is the cone at the top-right corner.**

For `m, n ≥ 1`, `acell` equals `coneC ⟨m − 1, n − 1⟩ (m + n − 2)`:

  `(i + j : ℤ) = (m + n − 2) − ((m − 1 − i) + (n − 1 − j))`.

This identification places `acell` inside the cone-classification machinery
of `Degree2.lean`, allowing the corner-gradient properties (unique extrema,
degree 2) to apply to it directly. -/
theorem acell_eq_coneC_top (hm : 1 ≤ m) (hn : 1 ≤ n) :
    (acell (m := m) (n := n))
      = coneC (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩) ((m + n - 2 : ℕ) : ℤ) := by
  funext v
  unfold acell coneC gdist
  -- After unfold:
  --   Goal: (v.1.val : ℤ) + v.2.val = ↑(m+n-2) - ↑((d1.natAbs + d2.natAbs : ℕ))
  --   where d1 = (⟨m-1, _⟩.val : ℤ) - v.1.val = (m-1 : ℕ) - v.1.val   (in ℤ)
  --         d2 = (⟨n-1, _⟩.val : ℤ) - v.2.val = (n-1 : ℕ) - v.2.val   (in ℤ)
  -- Since v.1.val ≤ m-1 and v.2.val ≤ n-1, both differences are nonneg,
  -- so each .natAbs equals the difference itself.
  have h1 : v.1.val < m := v.1.isLt
  have h2 : v.2.val < n := v.2.isLt
  have hmn : 2 ≤ m + n := by omega
  -- Normalize the Nat-cast subtractions globally.
  push_cast [Nat.cast_sub hm, Nat.cast_sub hn, Nat.cast_sub hmn]
  -- Now the goal has shape (using ℤ-abs `|·|` from natAbs):
  --   ↑v.1.val + ↑v.2.val = (↑m + ↑n - 2) - (|↑m - 1 - ↑v.1.val| + |↑n - 1 - ↑v.2.val|)
  -- with v.1.val ≤ m-1 and v.2.val ≤ n-1 (both as ℤ via the cast bounds).
  have hv1 : (v.1.val : ℤ) ≤ (m : ℤ) - 1 := by
    have : (v.1.val : ℤ) + 1 ≤ (m : ℤ) := by exact_mod_cast h1
    linarith
  have hv2 : (v.2.val : ℤ) ≤ (n : ℤ) - 1 := by
    have : (v.2.val : ℤ) + 1 ≤ (n : ℤ) := by exact_mod_cast h2
    linarith
  -- Resolve the two abs's by sign (each argument is nonneg).
  rw [abs_of_nonneg (by linarith : (0 : ℤ) ≤ (m : ℤ) - 1 - v.1.val),
      abs_of_nonneg (by linarith : (0 : ℤ) ≤ (n : ℤ) - 1 - v.2.val)]
  ring

/-! ## Unique-maximum corollary -/

/-- **`acell` has unique strict local max at the top-right corner.**

For `m, n ≥ 1`, the cell `(m − 1, n − 1)` is the only strict local maximum
of `acell`.  Immediate from `coneC_unique_max` via `acell_eq_coneC_top`. -/
theorem acell_unique_max (hm : 1 ≤ m) (hn : 1 ≤ n) :
    ∀ q', IsStrictLocalMax (acell (m := m) (n := n)) q'
      → q' = (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩) := by
  rw [acell_eq_coneC_top hm hn]
  exact coneC_unique_max _ _

end OrigamiCone
