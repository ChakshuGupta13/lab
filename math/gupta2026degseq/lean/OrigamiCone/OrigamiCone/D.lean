import OrigamiCone.DSquare
import OrigamiCone.D2n
import OrigamiCone.D3n
import OrigamiCone.D4n

/-!
# The diameter quantity `Dmn m n` and its closed forms (`thm:diam`)

The paper's **Theorem `thm:diam`** asserts a single quantity
`D(m, n) := \min_K \sum_{i,j} |(i + j) - K|` and four closed-form
specialisations:

| identity | formula | hypothesis |
|----------|---------|------------|
| `D(2, n)`  | `⌈n² / 2⌉ = (n² + 1) / 2`  | `(none, vacuous at n = 0)` |
| `D(3, n)`  | `⌊3 n² / 4⌋ + 2`            | `1 ≤ n` |
| `D(4, n)`  | `n² + 4 + [n odd]`          | `2 ≤ n` |
| `D(m, m)`  | `(m³ − m) / 3`              | `2 ≤ m` |
| symmetry   | `D(m, n) = D(n, m)`         | (all `m, n`)         |

The existing modules (`D2n`, `D3n`, `D4n`, `DSquare`) each prove
`IsMedianMin acell <closed-form value>` for their respective grid shape.
This module introduces a **named** function `Dmn : ℕ → ℕ → ℤ` (extracted
via `Classical.choose` of `medianMin_exists`) and packages the four closed
forms as equations `Dmn m n = <closed-form value>`, derived from the
existing `IsMedianMin` lemmas via the `medianMin_unique` extensionality
principle.

This unifies the paper's `thm:diam` into a single named diameter quantity,
closing the gap between the median-characterisation theorems and the
paper's `D(m, n)` notation.

Results:
* `Dmn` — the named diameter quantity `Dmn m n : ℤ`;
* `Dmn_isMedianMin` — characterising property: `IsMedianMin acell (Dmn m n)`;
* `Dmn_unique` — extensionality: `IsMedianMin acell D ⟹ Dmn m n = D`;
* `Dmn_symm` — `Dmn m n = Dmn n m` (paper claim, restated from
  `medianMin_swap`);
* `Dmn_2n`, `Dmn_3n`, `Dmn_4n`, `Dmn_mm` — the four paper closed forms,
  each as an equation `Dmn _ _ = <value>`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-! ## The named diameter quantity -/

/-- **The diameter quantity** `Dmn m n : ℤ` of the `m × n` grid:
`Dmn m n = min_K Σ_v |acell v - K|`, the minimised median-dispersion of
the antidiagonal `acell`.

Extracted via `Classical.choose` of `medianMin_exists`; uniqueness
(`medianMin_unique`) makes the choice immaterial, exposed via `Dmn_unique`. -/
noncomputable def Dmn (m n : ℕ) : ℤ :=
  Classical.choose (medianMin_exists (acell (m := m) (n := n)))

/-- `Dmn m n` is the minimised dispersion of `acell`. -/
theorem Dmn_isMedianMin (m n : ℕ) :
    IsMedianMin (acell (m := m) (n := n)) (Dmn m n) :=
  Classical.choose_spec (medianMin_exists (acell (m := m) (n := n)))

/-- **Extensionality** for `Dmn`: any `IsMedianMin acell D` value equals
`Dmn m n`.  This is the universal property of the diameter quantity. -/
theorem Dmn_unique {D : ℤ} (h : IsMedianMin (acell (m := m) (n := n)) D) :
    Dmn m n = D :=
  medianMin_unique (Dmn_isMedianMin m n) h

/-! ## Symmetry: `Dmn m n = Dmn n m` -/

/-- **Symmetry of the diameter quantity** `Dmn m n = Dmn n m`.
The grids `G_{m,n}` and `G_{n,m}` are isomorphic by swapping coordinates,
and the antidiagonal `acell v = v.1.val + v.2.val` is symmetric, so the
minimised dispersions agree.  Direct corollary of `medianMin_swap` in
`DiameterLower.lean`. -/
theorem Dmn_symm (m n : ℕ) : Dmn m n = Dmn n m :=
  Dmn_unique (medianMin_swap (Dmn_isMedianMin n m))

/-! ## Closed forms -/

/-- **`D(2, n)` closed form** (paper `thm:diam`):
`Dmn 2 n = ⌈n² / 2⌉ = (n² + 1) / 2`.

Unconditional; vacuously `Dmn 2 0 = 0` (the empty grid). -/
theorem Dmn_2n (n : ℕ) : Dmn 2 n = ((n * n + 1 : ℕ) : ℤ) / 2 :=
  Dmn_unique (D_2n n)

/-- **`D(3, n)` closed form** (paper `thm:diam`):
`Dmn 3 n = ⌊3 n² / 4⌋ + 2`, requires `n ≥ 1`. -/
theorem Dmn_3n (n : ℕ) (hn : 1 ≤ n) :
    Dmn 3 n = ((3 * n * n : ℕ) : ℤ) / 4 + 2 :=
  Dmn_unique (D_3n n hn)

/-- **`D(4, n)` closed form** (paper `thm:diam`):
`Dmn 4 n = n² + 4 + [n odd]` (Iverson bracket as `n % 2`),
requires `n ≥ 2`. -/
theorem Dmn_4n (n : ℕ) (hn : 2 ≤ n) :
    Dmn 4 n = ((n * n + 4 + n % 2 : ℕ) : ℤ) :=
  Dmn_unique (D_4n n hn)

/-- **`D(m, m)` closed form** (paper `thm:diam`):
`Dmn m m = (m³ − m) / 3`, requires `m ≥ 2`. -/
theorem Dmn_mm (m : ℕ) (hm : 2 ≤ m) :
    Dmn m m = (((m - 1) * m * (m + 1) / 3 : ℕ) : ℤ) :=
  Dmn_unique (D_mm hm)

end OrigamiCone
