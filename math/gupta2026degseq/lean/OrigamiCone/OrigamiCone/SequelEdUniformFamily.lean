import OrigamiCone.SequelEd
import OrigamiCone.SequelEdUniformAssembly

/-!
# Sequel: family-parametrized assembly + Ed bridge (Task E.δ.h Ed interface)

`SequelEdUniformAssembly` supplies the end-to-end arithmetic wrapper
`uniform_polynomial_from_partition` for a **fixed** target type `X`.  The
paper's `lem:uniform` operates on the height-function set
`{h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d}`, whose type
`Cell m n → ℤ` **depends on `n`**.  This module supplies:

1. `hdecomp_from_partition_family`, `uniform_polynomial_from_partition_family`
   — dependently-typed versions where `X : ℕ → Type*` varies with `n`;
2. `Ed_finset` — a `Finset (Cell m n → ℤ)` witnessing the height-fn fiber at
   column count `n`, via `SequelEd.Ed_fibre_finite`;
3. **`Ed_polynomial_of_partition`** — the concrete bridge: given a
   substrate-heavy partition of `Ed_finset` into type-fibers with each fiber's
   card = `runExtension_card` output, `n ↦ ((Ed d m n : ℕ) : ℚ)` agrees on
   `{n ≥ N}` with a polynomial of natDegree ≤ `D`.  This is exactly the shape
   `SequelEd.Ed_thm_poly_of_perAxis` consumes as `hrow`.

## Consumer interface for `lem:uniform`

To prove `lem:uniform` on `Ed d m ·` a consumer needs only:

1. A finite `types : Finset ι` (a `Fintype` `ι` works trivially).
2. `fiber : ι → (n : ℕ) → Finset (Cell m n → ℤ)` — the type-`t` fiber at
   column count `n`.
3. Arithmetic bounds `hr, hrW, hbound, hWN` on `r t, W t`.
4. Per-fiber card `hfiber` matching `runExtension_card`.
5. Partition `hpart` and disjointness `hdisj` of the fibers.

Then `Ed_polynomial_of_partition` produces the polynomial witness.  Steps 1–5
are the substrate-heavy contraction map, still deferred.

## Substrate

Imports `SequelEd` (for `Ed`, `Ed_fibre_finite`, `Ed_eq_finset_card`) and
`SequelEdUniformAssembly` (for `degreeBound_assembly`).

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

/-! ## Family-parametrized assembly

Generalises `hdecomp_from_partition` and `uniform_polynomial_from_partition`
to a target type family `X : ℕ → Type*` (the height-function set changes type
with the column count `n`).  The proofs are identical to the fixed-type
versions; only the signatures differ.
-/

/-- Family version of `hdecomp_from_partition`.  The Finsets `S n`, `fiber t n`
live in `Finset (X n)` for a family `X : ℕ → Type*`.  Proof is identical to
the fixed-type version. -/
theorem hdecomp_from_partition_family
    {ι : Type*} {X : ℕ → Type*} [∀ n, DecidableEq (X n)]
    (types : Finset ι) (S : (n : ℕ) → Finset (X n))
    (fiber : ι → (n : ℕ) → Finset (X n))
    (r W : ι → ℕ) (N : ℕ)
    (hWN : ∀ t ∈ types, W t ≤ N)
    (hfiber : ∀ t ∈ types, ∀ n, W t ≤ n →
        (fiber t n).card = (n - (W t - r t + 1)).choose (r t - 1))
    (hpart : ∀ n, N ≤ n → S n = types.biUnion (fun t => fiber t n))
    (hdisj : ∀ n, N ≤ n → (↑types : Set ι).PairwiseDisjoint (fun t => fiber t n)) :
    ∀ n, N ≤ n → ((S n).card : ℚ)
      = ∑ t ∈ types, ((n - (W t - r t + 1)).choose (r t - 1) : ℚ) := by
  intro n hn
  rw [hpart n hn, Finset.card_biUnion (hdisj n hn)]
  push_cast
  apply Finset.sum_congr rfl
  intro t ht
  have := hfiber t ht n (le_trans (hWN t ht) hn)
  exact_mod_cast this

/-- Family version of `uniform_polynomial_from_partition`.  Given a partition
of `S n : Finset (X n)` (with `X : ℕ → Type*` dependent) into type-fibers with
each fiber's card matching `runExtension_card`, produces the polynomial witness
for `((S n).card : ℚ)`. -/
theorem uniform_polynomial_from_partition_family
    {ι : Type*} {X : ℕ → Type*} [∀ n, DecidableEq (X n)]
    (types : Finset ι) (S : (n : ℕ) → Finset (X n))
    (fiber : ι → (n : ℕ) → Finset (X n))
    (r W : ι → ℕ) (D N : ℕ)
    (hr : ∀ t ∈ types, 1 ≤ r t)
    (hrW : ∀ t ∈ types, r t ≤ W t)
    (hbound : ∀ t ∈ types, r t - 1 ≤ D)
    (hWN : ∀ t ∈ types, W t ≤ N)
    (hfiber : ∀ t ∈ types, ∀ n, W t ≤ n →
        (fiber t n).card = (n - (W t - r t + 1)).choose (r t - 1))
    (hpart : ∀ n, N ≤ n → S n = types.biUnion (fun t => fiber t n))
    (hdisj : ∀ n, N ≤ n → (↑types : Set ι).PairwiseDisjoint (fun t => fiber t n)) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
      ∀ n : ℕ, N ≤ n → ((S n).card : ℚ) = p.eval (n : ℚ) := by
  refine degreeBound_assembly types (fun _ => (1 : ℚ))
    (fun t => W t - r t + 1) (fun t => r t - 1) D N (fun n => ((S n).card : ℚ))
    hbound ?_ ?_
  · intro t ht
    have h1 := hr t ht; have h2 := hrW t ht; have h3 := hWN t ht
    simp only; omega
  · intro n hn
    show ((S n).card : ℚ)
        = ∑ t ∈ types, 1 * ((n - (W t - r t + 1)).choose (r t - 1) : ℚ)
    rw [hdecomp_from_partition_family types S fiber r W N hWN hfiber hpart hdisj n hn]
    simp

/-! ## Ed bridge

Materialises the height-fn set `{h : Cell m n → ℤ | IsCanonicalHeight h ∧
numExtrema h = d}` as a `Finset (Cell m n → ℤ)` via `Ed_fibre_finite`, giving
`(Ed_finset d m hm n).card = Ed d m n` for `n ≥ 1`.  The family assembly then
produces the polynomial witness on `((Ed d m n : ℕ) : ℚ)`.
-/

/-- Height-fn fiber as a Finset (for `1 ≤ n`; empty otherwise).  Its card
equals `Ed d m n` on the nonempty range. -/
noncomputable def Ed_finset (d m : ℕ) (hm : 1 ≤ m) (n : ℕ) : Finset (Cell m n → ℤ) :=
  if h : 1 ≤ n then (Ed_fibre_finite d hm h).toFinset else ∅

/-- Card of `Ed_finset` equals `Ed d m n` on the nonempty range `n ≥ 1`. -/
theorem Ed_finset_card_eq_Ed (d m : ℕ) (hm : 1 ≤ m) (n : ℕ) (hn : 1 ≤ n) :
    (Ed_finset d m hm n).card = Ed d m n := by
  unfold Ed_finset
  rw [dif_pos hn]
  exact (Ed_eq_finset_card d hm hn).symm

/-- **End-to-end Ed polynomial witness** (paper `lem:uniform` per-axis
conclusion).  Given a partition of the height-fn set on `Cell m n → ℤ` into
type-fibers indexed by `ι` with each fiber's card equal to a `runExtension_card`
output, `n ↦ ((Ed d m n : ℕ) : ℚ)` agrees on `{n ≥ N}` with a single
polynomial of natDegree ≤ `D` (`D = d - 2` in the paper).

This is exactly the shape `SequelEd.Ed_thm_poly_of_perAxis` consumes as
`hrow`.  Consumer obligations `hpart`, `hdisj`, `hfiber` are the
substrate-heavy contraction-map pieces (define `ι` and `fiber`, prove
partition/disjointness/per-fiber card).  All arithmetic obligations flow
through this wrapper without further work. -/
theorem Ed_polynomial_of_partition
    (d m D N : ℕ) (hm : 1 ≤ m) (hN : 1 ≤ N)
    {ι : Type*}
    (types : Finset ι)
    (fiber : ι → (n : ℕ) → Finset (Cell m n → ℤ))
    (r W : ι → ℕ)
    (hr : ∀ t ∈ types, 1 ≤ r t)
    (hrW : ∀ t ∈ types, r t ≤ W t)
    (hbound : ∀ t ∈ types, r t - 1 ≤ D)
    (hWN : ∀ t ∈ types, W t ≤ N)
    (hfiber : ∀ t ∈ types, ∀ n, W t ≤ n →
        (fiber t n).card = (n - (W t - r t + 1)).choose (r t - 1))
    (hpart : ∀ n, N ≤ n → Ed_finset d m hm n = types.biUnion (fun t => fiber t n))
    (hdisj : ∀ n, N ≤ n → (↑types : Set ι).PairwiseDisjoint (fun t => fiber t n)) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
      ∀ n : ℕ, N ≤ n → ((Ed d m n : ℕ) : ℚ) = p.eval (n : ℚ) := by
  obtain ⟨p, hp_deg, hp_eval⟩ :=
    uniform_polynomial_from_partition_family types (Ed_finset d m hm) fiber r W D N
      hr hrW hbound hWN hfiber hpart hdisj
  refine ⟨p, hp_deg, ?_⟩
  intro n hn
  have hn1 : 1 ≤ n := le_trans hN hn
  rw [← Ed_finset_card_eq_Ed d m hm n hn1]
  exact hp_eval n hn

/-- **hrow shape** (paper `Ed_thm_poly_of_perAxis`'s per-row hypothesis).
Given a per-`m` polynomial witness (from `Ed_polynomial_of_partition`) at
onset `lo`, package into the `hrow`-compatible shape.  Only difference from
`Ed_polynomial_of_partition`'s output is the cast `((Ed d m n : ℕ) : ℚ)` vs
`(Ed d m n : ℚ)`, closed by `exact_mod_cast`. -/
theorem Ed_hrow_of_per_m
    (d D lo : ℕ)
    (per_m : ∀ m, lo ≤ m →
      ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
        ∀ n, lo ≤ n → ((Ed d m n : ℕ) : ℚ) = p.eval (n : ℚ)) :
    ∀ a, lo ≤ a → ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
      ∀ b, lo ≤ b → (Ed d a b : ℚ) = p.eval (b : ℚ) := by
  intro a ha
  obtain ⟨p, hdeg, hev⟩ := per_m a ha
  refine ⟨p, hdeg, ?_⟩
  intro b hb
  have := hev b hb
  exact_mod_cast this

/-- **Full paper `thm:poly` from per-row witness alone** (paper `lem:uniform`
conclusion + `Ed_thm_poly_of_perAxis` chassis + `Ed_symm`).  If for every
`m ≥ lo` there is a polynomial in `n` of natDegree ≤ `D` matching
`(Ed d m n : ℚ)` on `n ≥ lo`, then `Ed d · ·` matches a rank-`≤ D+1`
tensor of polynomials `∑ g_i(b) L_i(a)` symmetric in `(a, b)`.

**One-stop shop for `lem:uniform` consumers.**  The substrate-heavy pieces
are absorbed into the per-`m` witness (typically produced by
`Ed_polynomial_of_partition`); `hcol` is derived from `hrow` via `Ed_symm`,
so a consumer needs to prove the row direction only, not both.  Then this
theorem composes with `Ed_thm_poly_of_perAxis` to yield the bivariate
tensor form of the paper's `thm:poly` (§8). -/
theorem Ed_thm_poly_from_per_m
    (d D lo : ℕ) (hlo : 1 ≤ lo)
    (per_m : ∀ m, lo ≤ m →
      ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
        ∀ n, lo ≤ n → ((Ed d m n : ℕ) : ℚ) = p.eval (n : ℚ)) :
    ∃ (g L : Fin (D + 1) → Polynomial ℚ),
      (∀ i, (g i).natDegree ≤ D) ∧ (∀ i, (L i).natDegree ≤ D) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        (Ed d a b : ℚ) = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) ∧
      (∀ a b : ℚ,
        (∑ i, (g i).eval b * (L i).eval a) =
          (∑ i, (g i).eval a * (L i).eval b)) := by
  have hrow := Ed_hrow_of_per_m d D lo per_m
  have hcol : ∀ b, lo ≤ b → ∃ q : Polynomial ℚ, q.natDegree ≤ D ∧
      ∀ a, lo ≤ a → (Ed d a b : ℚ) = q.eval (a : ℚ) := by
    intro b hb
    obtain ⟨p, hpdeg, hpev⟩ := per_m b hb
    refine ⟨p, hpdeg, ?_⟩
    intro a ha
    have h_symm : Ed d a b = Ed d b a :=
      Ed_symm d (le_trans hlo ha) (le_trans hlo hb)
    have := hpev a ha
    push_cast at this ⊢
    rw [h_symm]; exact_mod_cast this
  exact Ed_thm_poly_of_perAxis d D lo hlo hrow hcol

/-! ## Direct arithmetic interface

For consumers who have a direct combinatorial proof of the count decomposition
(e.g., via a `Fintype` bijection to a concrete indexing set + `card_congr`)
without materialising Finset fibers, the following wrapper bypasses the
partition machinery.  It takes the arithmetic `hdecomp` alone. -/

/-- **Direct arithmetic `hdecomp` interface.**  The lightest-weight `Ed`
polynomial-witness wrapper: skips the Finset partition/disjointness/fiber
machinery, taking only the arithmetic count decomposition (matching the
`runExtension_card` shape) and producing the natDegree ≤ `D` polynomial. -/
theorem Ed_polynomial_direct
    (d m D N : ℕ) (_hm : 1 ≤ m)
    {ι : Type*} (types : Finset ι) (mult : ι → ℚ) (r W : ι → ℕ)
    (hr : ∀ t ∈ types, 1 ≤ r t)
    (hrW : ∀ t ∈ types, r t ≤ W t)
    (hbound : ∀ t ∈ types, r t - 1 ≤ D)
    (hWN : ∀ t ∈ types, W t ≤ N)
    (hdecomp : ∀ n, N ≤ n →
        ((Ed d m n : ℕ) : ℚ)
          = ∑ t ∈ types, mult t * ((n - (W t - r t + 1)).choose (r t - 1) : ℚ)) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
      ∀ n : ℕ, N ≤ n → ((Ed d m n : ℕ) : ℚ) = p.eval (n : ℚ) := by
  refine degreeBound_assembly types mult
    (fun t => W t - r t + 1) (fun t => r t - 1) D N (fun n => ((Ed d m n : ℕ) : ℚ))
    hbound ?_ hdecomp
  intro t ht
  have h1 := hr t ht; have h2 := hrW t ht; have h3 := hWN t ht
  simp only; omega

end OrigamiCone.Sequel
