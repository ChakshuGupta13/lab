import Mathlib

/-!
# Sequel meta-theorem: separability of a product-grid envelope (`lem:sep`, if-direction)

Standalone formalisation of the forward direction of the Separability Lemma of the
sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:sep` characterises when a lower envelope `E_{A,c}` factors additively:

> The envelope `E_{A,c}` is additively separable iff its apex set is a product grid
> `A = R × C` of a row set `R` and a column set `C`, and its offsets factor as
> `c_{(r,c)} = α_r + β_c`.

The **if-direction** is the explicit min-factorisation used throughout the leading
-coefficient and dimension analysis:

> *Proof (if).* If `A = R × C` with `c_{(r,c)} = α_r + β_c`, the minimum factors:
> `E(i,j) = min_{r∈R, c∈C}(α_r + β_c + |i-r| + |j-c|)
>         = min_{r∈R}(α_r + |i-r|) + min_{c∈C}(β_c + |j-c|)`.

This module proves that factorisation, working on the integer lattice `ℤ × ℤ` with
apexes given by a product of two abstract nonempty index sets:

* `env1` : the one-dimensional lower envelope `min_s (o_s + |pos_s - t|)` on a path;
* `env2prod` : the two-dimensional envelope over the product apex grid `R ×ˢ C`,
  apex `(r,c)` seated at `(pos_R r, pos_C c)` with offset `α_r + β_c`;
* `separable_factor` (`lem:sep`, if-direction): `env2prod` factors as
  `env1 (row part) + env1 (column part)`, i.e. the product-grid envelope is
  additively separable.

This is exactly the `Env` of the companion module `SequelEnvelope` specialised to a
product apex set `A = R ×ˢ C` with cone seats `(r,c) ↦ (pos_R r, pos_C c)` and
offsets `(r,c) ↦ α_r + β_c`, using `|pos_R r - i| + |pos_C c - j| = d(seat, (i,j))`.

Scope: only the **if-direction** is proved. The converse (`E = φ + ψ` ⟹ the apex
set is a product grid with factoring offsets) needs the Envelope Structure Theorem
identification of the minimum set as the apex set together with
`SLM(φ + ψ) = SLM(φ) × SLM(ψ)`, the strict-local-minimum product formula, which
relies on the extremum machinery and is **not** formalised here.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.separable_factor`.
-/

namespace OrigamiCone.Sequel

open Finset

/-- The one-dimensional lower envelope `min_s (o_s + |pos_s - t|)` on a path, with
cone seats `pos` and offsets `o` over a nonempty index set `S`. -/
def env1 {ι : Type*} (S : Finset ι) (hS : S.Nonempty) (o pos : ι → ℤ) (t : ℤ) : ℤ :=
  S.inf' hS (fun s => o s + |pos s - t|)

/-- The two-dimensional lower envelope over the **product** apex grid `R ×ˢ C`: the
apex `(r, c)` is seated at `(pos_R r, pos_C c)` with offset `α_r + β_c`. -/
def env2prod {ιR ιC : Type*} (R : Finset ιR) (C : Finset ιC)
    (hR : R.Nonempty) (hC : C.Nonempty) (α : ιR → ℤ) (β : ιC → ℤ)
    (posR : ιR → ℤ) (posC : ιC → ℤ) (i j : ℤ) : ℤ :=
  (R ×ˢ C).inf' (hR.product hC)
    (fun p => (α p.1 + β p.2) + (|posR p.1 - i| + |posC p.2 - j|))

/-- **Separable factorisation** (`lem:sep`, if-direction only — the converse is
disclaimed below). The lower envelope over a product apex grid factors additively
into the two one-dimensional envelopes — `E_{R×C, α⊕β}(i,j) = φ(i) + ψ(j)` with
`φ = env1 R …`, `ψ = env1 C …`. -/
theorem separable_factor {ιR ιC : Type*} (R : Finset ιR) (C : Finset ιC)
    (hR : R.Nonempty) (hC : C.Nonempty) (α : ιR → ℤ) (β : ιC → ℤ)
    (posR : ιR → ℤ) (posC : ιC → ℤ) (i j : ℤ) :
    env2prod R C hR hC α β posR posC i j
      = env1 R hR α posR i + env1 C hC β posC j := by
  unfold env2prod env1
  apply le_antisymm
  · obtain ⟨r0, hr0, hr0e⟩ := R.exists_mem_eq_inf' hR (fun r => α r + |posR r - i|)
    obtain ⟨c0, hc0, hc0e⟩ := C.exists_mem_eq_inf' hC (fun c => β c + |posC c - j|)
    rw [hr0e, hc0e]
    have hmem : (r0, c0) ∈ R ×ˢ C := mem_product.mpr ⟨hr0, hc0⟩
    calc (R ×ˢ C).inf' (hR.product hC)
            (fun p => (α p.1 + β p.2) + (|posR p.1 - i| + |posC p.2 - j|))
          ≤ (α r0 + β c0) + (|posR r0 - i| + |posC c0 - j|) := Finset.inf'_le _ hmem
      _ = (α r0 + |posR r0 - i|) + (β c0 + |posC c0 - j|) := by ring
  · apply Finset.le_inf'
    rintro ⟨r, c⟩ hrc
    rw [mem_product] at hrc
    have h1 : R.inf' hR (fun r => α r + |posR r - i|) ≤ α r + |posR r - i| :=
      Finset.inf'_le _ hrc.1
    have h2 : C.inf' hC (fun c => β c + |posC c - j|) ≤ β c + |posC c - j| :=
      Finset.inf'_le _ hrc.2
    calc R.inf' hR (fun r => α r + |posR r - i|) + C.inf' hC (fun c => β c + |posC c - j|)
        ≤ (α r + |posR r - i|) + (β c + |posC c - j|) := add_le_add h1 h2
      _ = (α r + β c) + (|posR r - i| + |posC c - j|) := by ring

end OrigamiCone.Sequel
