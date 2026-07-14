import OrigamiCone.SequelEnvThm

/-!
# Sequel meta-theorem: `lem:sep` reverse direction (`separable ⟹ product-grid`)

Standalone formalisation of the reverse direction of the Separability
Lemma of the sequel paper

> *Degree-`d` vertex counts of the `m × n` origami flip graph:
> structure and a polynomial conjecture.*

`Lemma lem:sep` characterises separable envelopes: `E_{A,c}` is additively
separable (`E(i,j) = φ(i) + ψ(j)`) iff its apex set is a product grid
`A = R × C` and its offsets factor `c_{(r,c)} = α_r + β_c`. The forward
direction (`if`) — product apex set ⟹ separable — is proved in
`SequelSep.separable_factor`. This module proves the **reverse** (`only if`) —
separable ⟹ product apex set and factoring offsets.

The mechanism (paper §10.1 reverse):

1. Neighbour differences of `E = φ + ψ` separate axis-by-axis:
   `E(i±1,j) − E(i,j) = φ(i±1) − φ(i)` and `E(i,j±1) − E(i,j) = ψ(j±1) − ψ(j)`.
2. Hence `(i,j)` is a strict local minimum of `E` iff `i` is a strict local
   minimum of `φ` and `j` is a strict local minimum of `ψ`.
3. `SequelEnvThm.envelope_structure_reverse` identifies the strict local
   minima of `E` with the apex set `image p S` (under `Par` and `Active`).
4. So `image p S = SLM(φ) × SLM(ψ)`, a product grid.
5. At each apex, activity forces `Env(p_s) = c_s`, and by (1) also
   `Env(p_s) = φ((p_s).1) + ψ((p_s).2)`; combining gives the offset
   factorisation.

Contents:

* `IsSLMin1D φ k` : `k` is a strict local minimum of the integer function `φ`.
* `nbhd_cases` : 4-neighbour enumeration for `d2 v w = 1`.
* `isSLMin_sep_iff` : the algebraic core — `(∀ w, d2 v w = 1 → f v < f w)` for
  `f = φ + ψ` is equivalent to `IsSLMin1D φ v.1 ∧ IsSLMin1D ψ v.2`.
* `sep_apex_iff_product` : the paper's reverse direction — under `Par` and
  `∀ s ∈ S, Active`, if `Env = φ + ψ` pointwise, then the apex set equals the
  product `SLM(φ) × SLM(ψ)`.
* `sep_offset_factorises` : offset factorisation — `c s = φ (p s).1 + ψ (p s).2`.

Scope: everything on the full `ℤ × ℤ` lattice. The paper's grid bounds
`[1,m] × [1,n]` do not enter the reverse direction — it is a pointwise
statement about the envelope.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.sep_apex_iff_product`.
-/

namespace OrigamiCone.Sequel

/-- Strict local minimum of a 1D integer function: strictly below both immediate
neighbours. -/
def IsSLMin1D (φ : ℤ → ℤ) (k : ℤ) : Prop := φ k < φ (k - 1) ∧ φ k < φ (k + 1)

/-- **Neighbourhood enumeration.** A cell `w` is at `L¹` distance `1` from
`(a, b)` iff it is one of the four lattice neighbours. -/
private lemma nbhd_cases (a b : ℤ) (w : ℤ × ℤ) :
    d2 (a, b) w = 1 ↔ w = (a + 1, b) ∨ w = (a - 1, b) ∨ w = (a, b + 1) ∨ w = (a, b - 1) := by
  refine ⟨?_, ?_⟩
  · intro h
    simp only [d2] at h
    rcases abs_cases (a - w.1) with ⟨e1, h1⟩ | ⟨e1, h1⟩ <;>
    rcases abs_cases (b - w.2) with ⟨e2, h2⟩ | ⟨e2, h2⟩ <;>
    · obtain ⟨w1, w2⟩ := w; simp_all; omega
  · rintro (h | h | h | h) <;> subst h <;> simp only [d2] <;> norm_num

/-- **SLM of an additive separation** (algebraic core of `lem:sep` reverse).
For `f(v) = φ v.1 + ψ v.2`, the pointwise strict-local-minimum condition
`f v < f w` on every unit-distance neighbour `w` is equivalent to the
conjunction of 1D strict-local-minimum conditions in the two components. -/
theorem isSLMin_sep_iff (φ ψ : ℤ → ℤ) (v : ℤ × ℤ) :
    (∀ w : ℤ × ℤ, d2 v w = 1 → (φ v.1 + ψ v.2) < (φ w.1 + ψ w.2)) ↔
    IsSLMin1D φ v.1 ∧ IsSLMin1D ψ v.2 := by
  refine ⟨?fwd, ?rev⟩
  case fwd =>
    intro h
    have hnbhd_dn := h (v.1 - 1, v.2) (by rw [nbhd_cases]; tauto)
    have hnbhd_up := h (v.1 + 1, v.2) (by rw [nbhd_cases]; tauto)
    have hnbhd_lt := h (v.1, v.2 - 1) (by rw [nbhd_cases]; tauto)
    have hnbhd_rt := h (v.1, v.2 + 1) (by rw [nbhd_cases]; tauto)
    refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> simp_all
  case rev =>
    rintro ⟨⟨hφlt, hφgt⟩, ⟨hψlt, hψgt⟩⟩ w hw
    rw [nbhd_cases] at hw
    rcases hw with h | h | h | h <;> subst h <;> simp <;> linarith

variable {ι : Type*} (p : ι → ℤ × ℤ) (c : ι → ℤ) (S : Finset ι) (hS : S.Nonempty)

/-- **`lem:sep` reverse direction** (apex-set half). Given a configuration `(p, c)`
satisfying `Par` and all-active, and a pointwise additive separation
`Env = φ + ψ`, a cell `v` is an apex iff both coordinate projections are
1D strict local minima. -/
theorem sep_apex_iff_product (hpar : Par p c S)
    (hAct : ∀ s ∈ S, Active p c S s) (φ ψ : ℤ → ℤ)
    (hsep : ∀ v : ℤ × ℤ, Env p c S hS v = φ v.1 + ψ v.2) (v : ℤ × ℤ) :
    (∃ s ∈ S, p s = v) ↔ IsSLMin1D φ v.1 ∧ IsSLMin1D ψ v.2 := by
  -- Bridge via `envelope_structure_reverse`: apex ↔ SLM of Env.
  rw [← envelope_structure_reverse p c S hS hpar hAct v]
  -- SLM of Env at v ↔ (∀ w, d2 v w = 1 → Env v < Env w).
  -- Rewrite via `hsep`: Env u = φ u.1 + ψ u.2.
  have hEnvSum : ∀ u : ℤ × ℤ, Env p c S hS u = φ u.1 + ψ u.2 := hsep
  constructor
  · intro hmin
    apply (isSLMin_sep_iff φ ψ v).mp
    intro w hw
    have := hmin w hw
    rw [hEnvSum, hEnvSum] at this
    exact this
  · intro hSLM
    intro w hw
    rw [hEnvSum, hEnvSum]
    exact (isSLMin_sep_iff φ ψ v).mpr hSLM w hw

/-- **`lem:sep` reverse direction** (offset half). Under `Active` hypotheses, the
offset at each apex factorises: `c s = φ (p s).1 + ψ (p s).2`. Combined with
`sep_apex_iff_product`, this exhibits both halves of the paper's `A = R × C`
with `c_{(r, k)} = α_r + β_k` (`α := φ|_R`, `β := ψ|_C`). -/
theorem sep_offset_factorises
    (hAct : ∀ s ∈ S, Active p c S s) (φ ψ : ℤ → ℤ)
    (hsep : ∀ v : ℤ × ℤ, Env p c S hS v = φ v.1 + ψ v.2)
    (s : ι) (hsS : s ∈ S) : c s = φ (p s).1 + ψ (p s).2 := by
  -- c s = Env (p s) from activity, then rewrite via hsep.
  obtain ⟨_, hactlt⟩ := hAct s hsS
  have hEnvps : Env p c S hS (p s) = c s := by
    apply le_antisymm
    · have := Finset.inf'_le (fun t => c t + d2 (p t) (p s)) hsS
      simpa [Env, d2_self] using this
    · apply Finset.le_inf'
      intro t htS
      by_cases hts : t = s
      · subst hts; simp [d2_self]
      · exact le_of_lt (hactlt t htS hts)
  have := hsep (p s)
  rw [hEnvps] at this
  linarith

end OrigamiCone.Sequel
