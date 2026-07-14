import OrigamiCone.Cone

/-!
# The Parity Lemma (Lemma 3.x)

Formalisation of the **Parity Lemma**:

> For distinct cells `p₁, p₂` with `D = d(p₁,p₂)` and any integer `δ ≡ D (mod 2)`,
> the function `v ↦ min(d(p₁,v), δ + d(p₂,v))` is a height function.

This is the construction step for cone-pair height functions: given two apexes
and an offset of the correct parity, the lower envelope of the two distance cones
is again a height function (changes by exactly one across each edge).

The proof: the lower envelope of two `1`-Lipschitz functions is `1`-Lipschitz, so
it changes by at most one across each edge; and the parity condition forces both
distance cones — hence their minimum — to have the parity of `i+j` up to a common
shift, so the value genuinely changes (by exactly one) across each edge, where
`i+j` flips parity.

Main result: `parity_isHeight`.  No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Distance parity.** The grid distance `d(p,v)` has the parity of
`p₁+p₂+v₁+v₂` (sum of all four coordinates): `d(p,v) ≡ p.1+p.2+v.1+v.2 (mod 2)`.
-/
lemma gdist_parity (p v : Cell m n) :
    (gdist p v - ((p.1.val : ℤ) + p.2.val + v.1.val + v.2.val)) % 2 = 0 := by
  unfold gdist
  omega

/-- **Cone-pair offset parity.** If `δ ≡ d(p₁,p₂) (mod 2)`, then at every cell `v`
the two distance cones `d(p₁,v)` and `δ + d(p₂,v)` have the same parity. -/
lemma cone_pair_same_parity {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) (v : Cell m n) :
    (gdist p₁ v - (δ + gdist p₂ v)) % 2 = 0 := by
  have e1 := gdist_parity p₁ v
  have e2 := gdist_parity p₂ v
  have e3 := gdist_parity p₁ p₂
  omega

/-- **Parity Lemma.** For distinct cells `p₁, p₂` and an offset `δ` of the same
parity as `D = d(p₁,p₂)`, the lower envelope
`f v = min(d(p₁,v), δ + d(p₂,v))` is a height function: it changes by exactly one
across every edge. -/
theorem parity_isHeight {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) :
    IsHeight (fun v => min (gdist p₁ v) (δ + gdist p₂ v)) := by
  intro v v' hvv'
  -- the four cone values at the two endpoints
  set a := gdist p₁ v with ha
  set b := δ + gdist p₂ v with hb
  set a' := gdist p₁ v' with ha'
  set b' := δ + gdist p₂ v' with hb'
  -- each distance cone changes by exactly ±1 across the edge
  have hstep1 : gdist p₁ v = gdist p₁ v' + 1 ∨ gdist p₁ v = gdist p₁ v' - 1 :=
    gdist_adj_step (q := p₁) hvv'
  have hstep2 : gdist p₂ v = gdist p₂ v' + 1 ∨ gdist p₂ v = gdist p₂ v' - 1 :=
    gdist_adj_step (q := p₂) hvv'
  -- at v, the two cones share parity; at v', likewise; and the two parities differ
  have hpar : (a - b) % 2 = 0 := cone_pair_same_parity hδ v
  have hpar' : (a' - b') % 2 = 0 := cone_pair_same_parity hδ v'
  -- assemble: min changes by exactly one
  rw [abs_eq (by norm_num : (0:ℤ) ≤ 1)]
  simp only [ha, hb, ha', hb'] at *
  omega

end OrigamiCone
