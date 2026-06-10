/-
GS-CT Duality: the GS NTT is the adjoint of the CT NTT with negated twiddles
under the standard dot product on `Fin n → K`.

Key identity: `∑ i, (gsNtt Z u) i * v i = ∑ i, u i * (ntt (-Z) v) i`

This implies `rank(gsFaultDiffGen Z Z_repl F) = rank(faultDiffGen (-Z) (-Z_repl) F)`,
closing the GS rank lower bound via the CT rank theorem.

The duality follows from the per-group butterfly transpose:
  GS butterfly `(a, b) ↦ (a + b, z(b − a))` matrix `[[1, 1], [−z, z]]`
  CT butterfly `(a, b) ↦ (a − zb, a + zb)` matrix `[[1, −z], [1, z]]`
so `gsGroupLayer z` is the adjoint of `groupLayer (−z)`.
-/

import NttFaultRank.GsNtt
import NttFaultRank.Assembly
import Mathlib.LinearAlgebra.Matrix.Rank

namespace NttFaultRank

variable {K : Type*} [Field K]

/-! ### Negated twiddles -/

namespace LayerSpec
variable (spec : LayerSpec)

/-- Negate all twiddles. -/
def negTwiddles (Z : spec.Twiddles K) : spec.Twiddles K :=
  fun ℓ g => -(Z ℓ g)

lemma negTwiddles_ne_zero {Z : spec.Twiddles K} (hZ : ∀ ℓ g, Z ℓ g ≠ 0) :
    ∀ ℓ g, spec.negTwiddles Z ℓ g ≠ 0 := by
  intro ℓ g; exact neg_ne_zero.mpr (hZ ℓ g)

lemma negTwiddles_delta {Z Z_repl : spec.Twiddles K} {F : spec.FaultSet}
    (hδ : ∀ ℓ g, g ∈ F ℓ → Z ℓ g ≠ Z_repl ℓ g) :
    ∀ ℓ g, g ∈ F ℓ → spec.negTwiddles Z ℓ g ≠ spec.negTwiddles Z_repl ℓ g := by
  intro ℓ g hg; unfold negTwiddles; exact fun h => hδ ℓ g hg (neg_injective h)

/-- `faultedTwiddlesGen (-Z) (-Z') F = -(faultedTwiddlesGen Z Z' F)`. -/
lemma faultedTwiddlesGen_neg (Z Z_repl : spec.Twiddles K) (F : spec.FaultSet) :
    spec.faultedTwiddlesGen (spec.negTwiddles Z) (spec.negTwiddles Z_repl) F =
      spec.negTwiddles (spec.faultedTwiddlesGen Z Z_repl F) := by
  ext ℓ g; simp only [faultedTwiddlesGen, negTwiddles]; split <;> rfl

/-! ### Standard dot product -/

noncomputable def stdDot (u v : Fin spec.n → K) : K :=
  ∑ i : Fin spec.n, u i * v i

lemma stdDot_single_left (v : Fin spec.n → K) (j : Fin spec.n) :
    spec.stdDot (Pi.single j 1) v = v j := by
  simp [stdDot, Pi.single_apply, Finset.sum_ite_eq', Finset.mem_univ, mul_comm]

lemma stdDot_single_right (u : Fin spec.n → K) (i : Fin spec.n) :
    spec.stdDot u (Pi.single i 1) = u i := by
  simp [stdDot, Pi.single_apply, Finset.sum_ite_eq', Finset.mem_univ]

end LayerSpec

/-! ### Per-group bilinear identity -/

section GroupBilinear
variable (L : ℕ)

noncomputable def groupDot (u v : Fin 2 → Fin L → K) : K :=
  ∑ s : Fin 2, ∑ j : Fin L, u s j * v s j

theorem gsGroupLayer_adjoint_groupLayer_neg (z : K) (u v : Fin 2 → Fin L → K) :
    groupDot L (gsGroupLayer L z u) v =
      groupDot L u (groupLayer L (-z) v) := by
  unfold groupDot
  simp only [Fin.sum_univ_two, gsGroupLayer_apply_zero, gsGroupLayer_apply_one,
             groupLayer_apply_zero, groupLayer_apply_one]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro j _; ring

end GroupBilinear

/-! ### Full-layer bilinear identity -/

section FullBilinear
variable (G L : ℕ)

noncomputable def fullDot (u v : Fin G → Fin 2 → Fin L → K) : K :=
  ∑ g : Fin G, groupDot L (u g) (v g)

theorem gsFullLayer_adjoint (zs : Fin G → K)
    (u v : Fin G → Fin 2 → Fin L → K) :
    fullDot G L (gsFullLayer G L zs u) v =
      fullDot G L u (fullLayer G L (fun g => -(zs g)) v) := by
  unfold fullDot
  apply Finset.sum_congr rfl; intro g _
  show groupDot L (gsGroupLayer L (zs g) (u g)) (v g) =
       groupDot L (u g) (groupLayer L (-(zs g)) (v g))
  exact gsGroupLayer_adjoint_groupLayer_neg L (zs g) (u g) (v g)

end FullBilinear

/-! ### Dot product isometry through reshape and funReindex -/

/-- `coordEquiv.symm` reindexes a sum over `Fin(G*(2*L))` to a triple sum. -/
theorem dot_reshape_eq (G L : ℕ) (w w' : Fin (G * (2 * L)) → K) :
    (∑ i, w i * w' i) =
      ∑ g : Fin G, ∑ s : Fin 2, ∑ j : Fin L,
        w ((coordEquiv G L).symm (g, s, j)) * w' ((coordEquiv G L).symm (g, s, j)) := by
  rw [show (∑ g : Fin G, ∑ s : Fin 2, ∑ j : Fin L,
        w ((coordEquiv G L).symm (g, s, j)) * w' ((coordEquiv G L).symm (g, s, j)))
      = ∑ p : Fin G × (Fin 2 × Fin L),
        w ((coordEquiv G L).symm p) * w' ((coordEquiv G L).symm p) from by
    rw [← Finset.univ_product_univ, Finset.sum_product]
    apply Finset.sum_congr rfl; intro g _
    rw [← Finset.univ_product_univ, Finset.sum_product]]
  exact ((coordEquiv G L).symm.sum_comp (fun i => w i * w' i)).symm

/-- `finCastEquiv` reindexes a sum over `Fin n` to `Fin(G*(2*L))`. -/
theorem dot_funReindex_eq (n G L : ℕ) (h : G * (2 * L) = n)
    (v v' : Fin n → K) :
    (∑ i, v i * v' i) =
      ∑ m, v ((finCastEquiv n G L h).symm m) * v' ((finCastEquiv n G L h).symm m) := by
  exact ((finCastEquiv n G L h).symm.sum_comp (fun i => v i * v' i)).symm

/-! ### Per-layer bilinear form -/

namespace LayerSpec
variable (spec : LayerSpec)

/-- **Full bilinear form: gsLayer is the adjoint of layer with negated twiddles.** -/
theorem stdDot_gsLayer_eq
    (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (u v : Fin spec.n → K) :
    spec.stdDot (spec.gsLayer Z ℓ u) v =
      spec.stdDot u (spec.layer (spec.negTwiddles Z) ℓ v) := by
  simp only [stdDot]
  -- Step 1: reindex from Fin n to Fin(G*(2*L)) via finCastEquiv.
  set e₁ := finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)
  have hL : ∀ i, spec.gsLayer Z ℓ u i * v i =
      gsFullLayer' (spec.G ℓ) (spec.L ℓ) (Z ℓ)
        (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) u) (e₁ i) *
      funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) v (e₁ i) := by
    intro i; show (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)).symm _ i * v i = _
    simp [funReindex, finCastEquiv, e₁]
  have hR : ∀ i, u i * spec.layer (spec.negTwiddles Z) ℓ v i =
      funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) u (e₁ i) *
      fullLayer' (spec.G ℓ) (spec.L ℓ) (spec.negTwiddles Z ℓ)
        (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) v) (e₁ i) := by
    intro i; show u i * (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)).symm _ i = _
    simp [funReindex, finCastEquiv, e₁]
  simp_rw [hL, hR]
  rw [e₁.sum_comp (fun m =>
    gsFullLayer' _ _ (Z ℓ) (funReindex _ _ _ (spec.hGL ℓ) u) m *
    funReindex _ _ _ (spec.hGL ℓ) v m)]
  rw [e₁.sum_comp (fun m =>
    funReindex _ _ _ (spec.hGL ℓ) u m *
    fullLayer' _ _ (spec.negTwiddles Z ℓ) (funReindex _ _ _ (spec.hGL ℓ) v) m)]
  -- Step 2: reindex from Fin(G*(2*L)) to (Fin G × (Fin 2 × Fin L)) via coordEquiv.
  set e₂ := coordEquiv (spec.G ℓ) (spec.L ℓ)
  have hL2 : ∀ m, gsFullLayer' (spec.G ℓ) (spec.L ℓ) (Z ℓ)
        (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) u) m *
      funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) v m =
    gsFullLayer (spec.G ℓ) (spec.L ℓ) (Z ℓ)
        (reshape K (spec.G ℓ) (spec.L ℓ)
          (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) u))
        (e₂ m).1 (e₂ m).2.1 (e₂ m).2.2 *
      reshape K (spec.G ℓ) (spec.L ℓ)
        (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) v)
        (e₂ m).1 (e₂ m).2.1 (e₂ m).2.2 := by
    intro m; simp [gsFullLayer', reshape_symm_apply, reshape_apply,
                    LinearMap.comp_apply, LinearEquiv.coe_coe, e₂]
  have hR2 : ∀ m, funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) u m *
      fullLayer' (spec.G ℓ) (spec.L ℓ) (spec.negTwiddles Z ℓ)
        (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) v) m =
    reshape K (spec.G ℓ) (spec.L ℓ)
        (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) u)
        (e₂ m).1 (e₂ m).2.1 (e₂ m).2.2 *
      fullLayer (spec.G ℓ) (spec.L ℓ) (spec.negTwiddles Z ℓ)
        (reshape K (spec.G ℓ) (spec.L ℓ)
          (funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) v))
        (e₂ m).1 (e₂ m).2.1 (e₂ m).2.2 := by
    intro m; simp [fullLayer', reshape_symm_apply, reshape_apply,
                    LinearMap.comp_apply, LinearEquiv.coe_coe, e₂]
  simp_rw [hL2, hR2]
  rw [e₂.sum_comp (fun p =>
    gsFullLayer _ _ (Z ℓ)
      (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) u)) p.1 p.2.1 p.2.2 *
    reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) v) p.1 p.2.1 p.2.2)]
  rw [e₂.sum_comp (fun p =>
    reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) u) p.1 p.2.1 p.2.2 *
    fullLayer _ _ (spec.negTwiddles Z ℓ)
      (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) v)) p.1 p.2.1 p.2.2)]
  -- Step 3: the sum over (Fin G × (Fin 2 × Fin L)) = ∑_g ∑_(s,j) = fullDot.
  -- Convert product-type sum to nested sums matching fullDot/groupDot.
  have lhs_eq : (∑ p : Fin (spec.G ℓ) × (Fin 2 × Fin (spec.L ℓ)),
      gsFullLayer (spec.G ℓ) (spec.L ℓ) (Z ℓ)
        (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) u)) p.1 p.2.1 p.2.2 *
      reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) v) p.1 p.2.1 p.2.2) =
    fullDot (spec.G ℓ) (spec.L ℓ)
      (gsFullLayer (spec.G ℓ) (spec.L ℓ) (Z ℓ)
        (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) u)))
      (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) v)) := by
    simp only [fullDot, groupDot]
    rw [← Finset.univ_product_univ, Finset.sum_product]
    apply Finset.sum_congr rfl; intro g _
    rw [← Finset.univ_product_univ, Finset.sum_product]
  have rhs_eq : (∑ p : Fin (spec.G ℓ) × (Fin 2 × Fin (spec.L ℓ)),
      reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) u) p.1 p.2.1 p.2.2 *
      fullLayer (spec.G ℓ) (spec.L ℓ) (spec.negTwiddles Z ℓ)
        (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) v)) p.1 p.2.1 p.2.2) =
    fullDot (spec.G ℓ) (spec.L ℓ)
      (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) u))
      (fullLayer (spec.G ℓ) (spec.L ℓ) (spec.negTwiddles Z ℓ)
        (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) v))) := by
    simp only [fullDot, groupDot]
    rw [← Finset.univ_product_univ, Finset.sum_product]
    apply Finset.sum_congr rfl; intro g _
    rw [← Finset.univ_product_univ, Finset.sum_product]
  rw [lhs_eq, rhs_eq]
  exact gsFullLayer_adjoint (spec.G ℓ) (spec.L ℓ) (Z ℓ)
    (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) u))
    (reshape K _ _ (funReindex _ _ _ (spec.hGL ℓ) v))

/-- Entry-level transpose for one GS/CT layer. -/
theorem gsLayer_entry_transpose
    (Z : spec.Twiddles K) (ℓ : Fin spec.N)
    (i j : Fin spec.n) :
    spec.gsLayer Z ℓ (Pi.single j 1) i =
      spec.layer (spec.negTwiddles Z) ℓ (Pi.single i 1) j := by
  -- gsLayer(e_j)(i) = stdDot(gsLayer(e_j), e_i) = stdDot(e_j, layer(-Z)(e_i)) = layer(-Z)(e_i)(j)
  have h1 : spec.gsLayer Z ℓ (Pi.single j 1) i =
      spec.stdDot (spec.gsLayer Z ℓ (Pi.single j 1)) (Pi.single i 1) :=
    (spec.stdDot_single_right _ i).symm
  have h2 : spec.layer (spec.negTwiddles Z) ℓ (Pi.single i 1) j =
      spec.stdDot (Pi.single j 1) (spec.layer (spec.negTwiddles Z) ℓ (Pi.single i 1)) :=
    (spec.stdDot_single_left _ j).symm
  rw [h1, spec.stdDot_gsLayer_eq Z ℓ (Pi.single j 1) (Pi.single i 1), h2]

/-! ### gsNtt bilinear form by induction -/

/-- **Full bilinear form: gsPartial is the adjoint of runPrefix with negated twiddles.**
    By induction on `m`: at each step peel off the outermost gsLayer. -/
theorem stdDot_gsPartial_eq
    (Z : spec.Twiddles K) (m : ℕ) (hm : m ≤ spec.N) :
    ∀ (u v : Fin spec.n → K),
    spec.stdDot (spec.gsPartial Z m hm u) v =
      spec.stdDot u (spec.runPrefix (spec.negTwiddles Z) m hm v) := by
  induction m with
  | zero =>
    intro u v; simp [gsPartial, LayerSpec.runPrefix, stdDot]
  | succ m ih =>
    intro u v
    rw [spec.gsPartial_succ Z m hm]
    simp only [LinearMap.comp_apply]
    -- gsPartial(m+1)(u) = gsPartial(m)(gsLayer(m)(u))
    rw [ih (Nat.le_of_succ_le hm) (spec.gsLayer Z ⟨m, hm⟩ u) v]
    -- Goal: stdDot(gsLayer(m)(u), runPrefix(-Z, m)(v))
    --     = stdDot(u, layer(-Z, m)(runPrefix(-Z, m)(v)))
    rw [spec.stdDot_gsLayer_eq Z ⟨m, hm⟩ u
        (spec.runPrefix (spec.negTwiddles Z) m (Nat.le_of_succ_le hm) v)]
    -- Goal: stdDot(u, layer(-Z, m)(runPrefix(-Z, m)(v)))
    --     = stdDot(u, runPrefix(-Z, m+1)(v))
    rw [spec.runPrefix_succ (spec.negTwiddles Z) m hm]
    simp only [LinearMap.comp_apply]

/-- **Full bilinear form: gsNtt is the adjoint of ntt with negated twiddles.** -/
theorem stdDot_gsNtt_eq
    (Z : spec.Twiddles K)
    (u v : Fin spec.n → K) :
    spec.stdDot (spec.gsNtt Z u) v =
      spec.stdDot u (spec.ntt (spec.negTwiddles Z) v) := by
  -- gsNtt Z = gsPartial Z N, ntt(-Z) = runPrefix(-Z) N
  rw [spec.gsNtt_eq_gsPartial Z]
  show spec.stdDot (spec.gsPartial Z spec.N le_rfl u) v =
       spec.stdDot u (spec.ntt (spec.negTwiddles Z) v)
  rw [spec.stdDot_gsPartial_eq Z spec.N le_rfl u v]
  -- runPrefix(-Z, N) = ntt(-Z)
  rw [spec.ntt_eq_runPrefix_last (spec.negTwiddles Z)]

/-- Entry-level transpose for gsNtt. -/
theorem gsNtt_entry_transpose
    (Z : spec.Twiddles K)
    (i j : Fin spec.n) :
    spec.gsNtt Z (Pi.single j 1) i =
      spec.ntt (spec.negTwiddles Z) (Pi.single i 1) j := by
  have h1 := (spec.stdDot_single_right (spec.gsNtt Z (Pi.single j 1)) i).symm
  have h2 := (spec.stdDot_single_left (spec.ntt (spec.negTwiddles Z) (Pi.single i 1)) j).symm
  rw [h1, spec.stdDot_gsNtt_eq Z (Pi.single j 1) (Pi.single i 1), h2]

/-! ### Fault-diff transpose and rank equality -/

theorem gsFaultDiffGen_entry_transpose
    (Z Z_repl : spec.Twiddles K) (F : spec.FaultSet)
    (i j : Fin spec.n) :
    spec.gsFaultDiffGen Z Z_repl F (Pi.single j 1) i =
      spec.faultDiffGen (spec.negTwiddles Z) (spec.negTwiddles Z_repl) F
        (Pi.single i 1) j := by
  simp only [gsFaultDiffGen, gsNttFaultGen, faultDiffGen, nttFaultGen,
             LinearMap.sub_apply, Pi.sub_apply]
  rw [spec.faultedTwiddlesGen_neg Z Z_repl F]
  rw [spec.gsNtt_entry_transpose Z i j,
      spec.gsNtt_entry_transpose (spec.faultedTwiddlesGen Z Z_repl F) i j]

theorem gsFaultDiffGen_toMatrix'_eq_transpose
    (Z Z_repl : spec.Twiddles K) (F : spec.FaultSet) :
    LinearMap.toMatrix' (spec.gsFaultDiffGen Z Z_repl F) =
    (LinearMap.toMatrix'
      (spec.faultDiffGen (spec.negTwiddles Z) (spec.negTwiddles Z_repl) F)).transpose := by
  ext i j
  simp only [LinearMap.toMatrix'_apply, Matrix.transpose_apply]
  exact spec.gsFaultDiffGen_entry_transpose Z Z_repl F i j

private lemma rank_toMatrix'_eq (f : (Fin spec.n → K) →ₗ[K] (Fin spec.n → K)) :
    (LinearMap.toMatrix' f).rank = Module.finrank K (LinearMap.range f) := by
  unfold Matrix.rank
  rw [← Matrix.toLin'_apply' (LinearMap.toMatrix' f), Matrix.toLin'_toMatrix']

/-- **Rank equality: GS fault-diff has the same rank as the CT fault-diff
    with negated twiddles.** -/
theorem gsFaultDiffGen_rank_eq
    (Z Z_repl : spec.Twiddles K) (F : spec.FaultSet) :
    Module.finrank K (LinearMap.range (spec.gsFaultDiffGen Z Z_repl F)) =
    Module.finrank K (LinearMap.range (spec.faultDiffGen
      (spec.negTwiddles Z) (spec.negTwiddles Z_repl) F)) := by
  rw [← spec.rank_toMatrix'_eq, ← spec.rank_toMatrix'_eq]
  rw [spec.gsFaultDiffGen_toMatrix'_eq_transpose Z Z_repl F]
  exact Matrix.rank_transpose _

end LayerSpec
end NttFaultRank
