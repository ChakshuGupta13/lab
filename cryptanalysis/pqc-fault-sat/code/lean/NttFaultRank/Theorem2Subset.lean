/-
Phase E — Theorem 2 (⊇) direction: `e₀, e₁ ∈ ker (faultDiff)`.

This file decomposes the kernel-inclusion direction into named lemmas
and proves them. Status: FULLY PROVED (no sorry). Standard mathlib
axioms only.

  Pen-and-paper argument:
  1. (`layer_agree_of_b_zero`) Two layers with twiddles agreeing off
     group `g₀` produce the same output when the input's b-side at
     group `g₀` is zero.
  2. (`runPrefix_basis_bSide_zero`) Inductive invariant: after applying
     layers 0..k-1 to `basis p` (p ∈ {0, 1}), the b-side at every
     layer-k group is zero. Proved by induction on k: the support of
     the running image is `{p + g · 2 · L k : g < G k}`, all on a-sides.
  3. (`runPrefix_eq_under_fault`) By induction on k, applying the
     unfaulted and faulted NTT prefixes to `basis p` gives the same
     vector. Inductive step combines (1) and (2).
  4. (`basis_mem_ker`) Specialise (3) to `k = N` to conclude
     `basis p ∈ ker (faultDiff Z F)`.
-/

import NttFaultRank.Theorem2
import NttFaultRank.LayerFlat

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-- The first `k` layers composed via `Fin.foldl`, matching the
    definition of `ntt`. So `runPrefix Z spec.N = ntt Z` definitionally. -/
def runPrefix (Z : spec.Twiddles K) (k : ℕ) (hk : k ≤ spec.N) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  Fin.foldl k (fun acc i => (spec.layer Z (i.castLE hk)).comp acc) LinearMap.id

/-- The b-side projection of a vector `v : Fin spec.n → K` at layer ℓ
    group g: extract index `(g, 1, j)` from the reshape. -/
def bSide (v : Fin spec.n → K) (ℓ : Fin spec.N) (g : Fin (spec.G ℓ))
    (j : Fin (spec.L ℓ)) : K :=
  reshape K (spec.G ℓ) (spec.L ℓ)
    ((funReindex spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)) v) g 1 j

/-! ### Step 1 — localised agreement -/

/-- Helper on the indexed form: if twiddles agree off `g₀` and the
    b-side at group `g₀` is zero, the full indexed layer outputs agree. -/
lemma fullLayer_agree_of_b_zero {G L : ℕ} {zs zs' : Fin G → K}
    {g₀ : Fin G} (hagree : ∀ g ≠ g₀, zs g = zs' g)
    {w : Fin G → Fin 2 → Fin L → K} (hb : ∀ j, w g₀ 1 j = 0) :
    fullLayer G L zs w = fullLayer G L zs' w := by
  funext g
  simp only [fullLayer_apply]
  by_cases hg : g = g₀
  · subst hg
    funext s j
    rcases fin2_cases s with rfl | rfl
    · rw [groupLayer_apply_zero, groupLayer_apply_zero, hb j]
      ring
    · rw [groupLayer_apply_one, groupLayer_apply_one, hb j]
      ring
  · rw [hagree g hg]

/-- If `Z` and `Z'` agree off `g₀` at layer `ℓ`, and the b-side of `v`
    at layer-ℓ group `g₀` is zero, then `layer Z ℓ v = layer Z' ℓ v`. -/
theorem layer_agree_of_b_zero
    {Z Z' : spec.Twiddles K} {ℓ : Fin spec.N} {g₀ : Fin (spec.G ℓ)}
    (hagree : ∀ g ≠ g₀, Z ℓ g = Z' ℓ g)
    {v : Fin spec.n → K} (hb : ∀ j, spec.bSide v ℓ g₀ j = 0) :
    spec.layer Z ℓ v = spec.layer Z' ℓ v := by
  show layerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z ℓ) v =
       layerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z' ℓ) v
  unfold layerOn fullLayer'
  simp only [LinearMap.comp_apply, LinearEquiv.coe_coe]
  congr 2
  exact fullLayer_agree_of_b_zero (g₀ := g₀) hagree hb

/-- `runPrefix Z N = ntt Z` definitionally. -/
lemma ntt_eq_runPrefix_last (Z : spec.Twiddles K) :
    spec.ntt Z = spec.runPrefix Z spec.N le_rfl := by
  rfl

/-! ### Step 2 — support invariant -/

/-- Flat-index formula: `(coordEquiv G L).symm (g, s, j) = g * (2*L) + s * L + j`. -/
lemma coordEquiv_symm_val (G L : ℕ) (g : Fin G) (s : Fin 2) (j : Fin L) :
    ((coordEquiv G L).symm (g, s, j)).val = g.val * (2 * L) + s.val * L + j.val := by
  unfold coordEquiv
  simp [finProdFinEquiv, Equiv.prodCongr]
  ring

/-- The flat index corresponding to `bSide v ℓ g j`. -/
def bSideIdx (ℓ : Fin spec.N) (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    Fin spec.n :=
  (finCastEquiv spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ)).symm
    ((coordEquiv (spec.G ℓ) (spec.L ℓ)).symm (g, 1, j))

lemma bSideIdx_val (ℓ : Fin spec.N) (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    (spec.bSideIdx ℓ g j).val = g.val * (2 * spec.L ℓ) + spec.L ℓ + j.val := by
  unfold bSideIdx
  simp [finCastEquiv, coordEquiv_symm_val]

/-- `bSide v ℓ g j = v (bSideIdx ℓ g j)`. -/
lemma bSide_eq_apply (v : Fin spec.n → K) (ℓ : Fin spec.N)
    (g : Fin (spec.G ℓ)) (j : Fin (spec.L ℓ)) :
    spec.bSide v ℓ g j = v (spec.bSideIdx ℓ g j) := by
  rfl

/-- Base case: at layer 0, `runPrefix Z 0 (basis p) = basis p`; the b-side
    is zero provided `G 0 = 1` and `p.val < L 0`. -/
lemma runPrefix_basis_bSide_zero_at_zero
    (p : Fin spec.n)
    (h0 : 0 < spec.N)
    (hp_lt : p.val < spec.L ⟨0, h0⟩) :
    ∀ g : Fin (spec.G ⟨0, h0⟩), ∀ j : Fin (spec.L ⟨0, h0⟩),
      spec.bSide (spec.basis (K := K) p) ⟨0, h0⟩ g j = 0 := by
  intro g j
  rw [bSide_eq_apply (K := K)]
  unfold basis
  rw [if_neg]
  intro hb
  have hval := spec.bSideIdx_val ⟨0, h0⟩ g j
  have heq : (spec.bSideIdx ⟨0, h0⟩ g j).val = p.val := by rw [hb]
  rw [hval] at heq
  omega

/-! ### Support invariant (kernel-inclusion bookkeeping)

    Mechanises the paper's prose argument:

      "after processing layers 0, …, ℓ-1, the nonzero entries of e_p
       sit at relative position p within each layer-ℓ group, satisfying
       p < len_ℓ, so position p is an a-input, making every b-input
       zero at layer ℓ"

    The induction goes through via the flat-index layer formula
    `(layer Z ℓ_k v) i = v(i) + z · v(partner i)` (or the b-side
    variant), where the partner index has the same value mod L_{ℓ_k} as i.
    Both `v i` and `v (partner i)` lie in the IH-zero class, so their
    combination is zero.

    Status: FULLY PROVED (no sorry). Uses `KyberLike` recurrence
    `L_{k+1} · 2 = L_k` and `G_{k+1} = 2 G_k`. -/

/-- **Support invariant.**

    After applying `k` layers, the value at flat index `i` is zero unless
    `i.val % (2 * spec.L k) = p.val % (2 * spec.L k)`. This characterises
    the support as the orbit `{p + g · (2 L k) : g ∈ Fin G k}`. -/
theorem runPrefix_basis_support_invariant
    [KyberLike spec] (Z : spec.Twiddles K) (p : Fin spec.n)
    (k : ℕ) (hk : k ≤ spec.N) (ℓ : Fin spec.N) (hℓ : ℓ.val = k) :
    ∀ i : Fin spec.n,
      i.val % (2 * spec.L ℓ) ≠ p.val % (2 * spec.L ℓ) →
      (spec.runPrefix Z k hk (spec.basis p)) i = 0 := by
  -- Generalize ℓ and hℓ so that the IH quantifies over them.
  revert ℓ hℓ
  induction k with
  | zero =>
    intro ℓ hℓ
    -- Base case: runPrefix Z 0 = id, so the value is (basis p) i.
    -- We need to show: i.val % (2 L_0) ≠ p.val % (2 L_0) ⇒ (basis p) i = 0.
    -- Since 2 * L_0 = n (by KyberLike.L_zero_doubled), the modular condition
    -- reduces to i.val ≠ p.val (both are < n), i.e. i ≠ p.
    intro i hmod
    have h0lt : 0 < spec.N := by
      have := ℓ.isLt; omega
    have hL0 : 2 * spec.L ℓ = spec.n := by
      have hzero : ℓ = ⟨0, h0lt⟩ := by
        apply Fin.ext; exact hℓ
      rw [hzero]
      exact KyberLike.L_zero_doubled (spec := spec) h0lt
    -- Now i.val < n and p.val < n, both reduced mod n are themselves.
    have hi_mod : i.val % (2 * spec.L ℓ) = i.val := by
      rw [hL0]; exact Nat.mod_eq_of_lt i.isLt
    have hp_mod : p.val % (2 * spec.L ℓ) = p.val := by
      rw [hL0]; exact Nat.mod_eq_of_lt p.isLt
    rw [hi_mod, hp_mod] at hmod
    -- hmod : i.val ≠ p.val. Reduce runPrefix Z 0 to identity, then basis check.
    have hrun : spec.runPrefix Z 0 hk (spec.basis (K := K) p) = spec.basis (K := K) p := by
      simp [runPrefix]
    rw [hrun]
    unfold basis
    rw [if_neg]
    intro habs
    apply hmod
    rw [habs]
  | succ k ih =>
    intro ℓ hℓ
    -- ℓ has ℓ.val = k+1; let ℓ_k be the previous layer.
    -- Goal: runPrefix Z (k+1) hk (basis p) i = 0 when
    --       i.val % (2 L_ℓ) ≠ p.val % (2 L_ℓ).
    -- Use runPrefix_succ to split: layer Z ⟨k,_⟩ ∘ runPrefix Z k.
    have hk1_lt_N : k + 1 < spec.N := hℓ ▸ ℓ.isLt
    have hkN : k < spec.N := Nat.lt_of_succ_lt hk1_lt_N
    have hk_le : k ≤ spec.N := Nat.le_of_lt hkN
    -- ℓ_k: the previous layer index.
    set ℓ_k : Fin spec.N := ⟨k, hkN⟩ with hℓk_def
    -- KyberLike recurrence: 2 L_ℓ = L_{ℓ_k}.
    have hLrec : 2 * spec.L ℓ = spec.L ℓ_k := by
      have heq : ℓ = ⟨k + 1, hk1_lt_N⟩ := by
        apply Fin.ext; exact hℓ
      have hLs := KyberLike.L_succ (spec := spec) k hk1_lt_N
      rw [heq]
      linarith
    intro i hmod
    -- Apply runPrefix unfold (inline via Fin.foldl_succ_last)
    have hsucc : spec.runPrefix Z (k + 1) hk =
        (spec.layer Z ⟨k, hkN⟩).comp
          (spec.runPrefix Z k hk_le) := by
      unfold runPrefix
      rw [Fin.foldl_succ_last]
      rfl
    rw [hsucc]
    simp only [LinearMap.comp_apply]
    -- Goal: (layer Z ℓ_k (runPrefix Z k _ (basis p))) i = 0
    -- Apply layer_eq_zero_of_partner_zero with IH at i and at partnerIdx ℓ_k i.
    set v := spec.runPrefix Z k hk_le (spec.basis (K := K) p)
    apply spec.layer_eq_zero_of_partner_zero Z ℓ_k v i
    · -- IH at i with ℓ' = ℓ_k.
      have ih_at_i := ih hk_le ℓ_k rfl i
      apply ih_at_i
      intro habs_2L
      apply hmod
      rw [hLrec]
      have hdvd : spec.L ℓ_k ∣ 2 * spec.L ℓ_k := ⟨2, by ring⟩
      calc i.val % spec.L ℓ_k
          = (i.val % (2 * spec.L ℓ_k)) % spec.L ℓ_k := (Nat.mod_mod_of_dvd _ hdvd).symm
        _ = (p.val % (2 * spec.L ℓ_k)) % spec.L ℓ_k := by rw [habs_2L]
        _ = p.val % spec.L ℓ_k := Nat.mod_mod_of_dvd _ hdvd
    · -- IH at partnerIdx ℓ_k i with ℓ' = ℓ_k.
      have ih_at_p := ih hk_le ℓ_k rfl (spec.partnerIdx ℓ_k i)
      apply ih_at_p
      intro habs_2L
      apply hmod
      rw [hLrec]
      have hmod_eq := spec.partnerIdx_mod_L_eq ℓ_k i
      have hdvd : spec.L ℓ_k ∣ 2 * spec.L ℓ_k := ⟨2, by ring⟩
      calc i.val % spec.L ℓ_k
          = (spec.partnerIdx ℓ_k i).val % spec.L ℓ_k := hmod_eq.symm
        _ = ((spec.partnerIdx ℓ_k i).val % (2 * spec.L ℓ_k)) % spec.L ℓ_k :=
              (Nat.mod_mod_of_dvd _ hdvd).symm
        _ = (p.val % (2 * spec.L ℓ_k)) % spec.L ℓ_k := by rw [habs_2L]
        _ = p.val % spec.L ℓ_k := Nat.mod_mod_of_dvd _ hdvd


/-- The b-side at every layer-k group of `runPrefix Z k (basis p)` is
    zero, when `p < spec.L k`. Follows from the support invariant
    (`runPrefix_basis_support_invariant`) by direct arithmetic on the
    `bSideIdx`. -/
theorem runPrefix_basis_bSide_zero
    [KyberLike spec] (Z : spec.Twiddles K) (p : Fin spec.n)
    (k : Fin spec.N) (hp_lt : p.val < spec.L k) :
    ∀ g : Fin (spec.G k), ∀ j : Fin (spec.L k),
      spec.bSide (spec.runPrefix Z k.val (Nat.le_of_lt k.isLt)
                    (spec.basis p)) k g j = 0 := by
  intro g j
  rw [bSide_eq_apply]
  apply spec.runPrefix_basis_support_invariant Z p k.val (Nat.le_of_lt k.isLt) k rfl
  rw [bSideIdx_val]
  have hjlt : j.val < spec.L k := j.isLt
  have hpL : p.val < spec.L k := hp_lt
  have hLpos : 0 < spec.L k := by linarith
  have h2Lpos : 0 < 2 * spec.L k := by linarith
  -- LHS mod = (L + j) since g*(2L) is a multiple of (2L), and L+j < 2L.
  -- RHS mod = p (since p < L < 2L).
  -- These differ since L+j ≥ L > p.
  have hLHS : (g.val * (2 * spec.L k) + spec.L k + j.val) % (2 * spec.L k)
               = spec.L k + j.val := by
    have : g.val * (2 * spec.L k) + spec.L k + j.val
            = (spec.L k + j.val) + (2 * spec.L k) * g.val := by ring
    rw [this, Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt (by linarith)
  have hRHS : p.val % (2 * spec.L k) = p.val :=
    Nat.mod_eq_of_lt (by linarith)
  rw [hLHS, hRHS]
  intro habs
  linarith

/-! ### Step 3 — prefix agreement under fault -/

/-- Unfolding lemma: `runPrefix Z (k+1) = layer Z ⟨k,·⟩ ∘ runPrefix Z k`. -/
lemma runPrefix_succ (Z : spec.Twiddles K) (k : ℕ) (hk : k + 1 ≤ spec.N) :
    spec.runPrefix Z (k + 1) hk =
      (spec.layer Z ⟨k, hk⟩).comp
        (spec.runPrefix Z k (Nat.le_of_succ_le hk)) := by
  unfold runPrefix
  rw [Fin.foldl_succ_last]
  rfl

/-- The faulted twiddles agree with `Z` off the unique fault index at
    each layer. -/
lemma faultedTwiddles_agree_off {Z : spec.Twiddles K} {F : spec.FaultSet}
    (hF : spec.OneFaultPerLayer F) (ℓ : Fin spec.N) :
    let g₀ := (Finset.card_eq_one.mp (hF ℓ)).choose
    ∀ g ≠ g₀, Z ℓ g = spec.faultedTwiddles Z F ℓ g := by
  intro g₀ g hg
  have h₁ : F ℓ = {g₀} := (Finset.card_eq_one.mp (hF ℓ)).choose_spec
  unfold faultedTwiddles
  simp [h₁, hg]

/-- For `p` with `p < spec.L ℓ` at every layer, the unfaulted and
    faulted NTT prefixes agree on `basis p`. -/
theorem runPrefix_eq_under_fault
    [KyberLike spec] {Z : spec.Twiddles K} {F : spec.FaultSet}
    (hF : spec.OneFaultPerLayer F)
    (p : Fin spec.n) (hp_lt : ∀ k : Fin spec.N, p.val < spec.L k)
    (k : ℕ) (hk : k ≤ spec.N) :
    spec.runPrefix Z k hk (spec.basis p) =
      spec.runPrefix (spec.faultedTwiddles Z F) k hk (spec.basis p) := by
  induction k with
  | zero =>
    simp [runPrefix]
  | succ k ih =>
    rw [spec.runPrefix_succ Z k hk, spec.runPrefix_succ _ k hk]
    simp only [LinearMap.comp_apply]
    rw [ih (Nat.le_of_succ_le hk)]
    -- Now both sides have the same running input from the faulted prefix,
    -- and we need: layer Z ⟨k,·⟩ v = layer (faultedTwiddles Z F) ⟨k,·⟩ v.
    set v := spec.runPrefix (spec.faultedTwiddles Z F) k _ (spec.basis p)
    -- Apply layer_agree_of_b_zero. The layer indices are ⟨k, hk⟩.
    -- We need: twiddles agree off some g₀, and v's b-side at g₀ is zero.
    have ℓ : Fin spec.N := ⟨k, hk⟩
    obtain ⟨g₀, hg₀⟩ := Finset.card_eq_one.mp (hF ⟨k, hk⟩)
    refine spec.layer_agree_of_b_zero (ℓ := ⟨k, hk⟩) (g₀ := g₀) ?_ ?_
    · intro g hg
      unfold faultedTwiddles
      simp [hg₀, hg]
    · -- v's b-side at g₀ is zero. v = runPrefix (faultedTwiddles) k (basis p).
      -- Use runPrefix_basis_bSide_zero on the faulted twiddles.
      intro j
      exact spec.runPrefix_basis_bSide_zero (spec.faultedTwiddles Z F) p
        ⟨k, hk⟩ (hp_lt ⟨k, hk⟩) g₀ j

/-! ### Step 4 — basis vectors in kernel -/

/-- `basis p ∈ ker (faultDiff Z F)` whenever `p.val < spec.L ℓ` for
    every layer ℓ. -/
theorem basis_mem_ker
    [KyberLike spec] {Z : spec.Twiddles K} {F : spec.FaultSet}
    (hF : spec.OneFaultPerLayer F)
    (p : Fin spec.n) (hp_lt : ∀ k : Fin spec.N, p.val < spec.L k) :
    spec.basis p ∈ LinearMap.ker (spec.faultDiff Z F) := by
  rw [LinearMap.mem_ker]
  show (spec.faultDiff Z F) (spec.basis p) = 0
  unfold faultDiff nttFault
  rw [LinearMap.sub_apply, sub_eq_zero]
  rw [spec.ntt_eq_runPrefix_last Z, spec.ntt_eq_runPrefix_last (spec.faultedTwiddles Z F)]
  exact spec.runPrefix_eq_under_fault hF p hp_lt spec.N le_rfl

end LayerSpec

/-! ### Theorem 2 — (⊇) direction packaged for Kyber -/

/-- The (⊇) direction of Theorem 2: `span{e₀, e₁} ⊆ ker (faultDiff Z F)`
    for the Kyber spec, modulo `runPrefix_basis_support_invariant`. -/
theorem theorem2_kyber_supset
    {Z : kyberSpec.Twiddles K}
    {F : kyberSpec.FaultSet}
    (hF : kyberSpec.OneFaultPerLayer F) :
    Submodule.span K
        ({kyberSpec.basis ⟨0, by decide⟩, kyberSpec.basis ⟨1, by decide⟩} :
          Set (Fin kyberSpec.n → K))
      ≤ LinearMap.ker (kyberSpec.faultDiff Z F) := by
  rw [Submodule.span_le]
  intro v hv
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hv
  rcases hv with rfl | rfl
  · apply kyberSpec.basis_mem_ker hF ⟨0, by decide⟩
    intro k
    show 0 < kyberSpec.L k
    fin_cases k <;> decide
  · apply kyberSpec.basis_mem_ker hF ⟨1, by decide⟩
    intro k
    show 1 < kyberSpec.L k
    fin_cases k <;> decide

end NttFaultRank


