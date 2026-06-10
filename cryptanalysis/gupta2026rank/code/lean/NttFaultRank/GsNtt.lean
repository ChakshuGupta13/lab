/-
Gentleman-Sande NTT composition, fault model, telescope identity,
and hybrid split infrastructure.
-/

import NttFaultRank.GsLayer
import NttFaultRank.Ntt

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

def gsLayer (Z : spec.Twiddles K) (ℓ : Fin spec.N) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  gsLayerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z ℓ)

-- gsNtt as Fin.foldl. Builds gsLayer 0 ∘ gsLayer 1 ∘ ... ∘ gsLayer (N-1).
-- Applied to v: gsLayer 0 (gsLayer 1 (... (gsLayer (N-1) v))).
def gsNtt (Z : spec.Twiddles K) : (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  Fin.foldl spec.N (fun acc ℓ => acc.comp (spec.gsLayer Z ℓ)) LinearMap.id

/-! ### gsPartial: first m layers of gsNtt -/

-- gsPartial Z m applies layers 0..m-1 (layer m-1 innermost/first, layer 0 outermost/last).
def gsPartial (Z : spec.Twiddles K) (m : ℕ) (hm : m ≤ spec.N) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  Fin.foldl m
    (fun acc (ℓ : Fin m) => acc.comp (spec.gsLayer Z (ℓ.castLE hm)))
    LinearMap.id

@[simp] lemma gsPartial_zero (Z : spec.Twiddles K) (hm : 0 ≤ spec.N) :
    spec.gsPartial Z 0 hm = LinearMap.id := by
  unfold gsPartial; simp [Fin.foldl_zero]

lemma gsPartial_succ (Z : spec.Twiddles K) (m : ℕ) (hm : m + 1 ≤ spec.N) :
    spec.gsPartial Z (m + 1) hm =
      (spec.gsPartial Z m (Nat.le_of_succ_le hm)).comp
        (spec.gsLayer Z ⟨m, hm⟩) := by
  unfold gsPartial; rw [Fin.foldl_succ_last]; rfl

lemma gsNtt_eq_gsPartial (Z : spec.Twiddles K) :
    spec.gsNtt Z = spec.gsPartial Z spec.N le_rfl := by rfl

def gsNttFaultGen (Z Z' : spec.Twiddles K) (F : spec.FaultSet) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.gsNtt (spec.faultedTwiddlesGen Z Z' F)

def gsFaultDiffGen (Z Z' : spec.Twiddles K) (F : spec.FaultSet) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.gsNtt Z - spec.gsNttFaultGen Z Z' F

/-! ### Hybrid and telescope -/

def gsHybrid (Z Z' : spec.Twiddles K) (k : ℕ) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  Fin.foldl spec.N
    (fun acc ℓ => acc.comp (spec.gsLayer (if ℓ.val < k then Z' else Z) ℓ))
    LinearMap.id

lemma gsHybrid_zero (Z Z' : spec.Twiddles K) :
    spec.gsHybrid Z Z' 0 = spec.gsNtt Z := by
  unfold gsHybrid gsNtt; congr 1 with acc ℓ

lemma gsHybrid_ge (Z Z' : spec.Twiddles K) {k : ℕ} (hk : spec.N ≤ k) :
    spec.gsHybrid Z Z' k = spec.gsNtt Z' := by
  unfold gsHybrid gsNtt; congr 1 with acc ℓ
  simp [show ℓ.val < k from lt_of_lt_of_le ℓ.isLt hk]

def gsTeleTerm (Z Z' : spec.Twiddles K) (k : ℕ) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.gsHybrid Z Z' k - spec.gsHybrid Z Z' (k + 1)

theorem gs_diff_telescope_gen (Z Z' : spec.Twiddles K) (F : spec.FaultSet) :
    spec.gsNtt Z - spec.gsNtt (spec.faultedTwiddlesGen Z Z' F) =
      ∑ k ∈ Finset.range spec.N,
        spec.gsTeleTerm Z (spec.faultedTwiddlesGen Z Z' F) k := by
  set W := spec.faultedTwiddlesGen Z Z' F
  simp only [gsTeleTerm]
  rw [Finset.sum_range_sub' (fun k => spec.gsHybrid Z W k) spec.N,
      spec.gsHybrid_zero Z W, spec.gsHybrid_ge Z W (le_refl spec.N)]

theorem gsFaultDiffGen_eq_telescope (Z Z' : spec.Twiddles K) (F : spec.FaultSet) :
    spec.gsFaultDiffGen Z Z' F =
      ∑ k ∈ Finset.range spec.N,
        spec.gsTeleTerm Z (spec.faultedTwiddlesGen Z Z' F) k := by
  unfold gsFaultDiffGen gsNttFaultGen
  exact spec.gs_diff_telescope_gen Z Z' F

/-! ### Helper: Fin.foldl pointwise agreement -/

-- Helper: strong agreement (functional equality of step functions) is sufficient.
private lemma foldl_agree_of_step_agree {α : Type*} {n : ℕ}
    {f g : α → Fin n → α} {init : α}
    (h : ∀ (k : Fin n) (a : α), f a k = g a k) :
    Fin.foldl n f init = Fin.foldl n g init := by
  induction n with
  | zero => simp [Fin.foldl_zero]
  | succ n ih =>
    simp only [Fin.foldl_succ_last]
    have ih' := ih (fun k a => h k.castSucc a)
    rw [ih']; congr 1; exact h (Fin.last n) _

/-! ### Hybrid pointwise agreement -/

-- gsHybrid k v = gsHybrid (k+1) v when gsLayer Z k and gsLayer Z' k
-- agree as FUNCTIONS (i.e. on all w).
theorem gsHybrid_agree_at_k_strong
    (Z Z' : spec.Twiddles K) (k : ℕ) (hk : k < spec.N)
    (v : Fin spec.n → K)
    (h_agree : ∀ w : Fin spec.n → K,
      spec.gsLayer Z ⟨k, hk⟩ w = spec.gsLayer Z' ⟨k, hk⟩ w) :
    spec.gsHybrid Z Z' k v = spec.gsHybrid Z Z' (k + 1) v := by
  have h_eq : spec.gsHybrid Z Z' k = spec.gsHybrid Z Z' (k + 1) := by
    unfold gsHybrid
    apply foldl_agree_of_step_agree
    intro ℓ a
    by_cases hℓ : ℓ.val = k
    · have hℓk : ℓ = ⟨k, hk⟩ := Fin.ext hℓ
      subst hℓk
      simp only [show ¬(k < k) from lt_irrefl k, ↓reduceIte,
                 show k < k + 1 from Nat.lt_succ_self k]
      congr 1
      apply LinearMap.ext; intro w; exact h_agree w
    · congr 1
      by_cases hlt : ℓ.val < k
      · have : ℓ.val < k + 1 := by omega
        simp [hlt, this]
      · have : ¬(ℓ.val < k + 1) := by omega
        simp [hlt, this]
  rw [h_eq]

/-! ### Peeling lemma: gsHybrid via inner-layer peel -/

lemma gsHybrid_apply_succ_last (Z Z' : spec.Twiddles K) (k : ℕ) (N' : ℕ)
    (hN : N' + 1 ≤ spec.N) (v : Fin spec.n → K) :
    (Fin.foldl (N' + 1)
        (fun acc (ℓ : Fin (N' + 1)) =>
          acc.comp (spec.gsLayer
            (if (ℓ.castLE hN).val < k then Z' else Z) (ℓ.castLE hN)))
        LinearMap.id) v
    = (Fin.foldl N'
        (fun acc (ℓ : Fin N') =>
          acc.comp (spec.gsLayer
            (if (ℓ.castLE (le_of_lt hN)).val < k then Z' else Z)
            (ℓ.castLE (le_of_lt hN))))
        LinearMap.id)
      (spec.gsLayer
        (if N' < k then Z' else Z) ⟨N', hN⟩ v) := by
  rw [Fin.foldl_succ_last]
  simp only [LinearMap.comp_apply]
  rfl

end LayerSpec

end NttFaultRank
