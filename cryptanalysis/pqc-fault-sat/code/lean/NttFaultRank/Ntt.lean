/-
Phase C, items 1/3/4/5 — `LayerSpec`, NTT composition, faulted NTT.

A `LayerSpec` packages the per-layer Cooley-Tukey parameters: number of
layers `N`, ambient size `n`, per-layer group count `G ℓ` and butterfly
length `L ℓ`, plus the constraint `G ℓ * (2 * L ℓ) = n`.

`ntt` composes the `N` layers (with per-layer twiddle vectors). The
faulted variant `nttFault` is the same composition with twiddles in a
designated fault set zeroed out.

Concrete Kyber parameters (`n = 256`, `N = 7`, `G ℓ = 2^ℓ`,
`L ℓ = 128 / 2^ℓ`) are packaged as `kyberSpec` for use at theorem
sites in subsequent files.
-/

import NttFaultRank.LayerOn

namespace NttFaultRank

variable {K : Type*} [Field K]

/-- Cooley-Tukey layer parameters: `N` layers, ambient size `n`,
    per-layer group count `G ℓ` and butterfly length `L ℓ` satisfying
    `G ℓ * (2 * L ℓ) = n`. -/
structure LayerSpec where
  /-- Number of layers. -/
  N : ℕ
  /-- Ambient vector size. -/
  n : ℕ
  /-- Per-layer group count. -/
  G : Fin N → ℕ
  /-- Per-layer butterfly length. -/
  L : Fin N → ℕ
  /-- Per-layer dimension constraint. -/
  hGL : ∀ ℓ, G ℓ * (2 * L ℓ) = n

namespace LayerSpec
variable (spec : LayerSpec)

/-- A twiddle assignment: a vector of length `spec.G ℓ` for each layer. -/
abbrev Twiddles (K : Type*) : Type _ := ∀ ℓ : Fin spec.N, Fin (spec.G ℓ) → K

/-- One layer of the (unfaulted) NTT. -/
def layer (Z : spec.Twiddles K) (ℓ : Fin spec.N) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  layerOn spec.n (spec.G ℓ) (spec.L ℓ) (spec.hGL ℓ) (Z ℓ)

/-- The full Cooley-Tukey NTT: compose all `N` layers, layer 0 first.
    `ntt Z v = layer (N-1) (... (layer 1 (layer 0 v)))`. -/
def ntt (Z : spec.Twiddles K) : (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  Fin.foldl spec.N (fun acc ℓ => (spec.layer Z ℓ).comp acc) LinearMap.id

/-- Per-layer fault set: which group indices to zero out in each layer. -/
abbrev FaultSet : Type _ := ∀ ℓ : Fin spec.N, Finset (Fin (spec.G ℓ))

/-- The faulted twiddle assignment derived from `Z` and a fault set. -/
def faultedTwiddles (Z : spec.Twiddles K) (F : spec.FaultSet) : spec.Twiddles K :=
  fun ℓ g => if g ∈ F ℓ then 0 else Z ℓ g

/-- The faulted NTT: compose all layers using the faulted twiddles. -/
def nttFault (Z : spec.Twiddles K) (F : spec.FaultSet) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.ntt (spec.faultedTwiddles Z F)

/-! ### Generalized fault model: arbitrary replacement twiddles -/

/-- Generalized faulted twiddle assignment: faulted groups get replacement
    twiddles `Z'` instead of zero. -/
def faultedTwiddlesGen (Z Z' : spec.Twiddles K) (F : spec.FaultSet) : spec.Twiddles K :=
  fun ℓ g => if g ∈ F ℓ then Z' ℓ g else Z ℓ g

/-- Specialization: zeroing replacement twiddles recovers `faultedTwiddles`. -/
@[simp]
lemma faultedTwiddlesGen_zero (Z : spec.Twiddles K) (F : spec.FaultSet) :
    spec.faultedTwiddlesGen Z (fun _ _ => 0) F = spec.faultedTwiddles Z F := by
  ext ℓ g
  unfold faultedTwiddlesGen faultedTwiddles
  simp

/-- Generalized faulted NTT with arbitrary replacement twiddles. -/
def nttFaultGen (Z Z' : spec.Twiddles K) (F : spec.FaultSet) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.ntt (spec.faultedTwiddlesGen Z Z' F)

/-- Specialization: zeroing replacement twiddles recovers `nttFault`. -/
@[simp]
lemma nttFaultGen_zero (Z : spec.Twiddles K) (F : spec.FaultSet) :
    spec.nttFaultGen Z (fun _ _ => 0) F = spec.nttFault Z F := by
  unfold nttFaultGen nttFault
  rw [faultedTwiddlesGen_zero]

/-- Generalized fault-difference operator with arbitrary replacement twiddles. -/
def faultDiffGen (Z Z' : spec.Twiddles K) (F : spec.FaultSet) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.ntt Z - spec.nttFaultGen Z Z' F

end LayerSpec

/-! ### Kyber parameters: `n = 256`, `N = 7`, `G ℓ = 2^ℓ`, `L ℓ = 128 / 2^ℓ` -/

/-- Group count at layer ℓ for Kyber. -/
def kyberG (ℓ : Fin 7) : ℕ := 2 ^ ℓ.val

/-- Butterfly length at layer ℓ for Kyber. -/
def kyberL (ℓ : Fin 7) : ℕ := 128 / 2 ^ ℓ.val

/-- The Kyber layer specification. -/
def kyberSpec : LayerSpec where
  N := 7
  n := 256
  G := kyberG
  L := kyberL
  hGL ℓ := by
    fin_cases ℓ <;> decide

/-- The Cooley-Tukey recurrence: at every successive layer, butterfly
    length halves and group count doubles. Required for the (⊇) direction
    of Theorem 2. -/
class CooleyTukeyLike (spec : LayerSpec) : Prop where
  /-- `L` halves between consecutive layers. -/
  L_succ : ∀ k (h : k + 1 < spec.N),
    spec.L ⟨k + 1, h⟩ * 2 = spec.L ⟨k, Nat.lt_of_succ_lt h⟩
  /-- `G` doubles between consecutive layers. -/
  G_succ : ∀ k (h : k + 1 < spec.N),
    spec.G ⟨k + 1, h⟩ = 2 * spec.G ⟨k, Nat.lt_of_succ_lt h⟩
  /-- `2 L_0 = n` (the first layer covers the whole vector with one group of size n). -/
  L_zero_doubled : ∀ (h : 0 < spec.N), 2 * spec.L ⟨0, h⟩ = spec.n

/-- `kyberSpec` satisfies the Cooley-Tukey recurrence. -/
instance : CooleyTukeyLike kyberSpec where
  L_succ k h := by
    show kyberL ⟨k + 1, h⟩ * 2 = kyberL ⟨k, Nat.lt_of_succ_lt h⟩
    unfold kyberL
    show 128 / 2 ^ (k + 1) * 2 = 128 / 2 ^ k
    have hk : k < 6 := by
      have : k + 1 < 7 := h
      omega
    interval_cases k <;> decide
  G_succ k h := by
    show kyberG ⟨k + 1, h⟩ = 2 * kyberG ⟨k, Nat.lt_of_succ_lt h⟩
    unfold kyberG
    show 2 ^ (k + 1) = 2 * 2 ^ k
    ring
  L_zero_doubled h := by
    show 2 * kyberL ⟨0, h⟩ = 256
    unfold kyberL
    show 2 * (128 / 2 ^ (0 : Fin 7).val) = 256
    norm_num

/-! ### ML-DSA parameters: `n = 256`, `N = 8`, complete NTT with `L_{m-1} = 1` -/

/-- Group count at layer ℓ for ML-DSA (complete NTT). -/
def dsaG (ℓ : Fin 8) : ℕ := 2 ^ ℓ.val

/-- Butterfly length at layer ℓ for ML-DSA (complete NTT). -/
def dsaL (ℓ : Fin 8) : ℕ := 128 / 2 ^ ℓ.val

/-- The ML-DSA layer specification: 8 layers, innermost has L = 1. -/
def dsaSpec : LayerSpec where
  N := 8
  n := 256
  G := dsaG
  L := dsaL
  hGL ℓ := by fin_cases ℓ <;> decide

/-- `dsaSpec` satisfies the Cooley-Tukey recurrence (same shape as Kyber,
    one extra layer at the innermost length `dsaL ⟨7,_⟩ = 1`). -/
instance : CooleyTukeyLike dsaSpec where
  L_succ k h := by
    show dsaL ⟨k + 1, h⟩ * 2 = dsaL ⟨k, Nat.lt_of_succ_lt h⟩
    unfold dsaL
    show 128 / 2 ^ (k + 1) * 2 = 128 / 2 ^ k
    have hk : k < 7 := by
      have : k + 1 < 8 := h
      omega
    interval_cases k <;> decide
  G_succ k h := by
    show dsaG ⟨k + 1, h⟩ = 2 * dsaG ⟨k, Nat.lt_of_succ_lt h⟩
    unfold dsaG
    show 2 ^ (k + 1) = 2 * 2 ^ k
    ring
  L_zero_doubled h := by
    show 2 * dsaL ⟨0, h⟩ = 256
    unfold dsaL
    show 2 * (128 / 2 ^ (0 : Fin 8).val) = 256
    norm_num

end NttFaultRank
