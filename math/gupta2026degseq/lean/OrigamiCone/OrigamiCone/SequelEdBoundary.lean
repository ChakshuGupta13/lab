import Mathlib
import OrigamiCone.SequelEdActiveCol

/-!
# Sequel: boundary columns are active (Task E.δ.f — paper `lem:boundary`)

Paper `lem:boundary` (§8): in every height function on the `m × n` grid, the
first and last columns each carry a strict local extremum, hence are active
(`activeColumn`).  This discharges the boundary-activity hypotheses of
`SequelEdActiveCol.numFrozenRuns_le`.

## Proof structure

1. **Abstract ±1-walk lemma** (`boundary_walk_abstract`): over two sequences
   `a b : ℕ → ℤ` (a boundary column and its neighbour), both ±1 walks with
   `η_i := b i - a i = ±1`, some row is a strict local extremum.  The core
   observation (cleaner than the paper's two-part argument): a single forward
   induction shows the second-column walk `b(i+1) - b i = ε_i + (η_{i+1} -
   η_i)` *forces* `η` constant once `ε_i = -η_0`, and non-extremality
   propagates `ε_i = -η_0` down the column; the last row is then forced to be
   an extremum.
2. **Grid neighbour enumeration** (`adj_col0_cases`, `adj_colLast_cases`):
   any grid-neighbour of a boundary cell is one of the two vertical
   neighbours or the single horizontal neighbour.
3. **Instantiation** (`firstColumn_active`, `lastColumn_active`): set
   `a i = h(i, boundary)`, `b i = h(i, neighbour)`, verify the walk
   hypotheses from `IsHeight`, and convert the abstract extremum to a grid
   `IsStrictLocalExtremum`.
4. **Combined** (`boundary_columns_active`): both boundary columns active,
   the exact shape `numFrozenRuns_le` consumes.

## Substrate

Imports only `OrigamiCone.SequelEdActiveCol` (for `activeColumn`, `Cell`,
`adj`, `gdist`, `IsHeight`, `IsStrictLocalMax/Min`).  The abstract walk lemma
uses no grid structure.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
Check with `#print axioms OrigamiCone.Sequel.boundary_columns_active`.
-/

namespace OrigamiCone.Sequel

open OrigamiCone

/-! ## Abstract ±1-walk lemma -/

/-- Row `i` is a strict local MAX in the two-column strip `a` (boundary) / `b`
(neighbour): every present neighbour (`i-1` if `i>0`, `i+1` if `i<M`,
horizontal `b i`) is lower by 1. -/
def isMaxRow (M : ℕ) (a b : ℕ → ℤ) (i : ℕ) : Prop :=
  (0 < i → a (i-1) = a i - 1) ∧ (i < M → a (i+1) = a i - 1) ∧ b i = a i - 1

def isMinRow (M : ℕ) (a b : ℕ → ℤ) (i : ℕ) : Prop :=
  (0 < i → a (i-1) = a i + 1) ∧ (i < M → a (i+1) = a i + 1) ∧ b i = a i + 1

/-- **Abstract boundary-walk lemma.**  Two ±1 walks `a`, `b` on `0..M` with
`η_i := b i - a i = ±1` have a row that is a strict local extremum. -/
theorem boundary_walk_abstract (M : ℕ) (a b : ℕ → ℤ)
    (hA : ∀ i, i < M → a (i+1) - a i = 1 ∨ a (i+1) - a i = -1)
    (hη : ∀ i, i ≤ M → b i - a i = 1 ∨ b i - a i = -1)
    (hB : ∀ i, i < M → b (i+1) - b i = 1 ∨ b (i+1) - b i = -1) :
    ∃ i, i ≤ M ∧ (isMaxRow M a b i ∨ isMinRow M a b i) := by
  by_contra hcon
  have hne : ∀ i, i ≤ M → ¬ isMaxRow M a b i ∧ ¬ isMinRow M a b i :=
    fun i hi => ⟨fun hm => hcon ⟨i, hi, Or.inl hm⟩, fun hm => hcon ⟨i, hi, Or.inr hm⟩⟩
  rcases Nat.eq_zero_or_pos M with hM0 | hMpos
  · subst hM0
    rcases hη 0 (le_refl 0) with hh | hh
    · exact (hne 0 (le_refl 0)).2 ⟨fun h => absurd h (by omega), fun h => absurd h (by omega), by omega⟩
    · exact (hne 0 (le_refl 0)).1 ⟨fun h => absurd h (by omega), fun h => absurd h (by omega), by omega⟩
  · set η0 : ℤ := b 0 - a 0 with hη0def
    have hη0v : η0 = 1 ∨ η0 = -1 := hη 0 (by omega)
    have hε0 : a (0+1) - a 0 = -η0 := by
      have h0 := hne 0 (by omega)
      rcases hA 0 hMpos with he | he <;> rcases hη0v with h1 | h1
      · exact absurd ⟨fun h => absurd h (by omega), fun _ => by omega, by omega⟩ h0.2
      · omega
      · omega
      · exact absurd ⟨fun h => absurd h (by omega), fun _ => by omega, by omega⟩ h0.1
    have key : ∀ i, i ≤ M → (b i - a i = η0) ∧ (i < M → a (i+1) - a i = -η0) := by
      intro i
      induction i with
      | zero => intro _; exact ⟨by omega, fun _ => hε0⟩
      | succ i IH =>
        intro hle
        have hiM : i < M := by omega
        obtain ⟨hηi, hεi_imp⟩ := IH (by omega)
        have hεi : a (i+1) - a i = -η0 := hεi_imp hiM
        have hη_i1 : b (i+1) - a (i+1) = η0 := by
          have hBv := hB i hiM
          have hηnext := hη (i+1) hle
          have hid : (b (i+1) - b i) = (b (i+1) - a (i+1)) - (b i - a i) + (a (i+1) - a i) := by ring
          rcases hη0v with h1 | h1 <;> rcases hBv with hb | hb <;>
            rcases hηnext with hn | hn <;> omega
        refine ⟨hη_i1, ?_⟩
        intro hi1M
        have h1 := hne (i+1) (by omega)
        have hidxL : i + 1 - 1 = i := rfl
        by_contra hcontra
        have hεnextv := hA (i+1) hi1M
        have hεnext : a (i+1+1) - a (i+1) = η0 := by
          rcases hεnextv with h | h <;> rcases hη0v with hz | hz <;> omega
        rcases hη0v with hz | hz
        · exact h1.2 ⟨fun _ => by rw [hidxL]; omega, fun _ => by omega, by omega⟩
        · exact h1.1 ⟨fun _ => by rw [hidxL]; omega, fun _ => by omega, by omega⟩
    have hηM : b M - a M = η0 := (key M (le_refl M)).1
    have hεMm : a M - a (M-1) = -η0 := by
      have h := (key (M-1) (by omega)).2 (by omega)
      have hMm : M - 1 + 1 = M := by omega
      rw [hMm] at h; exact h
    have hneM := hne M (le_refl M)
    rcases hη0v with hz | hz
    · exact hneM.2 ⟨fun _ => by omega, fun h => absurd h (by omega), by omega⟩
    · exact hneM.1 ⟨fun _ => by omega, fun h => absurd h (by omega), by omega⟩

/-! ## Grid neighbour enumeration -/

variable {m n : ℕ}

/-- Neighbours of a first-column cell `(i, 0)`: `(i+1, 0)`, `(i-1, 0)`, `(i, 1)`. -/
theorem adj_col0_cases (i : Fin m) (h0 : (0 : ℕ) < n) (u : Cell m n)
    (hu : adj ((i, ⟨0, h0⟩) : Cell m n) u) :
    (∃ (hp : i.val + 1 < m), u = (⟨i.val + 1, hp⟩, ⟨0, h0⟩))
    ∨ (∃ (hp : 0 < i.val), u = (⟨i.val - 1, by omega⟩, ⟨0, h0⟩))
    ∨ (∃ (hq : 1 < n), u = (i, ⟨1, hq⟩)) := by
  obtain ⟨⟨p, hp⟩, ⟨q, hq⟩⟩ := u
  unfold adj gdist at hu
  have hu' : ((i.val : ℤ) - p).natAbs + ((0 : ℤ) - q).natAbs = 1 := by exact_mod_cast hu
  by_cases hqeq : q = 0
  · subst hqeq
    have hn1 : ((i.val : ℤ) - p).natAbs = 1 := by simpa using hu'
    rcases (Int.natAbs_eq_iff).mp hn1 with hpi | hpi
    · right; left; exact ⟨by omega, by ext <;> simp <;> omega⟩
    · left; exact ⟨by omega, by ext <;> simp <;> omega⟩
  · right; right
    have h0q : ((0 : ℤ) - q).natAbs = q := by simp
    have hq1 : q = 1 := by omega
    have hz : ((i.val : ℤ) - p).natAbs = 0 := by omega
    have hpi : p = i.val := by have := Int.natAbs_eq_zero.mp hz; omega
    subst hq1
    exact ⟨hq, by ext <;> simp <;> omega⟩

/-- Neighbours of a last-column cell `(i, n-1)`: `(i+1, n-1)`, `(i-1, n-1)`,
`(i, n-2)`. -/
theorem adj_colLast_cases (i : Fin m) (hn : 1 < n) (u : Cell m n)
    (hu : adj ((i, ⟨n-1, by omega⟩) : Cell m n) u) :
    (∃ (hp : i.val + 1 < m), u = (⟨i.val + 1, hp⟩, ⟨n-1, by omega⟩))
    ∨ (∃ (hp : 0 < i.val), u = (⟨i.val - 1, by omega⟩, ⟨n-1, by omega⟩))
    ∨ (u = (i, ⟨n-2, by omega⟩)) := by
  obtain ⟨⟨p, hp⟩, ⟨q, hq⟩⟩ := u
  unfold adj gdist at hu
  have hu' : ((i.val : ℤ) - p).natAbs + (((n-1 : ℕ) : ℤ) - q).natAbs = 1 := by exact_mod_cast hu
  by_cases hqeq : q = n - 1
  · subst hqeq
    have hn1 : ((i.val : ℤ) - p).natAbs = 1 := by simpa using hu'
    rcases (Int.natAbs_eq_iff).mp hn1 with hpi | hpi
    · right; left; exact ⟨by omega, by ext <;> simp <;> omega⟩
    · left; exact ⟨by omega, by ext <;> simp <;> omega⟩
  · right; right
    have hzq : (((n-1 : ℕ) : ℤ) - q).natAbs = 1 → q = n - 2 ∨ q = n := by
      intro h
      rcases (Int.natAbs_eq_iff).mp h with hh | hh
      · left; omega
      · right; omega
    have hz : ((i.val : ℤ) - p).natAbs = 0 ∧ (((n-1 : ℕ) : ℤ) - q).natAbs = 1 := by
      constructor
      · by_contra hc
        have : ((i.val : ℤ) - p).natAbs ≥ 1 := by omega
        have hqn : (((n-1 : ℕ) : ℤ) - q).natAbs = 0 := by omega
        have := Int.natAbs_eq_zero.mp hqn
        omega
      · omega
    have hpi : p = i.val := by have := Int.natAbs_eq_zero.mp hz.1; omega
    have hqv : q = n - 2 := by
      rcases hzq hz.2 with h | h
      · exact h
      · omega
    exact Prod.ext (Fin.ext hpi) (Fin.ext hqv)

/-! ## Instantiation on the grid -/

/-- **First column is active** (paper `lem:boundary`).  Some cell of column 0
is a strict local extremum. -/
theorem firstColumn_active (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    activeColumn h (⟨0, by omega⟩ : Fin n) := by
  set a : ℕ → ℤ := fun k => h (⟨min k (m-1), by omega⟩, ⟨0, by omega⟩) with hadef
  set b : ℕ → ℤ := fun k => h (⟨min k (m-1), by omega⟩, ⟨1, hn⟩) with hbdef
  have hak : ∀ k (hk : k < m), a k = h (⟨k, hk⟩, ⟨0, by omega⟩) := by
    intro k hk; have hmin : min k (m-1) = k := by omega
    simp only [hadef, hmin]
  have hbk : ∀ k (hk : k < m), b k = h (⟨k, hk⟩, ⟨1, hn⟩) := by
    intro k hk; have hmin : min k (m-1) = k := by omega
    simp only [hbdef, hmin]
  have hA : ∀ i, i < m - 1 → a (i+1) - a i = 1 ∨ a (i+1) - a i = -1 := by
    intro i hi
    rw [hak i (by omega), hak (i+1) (by omega)]
    have hadj : adj ((⟨i, by omega⟩, ⟨0, by omega⟩) : Cell m n) (⟨i+1, by omega⟩, ⟨0, by omega⟩) := by
      unfold adj gdist; simp only [Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i, by omega⟩, ⟨0, by omega⟩) - h (⟨i+1, by omega⟩, ⟨0, by omega⟩)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hη : ∀ i, i ≤ m - 1 → b i - a i = 1 ∨ b i - a i = -1 := by
    intro i hi
    rw [hak i (by omega), hbk i (by omega)]
    have hadj : adj ((⟨i, by omega⟩, ⟨0, by omega⟩) : Cell m n) (⟨i, by omega⟩, ⟨1, hn⟩) := by
      unfold adj gdist; simp only [Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i, by omega⟩, ⟨0, by omega⟩) - h (⟨i, by omega⟩, ⟨1, hn⟩)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hB : ∀ i, i < m - 1 → b (i+1) - b i = 1 ∨ b (i+1) - b i = -1 := by
    intro i hi
    rw [hbk i (by omega), hbk (i+1) (by omega)]
    have hadj : adj ((⟨i, by omega⟩, ⟨1, hn⟩) : Cell m n) (⟨i+1, by omega⟩, ⟨1, hn⟩) := by
      unfold adj gdist; simp only [Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i, by omega⟩, ⟨1, hn⟩) - h (⟨i+1, by omega⟩, ⟨1, hn⟩)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  obtain ⟨i, hiM, hext⟩ := boundary_walk_abstract (m-1) a b hA hη hB
  have him : i < m := by omega
  refine ⟨⟨i, him⟩, ?_⟩
  have hival : (⟨i, him⟩ : Fin m).val = i := rfl
  rcases hext with hmax | hmin
  · left
    intro u hu
    rcases adj_col0_cases ⟨i, him⟩ (by omega) u hu with ⟨hp, rfl⟩ | ⟨hp, rfl⟩ | ⟨hq, rfl⟩
    · have hc := hmax.2.1 (by omega); rw [hak (i+1) (by omega), hak i him] at hc; simpa using hc
    · have hc := hmax.1 (by omega); rw [hak (i-1) (by omega), hak i him] at hc; simpa using hc
    · have hc := hmax.2.2; rw [hbk i him, hak i him] at hc; simpa using hc
  · right
    intro u hu
    rcases adj_col0_cases ⟨i, him⟩ (by omega) u hu with ⟨hp, rfl⟩ | ⟨hp, rfl⟩ | ⟨hq, rfl⟩
    · have hc := hmin.2.1 (by omega); rw [hak (i+1) (by omega), hak i him] at hc; simpa using hc
    · have hc := hmin.1 (by omega); rw [hak (i-1) (by omega), hak i him] at hc; simpa using hc
    · have hc := hmin.2.2; rw [hbk i him, hak i him] at hc; simpa using hc

/-- **Last column is active** (paper `lem:boundary`).  Some cell of column
`n-1` is a strict local extremum. -/
theorem lastColumn_active (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    activeColumn h (⟨n-1, by omega⟩ : Fin n) := by
  set a : ℕ → ℤ := fun k => h (⟨min k (m-1), by omega⟩, ⟨n-1, by omega⟩) with hadef
  set b : ℕ → ℤ := fun k => h (⟨min k (m-1), by omega⟩, ⟨n-2, by omega⟩) with hbdef
  have hak : ∀ k (hk : k < m), a k = h (⟨k, hk⟩, ⟨n-1, by omega⟩) := by
    intro k hk; have hmin : min k (m-1) = k := by omega
    simp only [hadef, hmin]
  have hbk : ∀ k (hk : k < m), b k = h (⟨k, hk⟩, ⟨n-2, by omega⟩) := by
    intro k hk; have hmin : min k (m-1) = k := by omega
    simp only [hbdef, hmin]
  have hA : ∀ i, i < m - 1 → a (i+1) - a i = 1 ∨ a (i+1) - a i = -1 := by
    intro i hi
    rw [hak i (by omega), hak (i+1) (by omega)]
    have hadj : adj ((⟨i, by omega⟩, ⟨n-1, by omega⟩) : Cell m n) (⟨i+1, by omega⟩, ⟨n-1, by omega⟩) := by
      unfold adj gdist; simp only [Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i, by omega⟩, ⟨n-1, by omega⟩) - h (⟨i+1, by omega⟩, ⟨n-1, by omega⟩)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hη : ∀ i, i ≤ m - 1 → b i - a i = 1 ∨ b i - a i = -1 := by
    intro i hi
    rw [hak i (by omega), hbk i (by omega)]
    have hadj : adj ((⟨i, by omega⟩, ⟨n-1, by omega⟩) : Cell m n) (⟨i, by omega⟩, ⟨n-2, by omega⟩) := by
      unfold adj gdist; simp only [Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i, by omega⟩, ⟨n-1, by omega⟩) - h (⟨i, by omega⟩, ⟨n-2, by omega⟩)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  have hB : ∀ i, i < m - 1 → b (i+1) - b i = 1 ∨ b (i+1) - b i = -1 := by
    intro i hi
    rw [hbk i (by omega), hbk (i+1) (by omega)]
    have hadj : adj ((⟨i, by omega⟩, ⟨n-2, by omega⟩) : Cell m n) (⟨i+1, by omega⟩, ⟨n-2, by omega⟩) := by
      unfold adj gdist; simp only [Fin.val_mk]; omega
    have hv := hh _ _ hadj
    rcases abs_cases (h (⟨i, by omega⟩, ⟨n-2, by omega⟩) - h (⟨i+1, by omega⟩, ⟨n-2, by omega⟩)) with ⟨he, _⟩ | ⟨he, _⟩ <;> omega
  obtain ⟨i, hiM, hext⟩ := boundary_walk_abstract (m-1) a b hA hη hB
  have him : i < m := by omega
  refine ⟨⟨i, him⟩, ?_⟩
  have hival : (⟨i, him⟩ : Fin m).val = i := rfl
  rcases hext with hmax | hmin
  · left
    intro u hu
    rcases adj_colLast_cases ⟨i, him⟩ hn u hu with ⟨hp, rfl⟩ | ⟨hp, rfl⟩ | rfl
    · have hc := hmax.2.1 (by omega); rw [hak (i+1) (by omega), hak i him] at hc; simpa using hc
    · have hc := hmax.1 (by omega); rw [hak (i-1) (by omega), hak i him] at hc; simpa using hc
    · have hc := hmax.2.2; rw [hbk i him, hak i him] at hc; simpa using hc
  · right
    intro u hu
    rcases adj_colLast_cases ⟨i, him⟩ hn u hu with ⟨hp, rfl⟩ | ⟨hp, rfl⟩ | rfl
    · have hc := hmin.2.1 (by omega); rw [hak (i+1) (by omega), hak i him] at hc; simpa using hc
    · have hc := hmin.1 (by omega); rw [hak (i-1) (by omega), hak i him] at hc; simpa using hc
    · have hc := hmin.2.2; rw [hbk i him, hak i him] at hc; simpa using hc

/-- **Both boundary columns are active** (paper `lem:boundary`, combined).
Exactly the shape `SequelEdActiveCol.numFrozenRuns_le` consumes. -/
theorem boundary_columns_active (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    activeColumn h (⟨0, by omega⟩ : Fin n)
      ∧ activeColumn h (⟨n-1, by omega⟩ : Fin n) :=
  ⟨firstColumn_active h hh hm hn, lastColumn_active h hh hm hn⟩

end OrigamiCone.Sequel
