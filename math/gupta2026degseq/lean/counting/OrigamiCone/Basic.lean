import Mathlib

namespace OrigamiCone

variable {m n : ℕ}

abbrev Cell (m n : ℕ) := Fin m × Fin n

def gdist (p q : Cell m n) : ℤ :=
  ((((p.1.val : ℤ) - q.1.val).natAbs + ((p.2.val : ℤ) - q.2.val).natAbs : ℕ) : ℤ)

lemma gdist_nonneg (p q : Cell m n) : 0 ≤ gdist p q := by unfold gdist; omega

lemma gdist_self (p : Cell m n) : gdist p p = 0 := by unfold gdist; omega

lemma gdist_comm (p q : Cell m n) : gdist p q = gdist q p := by unfold gdist; omega

lemma gdist_triangle (a b c : Cell m n) : gdist a c ≤ gdist a b + gdist b c := by
  unfold gdist; omega

lemma gdist_eq_zero {p q : Cell m n} : gdist p q = 0 ↔ p = q := by
  constructor
  · intro h
    unfold gdist at h
    have e1 : p.1.val = q.1.val := by omega
    have e2 : p.2.val = q.2.val := by omega
    exact Prod.ext_iff.2 ⟨Fin.ext e1, Fin.ext e2⟩
  · intro h; subst h; exact gdist_self p

lemma gdist_pos {p q : Cell m n} (hpq : p ≠ q) : 1 ≤ gdist p q := by
  have h0 := gdist_nonneg p q
  have hne : gdist p q ≠ 0 := fun h => hpq (gdist_eq_zero.1 h)
  omega

def adj (p q : Cell m n) : Prop := gdist p q = 1

def IsHeight (h : Cell m n → ℤ) : Prop := ∀ p q, adj p q → |h p - h q| = 1

def IsStrictLocalMax (h : Cell m n → ℤ) (q : Cell m n) : Prop :=
  ∀ u, adj q u → h u = h q - 1

def IsStrictLocalMin (h : Cell m n → ℤ) (p : Cell m n) : Prop :=
  ∀ u, adj p u → h u = h p + 1

lemma exists_step_toward {p q : Cell m n} (hpq : p ≠ q) :
    ∃ p', adj p p' ∧ gdist p' q = gdist p q - 1 := by
  have hcoord : p.1.val ≠ q.1.val ∨ p.2.val ≠ q.2.val := by
    by_contra h
    push_neg at h
    exact hpq (Prod.ext_iff.2 ⟨Fin.ext h.1, Fin.ext h.2⟩)
  rcases hcoord with hrow | hcol
  · rcases lt_or_gt_of_ne hrow with hlt | hgt
    ·
      have hb : p.1.val + 1 < m := by have := q.1.isLt; omega
      refine ⟨(⟨p.1.val + 1, hb⟩, p.2), ?_, ?_⟩
      · simp only [adj, gdist]; omega
      · simp only [gdist]; omega
    ·
      have hb : p.1.val - 1 < m := by have := p.1.isLt; omega
      refine ⟨(⟨p.1.val - 1, hb⟩, p.2), ?_, ?_⟩
      · simp only [adj, gdist]; omega
      · simp only [gdist]; omega
  · rcases lt_or_gt_of_ne hcol with hlt | hgt
    ·
      have hb : p.2.val + 1 < n := by have := q.2.isLt; omega
      refine ⟨(p.1, ⟨p.2.val + 1, hb⟩), ?_, ?_⟩
      · simp only [adj, gdist]; omega
      · simp only [gdist]; omega
    ·
      have hb : p.2.val - 1 < n := by have := p.2.isLt; omega
      refine ⟨(p.1, ⟨p.2.val - 1, hb⟩), ?_, ?_⟩
      · simp only [adj, gdist]; omega
      · simp only [gdist]; omega

lemma height_lipschitz {h : Cell m n → ℤ} (hh : IsHeight h) :
    ∀ (k : ℕ) (a b : Cell m n), gdist a b ≤ k → |h a - h b| ≤ gdist a b := by
  intro k
  induction k with
  | zero =>
    intro a b hk
    have hz : gdist a b = 0 := le_antisymm hk (gdist_nonneg a b)
    have hab : a = b := gdist_eq_zero.1 hz
    subst hab; simp [gdist_self]
  | succ k ih =>
    intro a b hk
    by_cases hab : a = b
    · subst hab; simp [gdist_self]
    · obtain ⟨a', ha', hd⟩ := exists_step_toward hab
      have hb' : gdist a' b ≤ k := by rw [hd]; omega
      have hih := ih a' b hb'
      have h1 : |h a - h a'| = 1 := hh a a' ha'
      have htri : |h a - h b| ≤ |h a - h a'| + |h a' - h b| := abs_sub_le _ _ _
      rw [hd] at hih
      linarith

lemma unique_max_global {h : Cell m n → ℤ} (hh : IsHeight h) {q : Cell m n}
    (huniq : ∀ q', IsStrictLocalMax h q' → q' = q) : ∀ v, h v ≤ h q := by
  haveI : Nonempty (Cell m n) := ⟨q⟩
  obtain ⟨w, hw⟩ := Finite.exists_max h
  have hwmax : IsStrictLocalMax h w := by
    intro u hu
    have hle := hw u
    have h1 := hh w u hu
    rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega
  have hwq : w = q := huniq w hwmax
  subst hwq
  exact hw

lemma cone_ascend {h : Cell m n → ℤ} (hh : IsHeight h) {q : Cell m n}
    (huniq : ∀ q', IsStrictLocalMax h q' → q' = q) (hglob : ∀ v, h v ≤ h q) :
    ∀ (k : ℕ) (v : Cell m n), (h q - h v).toNat ≤ k → gdist q v ≤ h q - h v := by
  intro k
  induction k with
  | zero =>
    intro v hk
    have heq : h v = h q := by
      have h2 := hglob v
      have hz : (h q - h v).toNat = 0 := Nat.le_zero.1 hk
      omega
    have hvmax : IsStrictLocalMax h v := by
      intro u hu
      have hle := hglob u
      have h1 := hh v u hu
      rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega
    have hvq : v = q := huniq v hvmax
    subst hvq
    simp [gdist_self]
  | succ k ih =>
    intro v hk
    by_cases hvq : v = q
    · subst hvq; simp [gdist_self]
    · have hvlt : h v < h q := by
        rcases lt_or_eq_of_le (hglob v) with hlt | heq
        · exact hlt
        · exfalso
          have hvmax : IsStrictLocalMax h v := by
            intro u hu
            have hle := hglob u
            have h1 := hh v u hu
            rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega
          exact hvq (huniq v hvmax)

      have hnotmax : ¬ ∀ u, adj v u → h u = h v - 1 :=
        fun hvmax => hvq (huniq v hvmax)
      push_neg at hnotmax
      obtain ⟨u, hadj, hune⟩ := hnotmax
      have h1 := hh v u hadj
      have hup : h u = h v + 1 := by
        rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2
        · omega
        · omega
      have hdefu : (h q - h u).toNat ≤ k := by
        have hglu := hglob u
        omega
      have hihu := ih u hdefu
      have htri := gdist_triangle q u v
      have huv : gdist u v = 1 := by rw [gdist_comm]; exact hadj
      have hstep : gdist q v ≤ (h q - h u) + 1 := by rw [huv] at htri; linarith
      linarith

theorem cone_max {h : Cell m n → ℤ} (hh : IsHeight h) {q : Cell m n}
    (huniq : ∀ q', IsStrictLocalMax h q' → q' = q) :
    ∀ v, h v = h q - gdist q v := by
  have hglob := unique_max_global hh huniq
  intro v
  have hle : gdist q v ≤ h q - h v :=
    cone_ascend hh huniq hglob (h q - h v).toNat v le_rfl
  have hlip : |h q - h v| ≤ gdist q v :=
    height_lipschitz hh (gdist q v).toNat q v
      (le_of_eq (Int.toNat_of_nonneg (gdist_nonneg q v)).symm)
  have hnn : 0 ≤ h q - h v := by have := hglob v; omega
  rw [abs_of_nonneg hnn] at hlip
  linarith

theorem cone_min {h : Cell m n → ℤ} (hh : IsHeight h) {p : Cell m n}
    (huniq : ∀ p', IsStrictLocalMin h p' → p' = p) :
    ∀ v, h v = h p + gdist p v := by
  have hh' : IsHeight (fun v => -h v) := by
    intro a b hab
    show |(-h a) - (-h b)| = 1
    rw [show (-h a) - (-h b) = -(h a - h b) by ring, abs_neg]
    exact hh a b hab
  have huniq' : ∀ q', IsStrictLocalMax (fun v => -h v) q' → q' = p := by
    intro q' hq'
    apply huniq
    intro u hu
    have hval : -h u = -h q' - 1 := hq' u hu
    show h u = h q' + 1
    linarith
  have hcone := cone_max hh' huniq'
  intro v
  have hv : -h v = -h p - gdist p v := hcone v
  linarith

end OrigamiCone
