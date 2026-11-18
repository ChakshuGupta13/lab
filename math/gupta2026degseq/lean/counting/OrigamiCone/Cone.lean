import OrigamiCone.Basic

namespace OrigamiCone

variable {m n : ℕ}

def cone (q b : Cell m n) : Cell m n → ℤ := fun v => gdist q b - gdist q v

lemma gdist_adj_step {q p p' : Cell m n} (h : adj p p') :
    gdist q p = gdist q p' + 1 ∨ gdist q p = gdist q p' - 1 := by
  unfold adj gdist at *
  omega

lemma cone_isHeight (q b : Cell m n) : IsHeight (cone q b) := by
  intro p p' hpp'
  rw [abs_eq (by norm_num : (0 : ℤ) ≤ 1)]
  simp only [cone]
  rcases gdist_adj_step (q := q) hpp' with h | h
  · right; omega
  · left; omega

lemma cone_max_at (q b : Cell m n) : IsStrictLocalMax (cone q b) q := by
  intro u hu
  have hu1 : gdist q u = 1 := hu
  have h0 : gdist q q = 0 := gdist_self q
  simp only [cone]
  omega

lemma cone_unique_max (q b : Cell m n) :
    ∀ q', IsStrictLocalMax (cone q b) q' → q' = q := by
  intro q' hq'
  by_contra hne
  obtain ⟨u, hadj, hd⟩ := exists_step_toward (show q' ≠ q from hne)
  have hval := hq' u hadj
  simp only [cone] at hval
  have hc1 : gdist q u = gdist u q := gdist_comm q u
  have hc2 : gdist q q' = gdist q' q := gdist_comm q q'
  omega

lemma cone_eq_apex_form (q b : Cell m n) (v : Cell m n) :
    cone q b v = cone q b q - gdist q v :=
  cone_max (cone_isHeight q b) (cone_unique_max q b) v

end OrigamiCone
