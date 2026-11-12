import TxGraffitiC1FC.Invariants

namespace SimpleGraph

theorem residueAux_cons_ne_zero (d : ℕ) (rest : List ℕ) (hd : d ≠ 0) :
    residueAux (d :: rest) = residueAux (havelHakimiStep (d :: rest)) := by
  rw [residueAux]
  exact hd

/-- **Sum after one Havel–Hakimi step.** `havelHakimiStep` re-sorts (a permutation, so
sum-invariant *independent of the comparator*), hence its sum is just the sum of the
decremented head-block plus the untouched remainder: `(havelHakimiStep (d :: rest)).sum =
((rest.take d).map (·-1)).sum + (rest.drop d).sum`. Process-arithmetic brick. NB the total
drops by `2 · (#positive entries among the top-`d` block)`, i.e. by `2d` only when `d ≥ 1`
and those entries are all positive; at `d = 0` the sum is unchanged. (Termination of the
`residueAux` recursion is carried separately by the length measure, not by this sum.) -/
theorem sum_havelHakimiStep (d : ℕ) (rest : List ℕ) :
    (havelHakimiStep (d :: rest)).sum
      = ((rest.take d).map (· - 1)).sum + (rest.drop d).sum := by
  simp only [havelHakimiStep, List.splitAt_eq]
  rw [(List.mergeSort_perm _ _).sum_eq, List.sum_append]

end SimpleGraph
