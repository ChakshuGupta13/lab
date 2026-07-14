# One construction for the Miura-ori flip-graph degree sequence

Verification code and Lean formalization for the paper *"One construction for
the Miura-ori flip-graph degree sequence"* (Chakshu Gupta, Georgia Institute of
Technology).

**Paper**: [arXiv:2607.05567](https://arxiv.org/abs/2607.05567) (math.CO, 2026).

This is the sequel to *"Height functions on the m × n Miura-ori flip graph:
degree sequence and diameter"* ([arXiv:2606.22614](https://arxiv.org/abs/2606.22614),
code at [`gupta2026origami`](../gupta2026origami)), which resolved the degree
sequence up to degree five. The present paper gives **one construction** that
computes the number of degree-`d` vertices for **every** degree `d`, uniformly.

## The problem

The **origami flip graph** `OFG(C)` of a flat-foldable crease pattern `C` has the
flat-foldable mountain–valley assignments of `C` as its vertices, with an edge
between two assignments that differ by a single **face flip**. For the `m × n`
Miura-ori `M_{m,n}`, the **degree sequence** of `OFG(M_{m,n})` — the number of
vertices of each degree — was previously known only for `m = 2` (all degrees) and
for degree `≤ 5` (all `m, n`), each by a separate argument.

## The method — envelope construction and lattice points

By the Ginepro–Hull bijection and the bipartite height-function lift,
`OFG(M_{m,n})` is identified with the integer height functions on the grid, where

> **the degree of a vertex equals the number of strict local extrema of its
> height function.**

The **envelope encoding** represents each height function by an integer
configuration, so counting the configurations with exactly `d` extrema is a
parametric lattice-point problem, piecewise quasi-polynomial in `(m, n)` by
Barvinok–Woods theory. On the high region `m, n ≥ max(d-1, 2)` this quasi-
polynomial collapses to a single symmetric polynomial `p_d(m, n)`.

Write `E_d(m, n)` for the number of degree-`d` vertices. Two independent engines
compute it: the **envelope split-counter** (`extract_pd.E_d`) and the **column
transfer-matrix DP** (`column_dp.Ed_DP`). They agree everywhere.

## Results

| Result | Statement | Status | Verifier |
|---|---|---|---|
| Envelope Structure Theorem | every height function is an envelope of distance cones at its minima; the encoding is Presburger | proved + verified | `envelope_structure.py` |
| Maxima Criterion | the strict local maxima of a `k`-cone envelope are characterised by a local active/parity rule | proved + verified | `maxima_criterion.py` |
| Joint quasi-polynomiality | `E_d(m, n)` is piecewise quasi-polynomial | proved (Barvinok–Woods) + verified | `transfer_check.py` |
| Polynomiality on the high region | `E_d = p_d`, a single symmetric polynomial, on `m, n ≥ max(d-1, 2)`; existence, symmetry, region, per-axis degree unconditional | proved + verified | `verify_claims.py`, `column_dp.py` |
| Closed forms | `p_d` computed explicitly through `d = 10` | computed | `extract_pd.py` (`d ≤ 7`), `compute_p10.py` (`d = 9, 10`) |
| Top-degree part | `4/(d-2)! · (m^{d-2} + n^{d-2})` for `d ≥ 5` | proved + verified | `verify_claims.py` |
| Degree bound | total degree of `p_d` is `d - 2` | conjectured; **proved for the separable case**; verified through `d = 7` | `verify_separable.py`, `confirm_d7.py`, `d8_leading_check.py` |
| Boundary correction | one step below threshold, leading coefficient `-4·Bax(d-3)` (Baxter numbers) | conjectured; verified through `d = 11` | `verify_claims.py`, `d8_leading_check.py` |

The existence, symmetry, region, and per-axis degree of `p_d` are unconditional.
Two features rest on conjecture: whether the total degree stays `d - 2` for every
`d` (the degree bound), and the exact form of the boundary correction below
threshold. The construction settles the degree bound where the count factors into
independent row and column contributions (the separable case).

## Code

Pure Python with [NumPy](https://numpy.org) and [SymPy](https://www.sympy.org);
the heavy `p_9`/`p_10` transfer-matrix DP additionally uses
[Numba](https://numba.pydata.org).

```sh
pip install numpy sympy numba
cd code
python3 verify_claims.py        # master: reproduces the headline claims (PASS/FAIL)
```

`verify_claims.py` runs in a few minutes (degrees `d ≤ 6`) and checks: the two
engines agree on `E_d`; the degree sequence is axis-symmetric; the `M_{2,n}`
vertex count is `2·3^{n-1}`; each `p_d` is a symmetric polynomial of total degree
`d - 2` with the predicted top-degree and boundary-correction coefficients.

| File | Role |
|---|---|
| `verify_claims.py` | **master** — reproduces the headline numerical claims |
| `extract_pd.py` | envelope split-counter for `E_d` and `p_d`; boundary stripes |
| `column_dp.py` | independent column transfer-matrix DP; rational-GF (single-pole) check |
| `transfer_matrix.py` | column transfer matrix `T_m` (Lemma ratGF) |
| `transfer_check.py` | the transfer DP reproduces `E_d` (cross-check) |
| `envelope_structure.py` | Envelope Structure Theorem, by enumeration |
| `maxima_criterion.py` | the `k`-cone Maxima Criterion |
| `verify_separable.py` | separable-case degree bound (`d = 4` is the `6mn` exception) |
| `boundary_lemma_verify.py` | a boundary column always carries a strict local extremum |
| `relation_B_complete_verify.py` | onset of the closed form at `d - 1` |
| `fast_split.py`, `split_diagnostic.py` | NumPy split counter + polynomial fitting |
| `fast_dp.py`, `fast_dp_nb2.py`, `interp_pd.py` | transfer-matrix DP (pure-Python + Numba) and exact interpolation |
| `compute_p9.py`, `compute_p10.py` | `p_9` / `p_10` via the Numba DP + interpolation |
| `transfer_period.py`, `cancellation_test.py` | supporting transfer-matrix diagnostics |

Heavier standalone checks (minutes each): `confirm_d7.py` (degree bound at
`d = 7`), `d8_leading_check.py` (`d = 8` leading term, envelope vs transfer
matrix), `compute_p10.py` (`p_10` via Numba, ~3 min).

## Validation

The column transfer-matrix DP reproduces the `M_{2,n}` degree sequence of
Christensen–Hull et al. (vertex count `2·3^{n-1}`), and the two independent
engines agree on `E_d(m, n)` for every degree and grid tested — the cross-check
that justifies trusting the closed forms.

## Lean formalization

`lean/` contains the Lean 4 formalization (Lean `v4.27.0` + Mathlib), a shared
`OrigamiCone` project covering both this paper and its companion. The sequel
modules are the `OrigamiCone/Sequel*.lean` files (envelope structure, the
transfer-matrix / rational generating function chain, separability, and the
polynomiality assembly).

Build (fetches the Mathlib cache; the `.lake` build tree is **not** shipped):

```sh
cd lean/OrigamiCone
lake exe cache get
lake build
```

The modules are `sorry`-free. Three results are taken as explicit Lean `axiom`s —
proved on paper, assumed in Lean — and are disclosed here:

- `Envelope_Lemma` (`OrigamiCone/SequelEnvFwd.lean`): every height function is the
  lower envelope of the distance cones at its strict local minima.
- `fiber_card_axiom`, `fiber_card_zero_axiom` (`OrigamiCone/SequelEdFiberCardAxiom.lean`):
  the fiber-cardinality identity for the edge-contraction count.

Every other sequel theorem is proved from Mathlib plus these three axioms; check
any result with `#print axioms <name>`.

## License

Released under the terms in the repository root. If you use this code or
formalization, please cite the paper.
