# An annihilation-number Caro-Wei bound

Exact-arithmetic verification code and a partial Lean formalization accompanying
Chakshu Gupta's paper, *An annihilation-number Caro-Wei bound: a TxGraffiti
conjecture and an independence-number bracket*.

**Paper:** [arXiv:2606.29553](https://arxiv.org/abs/2606.29553).

## Mathematical Results

For a finite simple graph, let $\alpha$ be its independence number, $a$ its
annihilation number, $R$ its Havel-Hakimi residue, $\Delta$ its maximum degree
and $W=\sum_v 1/(d(v)+1)$ its Caro-Wei sum.

The main estimate holds for every graph with at least one edge:

$$a\le\frac{\Delta+1}{2}W.$$

Equality is attained by regular graphs of positive degree and even order.
The proof combines the annihilation degree-sum estimate with the classical
reciprocal-degree estimate:

$$a\le n-\frac m\Delta\le\frac{\Delta+1}{2}W.$$

Together with the classical bounds, this gives

$$W\le R\le\alpha\le a\le\frac{\Delta+1}{2}W.$$

Thus $[R,a]$ is contained in $[W,(\Delta+1)W/2]$. Both endpoints improve
strictly on $K_{1,3}$, where the intervals are $[3,3]$ and $[7/4,7/2]$.
The sharp worst-case ratio $a/R=(\Delta+1)/2$ is still attained by connected
graphs of every positive maximum degree.

The TxGraffiti inequality

$$\Delta\alpha\ge a+R$$

holds for every connected graph with at least three vertices. The single edge
$K_2$ is the only nontrivial connected exception. Equality holds exactly for
$P_3$, $P_4$, $P_6$, $C_3$, $C_4$, $C_5$, $C_7$ and $K_4$.
For $\Delta\ge3$, the estimate $(a+R)/\Delta$ never exceeds the residue $R$.

## Quick Start

Install Python 3, [NetworkX](https://networkx.org) and `geng` from
[`nauty`](https://pallini.di.uniroma1.it/). On macOS, `brew install nauty`
provides `geng`. Run from this directory:

```sh
python3 -m pip install networkx
python3 code/c1_verify_fast.py 3 9
python3 code/c1_verify_fast.py --check 3 8
```

The first command checks all 273,191 connected graphs of orders 3 through 9
and reports zero violations. The per-order counts are
`2, 6, 21, 112, 853, 11117, 261080`, matching
[OEIS A001349](https://oeis.org/A001349).
The second command cross-checks the bitset independence-number computation
against NetworkX on all 12,111 connected graphs of orders 3 through 8.
Its annihilation-number and residue functions are copied from the reference
implementation, so those comparisons are consistency checks, not independent
oracles. The verifier checks generator exit status and includes $K_2$ as a
positive control for detecting a violated inequality.

These commands reproduce the finite check reported in the revised manuscript.
They are not a proof of the general statements and do not certify an order-ten
or a separate degree-sequence census. Use the explicit order ranges above;
the verifier's no-argument default requests order ten. The remaining scripts
are retained as supplementary investigations, not prerequisites for this run.

## Lean Coverage And Limits

The pinned [Lean project](lean/README.md) proves the core inequality
$a\le(\Delta+1)W/2$ over Mathlib's `SimpleGraph` without custom axioms.
It also proves the Caro-Wei and residue lower bounds, $\alpha\le a$,
the graph-level TxGraffiti inequality, its full eight-graph equality
classification, and strictness for connected graphs of maximum degree at
least four. The residue-deletion monotonicity step is proved, not assumed.
These results depend only on `propext`, `Classical.choice` and `Quot.sound`.

The rational-arithmetic lemmas in `Corollaries.lean` take their invariant
inequalities as hypotheses. They do not formalize the revised five-term
bracket or the connected sharpness family for every maximum degree.
The classical graph inequality $W\le R$ and domination of the TxGraffiti
estimate by $R$ are also outside the formalization. The full revised paper
is not claimed to be kernel-verified.

## Attribution

The annihilation-number definition and the bound $\alpha\le a$ appear in Pepper's
[*Binding Independence*](https://www.proquest.com/docview/305196562).
The reciprocal-degree estimate and residue inequalities are from Favaron,
Maheo and Sacle, [*On the residue of a graph*](https://doi.org/10.1002/jgt.3190150107).
The known Caro-Wei approximation estimate is presented by Boppana,
Halldorsson and Rawitz, [*Simple and local independent set approximation*](https://doi.org/10.1016/j.tcs.2020.09.018).
The conjecture is from Davila, Brimkov and Pepper's
[2025 preprint](https://arxiv.org/abs/2507.17780v1).
Kadrawi and Levit [refine the annihilation bound](https://arxiv.org/abs/2607.01438v1)
to an exact slack identity and give an independent proof of the TxGraffiti
inequality. Graph generation uses McKay and Piperno's `nauty`/`geng`.

## License

The code is distributed under the repository's [CC BY 4.0 license](../../LICENSE).
