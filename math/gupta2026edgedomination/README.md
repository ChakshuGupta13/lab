# Domination versus edge domination in regular graphs of degree at least seven

Verification code for the paper *"Domination versus edge domination in regular
graphs of degree at least seven"* (Chakshu Gupta, Georgia Institute of
Technology).

**Paper**: [arXiv:2608.22498](https://arxiv.org/abs/2608.22498) (math.CO);
submitted to *Graphs and Combinatorics* (Springer).

## The conjecture

For a graph $G$ write $\gamma(G)$ for the **domination number** (the least size
of a set $S$ of vertices with every vertex outside $S$ adjacent to $S$) and
$\gamma_e(G)$ for the **edge domination number** (the least size of a maximal
matching, equivalently a minimum maximal matching). Baste, Fürst, Henning, Mohr,
and Rautenbach (*Domination versus edge domination*, Discrete Applied
Mathematics 285 (2020) 343–349, [arXiv:1906.10420](https://arxiv.org/abs/1906.10420))
conjectured that regularity forces domination below edge domination:

$$\gamma(G) \;\le\; \gamma_e(G) \qquad\text{for every } \Delta\text{-regular graph } G,\ \Delta \ge 1.$$

Without regularity it can fail ($P_4$ has $\gamma = 2 > 1 = \gamma_e$). It is
sharp at $\Delta = 3$.

## The result

- Combining published domination bounds with the matching lower bound settles
  the inequality for every degree $\Delta \ge 9$.
- A reduction expresses the conjecture as the existence of a **dominating
  transversal** of a minimum maximal matching (one endpoint per matching edge
  dominating the unsaturated set), and the **Lovász Local Lemma** proves such a
  transversal exists for every $\Delta \ge 7$ — newly closing degrees seven and
  eight and leaving degrees three through six open.
- The cubic case is studied through a local-search reformulation of the
  transversal (potential $\Phi = 4\beta + P$), a **fractional-LP barrier**
  ($\beta = 1/6 > 0$, so no linear discharging closes it), a satisfiability
  reformulation, and an explicit **50-vertex cubic graph** at which the reduction
  provably fails ($\gamma = 14$, unique minimum maximal matching of size $15$,
  associated formula unsatisfiable) though $\gamma \le \gamma_e$ still holds.
- The inequality cannot be tightened: infinitely many cubic graphs have
  $\gamma = \gamma_e$.

## Code

Pure Python with [NumPy](https://numpy.org)/[SciPy](https://scipy.org) (linear
programs), [NetworkX](https://networkx.org), and
[python-sat](https://pysathq.github.io) (SAT). Exact integer/rational arithmetic
where it matters; every script recomputes its result from scratch, prints it,
and exits.

```sh
pip install numpy scipy networkx python-sat
cd code

# Sec. 1 Table 1 thresholds + Sec. 3 lopsided-LLL negative-dependency lemma
python3 lll_thresholds.py

# Sec. 2 counting identity  |M| = (Delta*n + 2q)/(4*Delta - 2)
python3 counting_identity.py

# Sec. 4 per-edge characterization of the potential change (checked at n = 10, 12)
python3 per_edge_characterization.py

# Sec. 4-5 exhaustive: the integer potential Phi = 4*beta + P has no beta >= 1
# local minimum over cubic graphs (pass a small order for a quick run)
python3 cubic_potential_check.py 8

# Sec. 5 fractional barrier: no linear discharging closes the cubic case
python3 discharging_lp.py
python3 barrier_lp_delta.py 3 4 5 6      # the barrier at each degree

# Sec. 7 sharpness: gamma = gamma_e on structured families at Delta = 3..6
python3 baste_sharpness_all_degrees.py

# Appendix A: independent verification of the 50-vertex integral obstruction
python3 appendix_certificate.py
```

| File | What it verifies |
|---|---|
| `lll_thresholds.py` | Sec. 1 Table 1 reachability thresholds; Sec. 3 lopsided-LLL negative-dependency lemma |
| `counting_identity.py` | Sec. 2 counting identity, and that the 50-vertex witness attains the matching floor |
| `per_edge_characterization.py` | Sec. 4 per-edge formula for the potential change $\Delta\Phi = L - G$ |
| `cubic_potential_check.py` | Sec. 4–5 exhaustive check that $\Phi = 4\beta + P$ has no $\beta \ge 1$ local minimum |
| `discharging_lp.py` | Sec. 5 fractional barrier: $\beta = 1/6 > 0$, so no linear discharging closes the cubic case |
| `barrier_lp_delta.py` | Sec. 5 the fractional barrier at each degree $\Delta$ |
| `baste_sharpness_all_degrees.py` | Sec. 7 equality $\gamma = \gamma_e$ on structured $\Delta$-regular families |
| `appendix_certificate.py` | Appendix A: the 50-vertex cubic obstruction ($\gamma = 14$, unique minimum maximal matching, unsatisfiable formula) |
| `cubic_construction_probe.py`, `baste_perturb_equality.py` | shared helpers (cubic-graph enumeration; domination / edge domination via SAT) |

## Validation

Each script prints its result and exits zero. The Appendix A certificate treats
the defining 20-clause formula as untrusted input and independently rebuilds the
50-vertex graph, confirms its unique minimum maximal matching of size $15$,
computes $\gamma(G) = 14$ directly, and verifies the associated formula is
unsatisfiable. The counting identity holds with zero violations across every
tested graph, and the per-edge characterization matches the direct potential
computation on all bad transversals at orders $n \in \{10, 12\}$.
