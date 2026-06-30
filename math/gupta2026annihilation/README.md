# An annihilation-number Caro–Wei bound: a TxGraffiti conjecture and an independence-number bracket

Verification code and Lean formalization for the paper *"An annihilation-number
Caro–Wei bound: a TxGraffiti conjecture and an independence-number bracket"*
(Chakshu Gupta, Georgia Institute of Technology).

**Paper**: [arXiv:2606.29553](https://arxiv.org/abs/2606.29553) (math.CO, 2026).

## The conjecture

For a graph $G$ write $\alpha(G)$ for the independence number, $\Delta(G)$ for
the maximum degree, $a(G)$ for the **annihilation number** (the largest $j$ such
that the $j$ smallest vertex degrees sum to at most the number of edges), and
$R(G)$ for the **residue** (the number of zeros left when the Havel–Hakimi
process is run on the degree sequence to termination). The automated conjecturing
program **TxGraffiti** (Davila–Brimkov–Pepper,
[arXiv:2507.17780](https://arxiv.org/abs/2507.17780)) produced, as its oldest
open conjecture (open since 2016):

$$\alpha(G) \;\ge\; \frac{a(G) + R(G)}{\Delta(G)} \qquad\text{for every nontrivial connected graph } G.$$

Prior work established it only for two special families (regular bipartite graphs
and cubic König–Egerváry graphs). This paper proves it for **every** nontrivial
connected graph; the single edge $K_2$ is the only counterexample, and the bound
is sharp, with equality at the complete graph $K_4$.

## The method

Let $W(G) = \sum_{v} 1/(\deg(v)+1)$ be the Caro–Wei sum. Substituting the
Caro–Wei bound $\alpha \ge W$ for the residue bound $\alpha \ge R$ reduces the
conjecture to a structure-free degree-sequence inequality, the **vehicle**

$$a \;\le\; (\Delta - 1)\,W \qquad (\Delta \ge 3),$$

with a direct argument for $\Delta = 2$ (where $a = \alpha$). Combined with the
classical lower bounds $\alpha \ge R$ and $\alpha \ge W$, the vehicle gives
$\Delta\alpha \ge a + R$ for every connected graph of maximum degree at least
three.

The vehicle itself splits by maximum degree:

- **$\Delta \ge 4$** — via AM–HM and the coupling $m \le (n - a)\Delta$, the
  inequality reduces to a quadratic $N(t)$ whose discriminant factor
  $E(\Delta) = -3\Delta^4 + 8\Delta^3 + 6\Delta^2 - 8\Delta + 1$ is negative.
- **$\Delta = 3$** — degrees lie in $\{1,2,3\}$, where the annihilation number
  has a closed form in the degree counts $(n_1, n_2, n_3)$ that yields
  $a \le 2W$ in three regimes.

## Results

| Result | Statement |
|---|---|
| Theorem 1 | $a \le \tfrac{\Delta+1}{2}\,W$ — the annihilation–Caro–Wei vehicle |
| Theorem 2 | $\Delta\alpha \ge a + R$ for every connected graph with $\Delta \ge 2$ (the conjecture for $\Delta \ge 2$) |
| Corollary 1 | sharpness: $\Delta\alpha = a + R$ iff $(\Delta-1)\alpha = a$ and $\alpha = R$; attained at $K_4$ |
| Corollary 2 | $a \le (\Delta - 1)\alpha$ for $\Delta \ge 3$ |
| Corollary 3 | the independence-number bracket between the polynomial-time computable $R$ and $a$ |

The only nontrivial connected graph for which the bound fails is the single edge
$K_2$ (maximum degree one).

## Code

Pure Python with [NetworkX](https://networkx.org) and `geng` from
[`nauty`](https://pallini.di.uniroma1.it/); exact integer/rational arithmetic
throughout. Every script runs in seconds to minutes on a laptop.

```sh
pip install networkx          # and: brew install nauty   (for geng)
cd code

# Theorem 2 directly (Delta*alpha >= a+R), all connected graphs n = 3..10
python3 c1_verify_fast.py 3 10        # 11,989,762 graphs (Delta>=2), 0 violations
python3 c1_verify_fast.py --check 3 8 # gate: alpha cross-validated vs NetworkX

# Theorem 1 + the vehicle a <= (Delta-1)*alpha, all connected graphs n = 3..9
python3 c1_verify.py 3 9

# the case split, the reduction to the vehicle, and the vehicle's proof
python3 c1_subclass.py 3 9            # Delta=2 ; Delta>=3, alpha>=n/2 ; residual
python3 c1_decompose.py 3 9           # reduction to L': a <= (Delta-1)W
python3 c1_lprime.py 4 13             # L' over all graphic degree sequences
python3 c1_ratio_proof_check.py       # step-by-step proof of a <= (Delta+1)/2 W
python3 c1_ratio_sup.py               # the uniform bound a/W <= (Delta+1)/2

# the two maximum-degree regimes
python3 c1_delta4.py 14               # Delta>=4: graphs + sequences + symbolic E(Delta)<0
python3 c1_delta3_proof.py 200        # Delta=3: closed form + a <= 2W (686,900 multisets)
```

| File | What it verifies |
|---|---|
| `c1_verify.py` | the conjecture and $a \le (\Delta-1)\alpha$ on all connected graphs (NetworkX $\alpha$) |
| `c1_verify_fast.py` | Theorem 2 directly to order ten (bitset $\alpha$; `--check` vs reference) |
| `c1_subclass.py` | the $\alpha \ge n/2$ case split and its load-bearing facts |
| `c1_decompose.py` | the reduction to the vehicle $L'\!: a \le (\Delta-1)W$, stratified |
| `c1_lprime.py` | $L'$ over all graphic degree sequences (Erdős–Gallai) |
| `c1_ratio_proof_check.py` | the step-by-step proof of $a \le \tfrac{\Delta+1}{2}W$ |
| `c1_ratio_sup.py` | the uniform bound $a/W \le \tfrac{\Delta+1}{2}$ |
| `c1_delta4.py` | the $\Delta \ge 4$ proof: graphs + sequences + symbolic discriminant |
| `c1_delta3_proof.py` | the $\Delta = 3$ regime closed form and $a \le 2W$ |

## Lean formalization

The `lean/` directory formalizes the paper's algebraic content (Theorem 1,
Theorem 2, Corollaries 1–3) against **Mathlib v4.30.0-rc2**, axiom-clean. See
[`lean/README.md`](lean/README.md) for the statement of each result.

## Validation

The exhaustive search is cross-validated: `c1_verify_fast.py --check` confirms the
bitset independence number against NetworkX on every graph up to order eight, and
the annihilation number and residue are computed two independent ways. No
violation occurs anywhere in the search — the conjecture holds on all
$11{,}989{,}762$ connected graphs of order at most ten, and the vehicle holds on
all graphic degree sequences up to order fourteen ($\Delta \ge 4$) and order two
hundred ($\Delta = 3$).
