# qbudget

A small utility that reports the **T-count and AND-uncomputation accounting
of a compute-uncompute Grover oracle**, in a standardized 7-field record.

The record makes it explicit which implementation conventions a reported
T-count assumes — specifically whether AND gates were uncomputed via
measurement (Q# `ApplyAnd` / ProjectQ QAND† convention, 0 T-gates reverse)
or unitarily (Qiskit Jones-AND inverse, 4 T-gates reverse) — so cross-paper
T-count comparisons are not silently miscalibrated.

`qbudget` does not introduce a new algorithm or optimization. It packages
a reporting template grounded in three prior discussions:

- **Gidney (2018)**, *Halving the cost of quantum addition*, [Quantum 2:74](https://doi.org/10.22331/q-2018-06-18-74).
  Introduces the temporary logical-AND construction (4 T forward, 0 T
  uncomputation via measurement) and notes its applicability to Grover
  oracles.
- **Lin, Xiang, Xu, Zeng, Zhang (2023)**, *Quantum circuit implementations
  of SM4 block cipher based on different gate sets*,
  [QIP 22:282](https://doi.org/10.1007/s11128-023-04002-4). Table 7 reports
  the same SM4 S-box at 204 T with QAND and 413 T without QAND
  (ratio ≈ 2.02), making the framework-availability dependence explicit
  within a single paper.
- **Liao, Luo (2025)**, *Quantum Circuit Synthesis for AES with Low DW-cost*.
  Footnote 3 cross-compares the Toffoli-depth-40 figure reported by Liu et al.
  (2023), Shi & Feng (2024), and Jang et al. (2025) and notes that this number
  depends on the QAND† convention; without QAND† the same constructions would
  report Toffoli depth 7 (S-box) and 70 (AES-128).

The convention is well established within the specialist community. This
repository exists to make it cheap to apply uniformly: one Python call
extracts the seven fields from a Qiskit circuit or a Q# Resource Estimator
result.

## Install

```bash
pip install -r requirements.txt
```

`qbudget.py` is a single file; copy it into your project or import
from this checkout.

## Quick start (Qiskit)

```python
from qiskit import QuantumCircuit
from qbudget import budget_qiskit

# your oracle (compute + marker + uncompute)
full_oracle  = build_my_oracle(...)
fwd_only     = build_my_oracle_forward(...)   # compute + marker, no inverse

d = budget_qiskit(
    forward_circuit = fwd_only,
    full_circuit    = full_oracle,
    domain          = "AES BP11 S-box Grover oracle (W=8)",
    framework       = "qiskit-1.0.2",
    opt_regime      = "qiskit transpile opt_level=1, basis Clifford+T",
    measurement_uncompute_used = False,        # Qiskit Jones-AND uses unitary reverse
    and_count_hint  = 34,                      # exact AND count from the cipher builder
)
print(d)
# {'D': 'AES BP11 S-box Grover oracle (W=8)',
#  'M': False, 'Tf': 136, 'To': 328,
#  'F': 'qiskit-1.0.2',
#  'O': 'qiskit transpile opt_level=1, basis Clifford+T',
#  'A': 34}
```

## Quick start (Q#)

```python
import qsharp
from qbudget import budget_qsharp

qsharp.eval(my_qsharp_program)
re_forward = qsharp.estimate("Forward()")    # forward-only half
re_full    = qsharp.estimate("Main()")       # full oracle

d = budget_qsharp(
    forward_estimate = re_forward,
    full_estimate    = re_full,
    domain           = "AES BP11 structural skeleton (N_and=34, W=8)",
    framework        = "qsharp-1.28.0",
    opt_regime       = "Q# Resource Estimator logical counts",
)
print(d)
# {'D': 'AES BP11 structural skeleton (N_and=34, W=8)',
#  'M': True, 'Tf': 164, 'To': 164,
#  'F': 'qsharp-1.28.0',
#  'O': 'Q# Resource Estimator logical counts',
#  'A': 41}
```

## The 7 fields

| Key | Name                       | Description                                                                                            |
|-----|----------------------------|--------------------------------------------------------------------------------------------------------|
| D   | Domain                     | What the circuit computes (free-text).                                                                 |
| M   | Measurement-uncompute used | `True` iff the oracle relies on measurement-based AND uncomputation (QAND†, ApplyAnd, or equivalent). |
| Tf  | Forward T-count            | T + T† in the forward (compute) half.                                                                  |
| To  | Total T-count              | T + T† over the entire oracle.                                                                         |
| F   | Framework                  | SDK / language identifier with version.                                                                |
| O   | Optimization regime        | Synthesis path / optimization level / basis.                                                           |
| A   | AND count                  | Number of two-input ANDs in the forward half.                                                          |

See [`SPEC.md`](SPEC.md) for the field definitions, the auto-detected vs
user-supplied split, and design notes.

## Worked examples

The [`examples/`](examples/) directory contains four small Grover oracle
skeletons built with their published structural AND counts:

| Oracle           | Cipher AND count | Marker width W | Qiskit T (Jones-AND V-chain) | Q# T (ApplyAnd V-chain) |
|------------------|-----------------:|---------------:|-----------------------------:|------------------------:|
| AES BP11 S-box   | 34               | 8              | 328                          | 164                     |
| SM4 Fermat S-box | 384              | 8              | 3128                         | 1564                    |
| SHA-256 Ch₃₂     | 32               | 32             | 504                          | 252                     |
| SHA-256 Maj₃₂    | 32               | 32             | 504                          | 252                     |

The ratio T<sub>Qiskit</sub> / T<sub>Q#</sub> = 2.000 in every row, which
follows by construction in any compute-uncompute Grover oracle: each forward
AND is paired with an uncomputation; Q#'s `ApplyAnd` uncomputes at 0 T via
measurement (Gidney 2018), while Qiskit's `Jones AND` uncomputes at 4 T
unitarily.

Run the master comparison:

```bash
cd examples
python oracle_comparison_master.py
```

This regenerates [`examples/results/oracle_comparison_master.json`](examples/results/oracle_comparison_master.json)
and the summary table above.

Other example scripts:

- [`mcx_vchain_audit.py`](examples/mcx_vchain_audit.py) — measures Qiskit
  V-chain MCX T-cost at several widths vs the default `MCXGate` decomposition.
- [`qsharp_re_endtoend.py`](examples/qsharp_re_endtoend.py) — Q# Resource
  Estimator probes for the 4 oracle skeletons and standalone V-chain widths.
- [`best_practice_oracles.py`](examples/best_practice_oracles.py) —
  Qiskit-side Jones-AND V-chain marker construction.
- [`oracle_correctness_test.py`](examples/oracle_correctness_test.py) —
  small-instance statevector check that the oracle marks the target value.

## Tests

```bash
pip install pytest
pytest tests/
```

## Non-goals

- **No claim of a new identity.** The 2× T-cost gap between measurement-
  uncomputation and unitary uncomputation is a well-established structural
  fact, explicitly stated by Gidney (2018) and exhibited within a single
  paper by Lin et al. (2023, Table 7).
- **No claim of an audit finding.** The disclosure convention is openly
  used by Jaques et al. (2020), Lin et al. (2023), Liao & Luo (2025),
  Jang et al. (2025), and others. This repository is a tool to apply the
  convention uniformly, not a critique of any paper.
- **No fault-tolerant cost model.** `qbudget` reports gate-level
  counts that feed into magic-state-distillation and logical-qubit
  estimates. It does not perform those estimates.

## License

MIT, see [`LICENSE`](LICENSE).

## Citation

If you use this utility in a publication, please cite the prior work that
motivates the convention (see [`references.bib`](references.bib)) and
optionally link this repository.
