# qbudget — Specification

A small utility that emits a standardized 7-item disclosure for a quantum
cryptanalysis circuit. The disclosure makes it unambiguous which
implementation conventions a reported gate count assumes, so cross-paper
comparisons are not silently miscalibrated.

The convention follows the discussion in:

- Gidney 2018, *Halving the cost of quantum addition*, Quantum 2:74
  (introduces the temporary-AND / measurement-uncompute idea, including
  the Grover-oracle generalization).
- Lin et al. 2023, *Quantum circuit implementations of SM4 block cipher
  based on different gate sets*, QIP 22:282 (Table 7 reports the same
  S-box with and without QAND†, demonstrating the cost difference).
- Liao & Luo 2025, *Quantum Circuit Synthesis for AES with Low DW-cost*
  (footnote 3 makes the QAND†-based accounting convention explicit and
  cross-compares three prior papers).

`qbudget` does not propose a new method. It packages a reporting
template so that authors and readers can publish a complete picture in a
single line of code.

## The 7 items

| Key | Name                       | Required? | Type      | Description                                                                                            |
|-----|----------------------------|-----------|-----------|--------------------------------------------------------------------------------------------------------|
| D   | Domain                     | yes       | string    | What this circuit computes (e.g. `"AES BP11 S-box Grover oracle"`).                                    |
| M   | Measurement-uncompute used | yes       | bool      | True iff the circuit relies on measurement-based AND uncomputation (QAND†, ApplyAnd, or equivalent).   |
| Tf  | Forward T-count            | yes       | int       | T+T† gates in the forward (compute) half of the oracle. Excludes the reverse pass.                     |
| To  | Total T-count              | yes       | int       | T+T† gates over the entire oracle (compute + marker + uncompute).                                      |
| F   | Framework                  | yes       | string    | The SDK / language that produced the gate count (e.g. `"qiskit-1.x"`, `"qsharp-1.28"`, `"projectq"`).  |
| O   | Optimization regime        | yes       | string    | What synthesis path was used (e.g. `"qiskit transpile opt_level=1"`, `"Q# RE logical"`, `"pyzx full_reduce"`). |
| A   | AND count                  | yes       | int       | Number of two-input AND gates in the forward half (the structural quantity that determines T-cost).    |

`Tf`, `To`, `A` are integers measured from the circuit. `D`, `M`, `F`, `O`
are metadata the author supplies — they cannot be auto-inferred from
gates alone with full reliability.

## Two Python entry points

### Qiskit

```python
from qbudget import budget_qiskit

disclosure = budget_qiskit(
    forward_circuit = oracle_forward_qc,   # the compute-only half
    full_circuit    = full_oracle_qc,      # compute + marker + uncompute
    domain          = "AES BP11 S-box Grover oracle (W=8, marker=0xA5)",
    framework       = "qiskit-1.0.2",
    opt_regime      = "qiskit transpile opt_level=1, basis Clifford+T",
    measurement_uncompute_used = False,    # Qiskit Jones-AND uses T-gate reverse
    and_count_hint  = 34,                  # optional override
)
print(disclosure)
# {
#   "D": "AES BP11 S-box Grover oracle (W=8, marker=0xA5)",
#   "M": False,
#   "Tf": 136,
#   "To": 328,
#   "F": "qiskit-1.0.2",
#   "O": "qiskit transpile opt_level=1, basis Clifford+T",
#   "A": 34,
# }
```

`budget_qiskit` transpiles both circuits to the Clifford+T basis and counts
`t` + `tdg` operations to produce `Tf` and `To`. If `and_count_hint` is
not supplied, `A` defaults to `Tf // 4` with a warning — the user should
prefer to pass the exact AND count from their cipher builder.

### Q#

```python
from qbudget import budget_qsharp

disclosure = budget_qsharp(
    forward_estimate = qsharp_re_forward,   # qsharp.estimate() result, forward half
    full_estimate    = qsharp_re_full,      # qsharp.estimate() result, full oracle
    domain           = "AES BP11 structural skeleton (N_and=34, W=8)",
    framework        = "qsharp-1.28.0",
    opt_regime       = "Q# Resource Estimator logical counts (Std lib)",
)
```

`budget_qsharp` reads `logicalCounts["cczCount"]` from each estimate and
converts to T-counts via `T = 4 * CCZ` (Jones 2013). It detects
measurement-uncompute usage by checking whether `measurementCount > 0`.

## Output

Both functions return a `dict` with the seven keys. JSON-serializable.

```python
import json
print(json.dumps(disclosure, indent=2))
```

## Non-goals

- `qbudget` does not invent or recommend a synthesis. It reports
  what the user-supplied circuit already encodes.
- It does not estimate fault-tolerant cost (magic state distillation,
  logical qubit overhead, etc.). It reports the gate-level counts that
  feed into such estimates.
- It does not claim that the 2x T-cost gap between measurement-uncompute
  and unitary-uncompute is novel. That gap is well established
  (Gidney 2018, Lin 2023, Liao 2025). The tool only makes the disclosure
  uniform across papers.

## Worked examples

See `examples/` for the four oracle skeletons (AES BP11, SM4 Fermat,
SHA-256 Ch32, SHA-256 Maj32) measured in both Qiskit and Q#. Each script
ends with a printed disclosure that matches the expected reference output
in `examples/results/`.
