# Verification Records

This directory records the configurations and replay material used for the
paper's algebraic and bounded implementation checks. The synthetic test coins
are included so the thirty-case ML-KEM result can be replayed independently.
They are experiment fixtures, not production credentials.

## Artifact Map

| Check | Material |
|---|---|
| Joint and stacked replacement ranks | [`algebraic/`](algebraic/) |
| ML-KEM thirty-case replay and KAT record | [`mlkem/`](mlkem/) |
| Cortex-M4 QEMU record | [`qemu/`](qemu/) |
| ML-DSA-44 public-key change | [`mldsa/`](mldsa/) |
| Ancillary unfaulted physical-board baseline | [`unfaulted-board/`](unfaulted-board/) |
| Exact algebraic configurations | [`configurations.md`](configurations.md) |

## Quick Checks

The algebraic replacement checker uses only the Python standard library.

```sh
python3 -B algebraic/verify_replacement_values.py
```

The QEMU record verifier uses Node.js 22 or later. It checks the published
vectors against the declared faulted transform and public-key observer.

```sh
node qemu/verify_record.mjs
```

## ML-KEM Replay

The complete replay requires Apple Clang or a compatible C compiler, Node.js
22 or later, and the pinned `mlkem-native` source.

```sh
cd mlkem
git clone https://github.com/pq-code-package/mlkem-native.git upstream/mlkem-native
git -C upstream/mlkem-native checkout d1b2fe782888bdb761a50336012923180be7f502
node replay.mjs test-coins.json expected-records.json
node run_kats.mjs
```

The first command rebuilds thirty instrumented cases and checks every public
record. The KAT command uses the same patched source, but the upstream KAT build
does not define the fault-hook macro.

## ML-DSA Replay

The ML-DSA-44 check requires Clang, Node.js 22 or later, and the pinned PQClean
source.

```sh
cd mldsa
git clone https://github.com/PQClean/PQClean.git upstream/PQClean
git -C upstream/PQClean checkout 2cc64716044832eea747234ddbffc06746ab815d
PQCLEAN_ROOT="$PWD/upstream/PQClean" node replay.mjs
```

## Limits

The QEMU record uses debugger-imposed state changes. It is not a pure
instruction skip or a physical fault demonstration. The ML-KEM cases hold the
other key-generation inputs fixed when checking the two free coordinates. The
ML-DSA record establishes a changed compressed public key, not secret
reconstruction. The algebraic checks do not establish optimized-code
equivalence.

## License

CC BY 4.0. See [LICENSE](../../../LICENSE).