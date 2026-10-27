# Unfaulted ML-KEM-512 Board Baseline

The retained nRF52840DK record reports three deterministic ML-KEM-512
key-generation, encapsulation and decapsulation roundtrips. Each case checks
the API return values and equality of the encapsulated and decapsulated shared
secrets. The fixed public test coins are not production randomness.

This is an unfaulted functional baseline, not a known-answer test, FIPS
validation, randomness test, performance benchmark, fault-injection experiment,
rank measurement or secret-recovery demonstration.

## Retained Material

| File | Contents |
|---|---|
| [main.c](main.c) | Ordinary unfaulted API test with three fixed-coin cases |
| [Makefile](Makefile) | RIOT build for `nrf52840dk` |
| [observed-markers.txt](observed-markers.txt) | Pass markers projected from the observed UART outcome, not the full raw log |
| [summary.json](summary.json) | Outcome, dependency pins, firmware hash and limitations |

| Dependency | Commit |
|---|---|
| mlkem-native | `d1b2fe782888bdb761a50336012923180be7f502` |
| RIOT | `62d0a88708b036bda687f33aacf7862450b6ef7f` |

The recorded firmware SHA-256 is
`0c1de81ad7351a0ab021fbeffa22899bf61fa5d4b3381816ddfd12169763f9da`.
The Makefile configures a 32,768-byte main-thread stack; this is not a measured
stack high-water mark.

## Host Check

From this directory, set `MLKEM_NATIVE_ROOT` to a checkout at the pinned commit.
The same ordinary API test can run on the host without RIOT or a board.

```sh
clang -std=c99 -O2 -Wall -Wextra -Werror \
  -I"$MLKEM_NATIVE_ROOT/mlkem" main.c -o /tmp/mlkem-unfaulted
printf 't\n' | /tmp/mlkem-unfaulted
```

The output starts with `MLKEM512_UNFAULTED_READY`, followed by the four lines in
[observed-markers.txt](observed-markers.txt). A host pass does not repeat the
physical-board run.