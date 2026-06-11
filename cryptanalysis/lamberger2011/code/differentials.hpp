// Piece 2: Differential characteristics for the 46-step boomerang attack
// Lamberger & Mendel (2011), Tables 1-3.
//
// Provides generic types for differential characteristics and one concrete
// instance (TABLE1) derived from the paper's worked example.

#pragma once

#include "sha256.hpp"
#include <array>

namespace diff46 {

using sha256::Word;
using sha256::State;

// Number of steps in the reduced compression function.
constexpr int STEPS = 46;

// Meeting point: E0 covers steps [0, SPLIT), E1 covers [SPLIT, STEPS).
constexpr int SPLIT = 21;

// ---------------------------------------------------------------------------
// Generic types
// ---------------------------------------------------------------------------

// Per-step state difference (8 registers, modular arithmetic).
// step_diffs[i] = expected state diff AFTER step i-1 (i.e. states[i]).
// Index 0 = IV diff, index STEPS = output diff.
struct StepDiffs {
    State d[STEPS + 1];
};

// Per-step message schedule difference.
struct MsgDiffs {
    Word dW[STEPS];
};

// A full differential characteristic for the 46-step boomerang.
struct Characteristic {
    // First-order input differences (24 words each: 16 msg + 8 IV).
    Word a1[24];
    Word a2[24];

    // State differences induced by a1 and a2 at every step.
    StepDiffs a1_state;
    StepDiffs a2_state;

    // Message schedule differences for a1 and a2.
    MsgDiffs a1_msg;
    MsgDiffs a2_msg;

    // Key derived values:
    State beta;     // a1 state diff at meeting point (step SPLIT)
    State gamma;    // a2 state diff at meeting point (step SPLIT)
    State alpha;    // a1 state diff at step 0 (backward output)
    State delta;    // a2 state diff at step STEPS (forward output)

    // Absorption boundaries:
    // backward: a1 state diff is zero from step bwd_zero_from through SPLIT
    // forward:  a2 state diff is zero from step fwd_zero_from through fwd_zero_to
    int bwd_zero_from;
    int fwd_zero_from;
    int fwd_zero_to;
};

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

// Verify a1 or a2 state diffs against a pair of traces.
// states0/states1 are arrays of STEPS+1 State values.
inline bool verify_state_diffs(const StepDiffs& expected,
                               const State* states0,
                               const State* states1) {
    for (int step = 0; step <= STEPS; ++step)
        for (int r = 0; r < 8; ++r)
            if (states1[step][r] - states0[step][r] != expected.d[step][r])
                return false;
    return true;
}

// Verify message schedule diffs.
inline bool verify_msg_diffs(const MsgDiffs& expected,
                             const Word* W0,
                             const Word* W1) {
    for (int i = 0; i < STEPS; ++i)
        if (W1[i] - W0[i] != expected.dW[i])
            return false;
    return true;
}

// ---------------------------------------------------------------------------
// Table 1 instance
// ---------------------------------------------------------------------------

// Concrete Table 1 base input y (message-first: M[0..15], IV[16..23]).
// This is the paper's worked example; the Characteristic below is the
// differential template that applies to any valid input.
constexpr Word TABLE1_Y[24] = {
    0x72939135, 0xc1570fea, 0x5c5d0c1d, 0xad031d03,
    0xd83c56b6, 0x41334f38, 0x12f67844, 0x0edd1fcb,
    0x016a5c6f, 0x39094c7b, 0x9e181d92, 0x54bfb0fa,
    0x506781eb, 0x0b081e5e, 0x607a28e0, 0x6318673a,
    0x21315086, 0x43909ad8, 0x23e8771b, 0x26ca42e8,
    0x1eecc4dd, 0x14649b3d, 0x9076304c, 0x29f92e96,
};

// Input differences (message-first layout: M[0..15], IV[16..23]).
constexpr Word TABLE1_A1[24] = {
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0xfffffffc, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000004, 0x00000000, 0x00000004,
    0xef800200, 0xfffffe00, 0x0ffffe00, 0xefbef7fc,
};

constexpr Word TABLE1_A2[24] = {
    0xa3a3e47f, 0x58cf0adb, 0x3fa82cc6, 0x0c907f06,
    0x3377c2cd, 0x1997456a, 0xa8bcf700, 0x2455c931,
    0xbffb159c, 0x504e97b0, 0x39e6d04a, 0x2a582f18,
    0x37fcb1a0, 0xd42be48e, 0x950d8d60, 0xed368ca3,
    0xe4ae4c2e, 0x989ee693, 0x8dd81c5e, 0x3abd607d,
    0x96c9d3bd, 0x589c4e77, 0x1a4afee7, 0xb9ba6518,
};

// a1 state differences at each step (extracted from Piece 1 trace).
constexpr StepDiffs TABLE1_A1_STATE = {{
    // step 0 (IV diff)
    {0x00000000, 0x00000004, 0x00000000, 0x00000004,
     0xef800200, 0xfffffe00, 0x0ffffe00, 0xefbef7fc},
    // step 1
    {0x00000000, 0x00000000, 0x00000004, 0x00000000,
     0x00000000, 0xef800200, 0xfffffe00, 0x0ffffe00},
    // step 2
    {0x00000000, 0x00000000, 0x00000000, 0x00000004,
     0x00000000, 0x00000000, 0xef800200, 0xfffffe00},
    // step 3
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000004, 0x00000000, 0x00000000, 0xef800200},
    // step 4
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000004, 0x00000000, 0x00000000},
    // step 5
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x00000004, 0x00000000},
    // step 6
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x00000000, 0x00000004},
    // steps 7-21: all zero
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    // step 22
    {0x08010000, 0x00000000, 0x00000000, 0x00000000,
     0x08010000, 0x00000000, 0x00000000, 0x00000000},
    // step 23
    {0x019e03f0, 0x08010000, 0x00000000, 0x00000000,
     0xf79e03d8, 0x08010000, 0x00000000, 0x00000000},
    // step 24
    {0x02bac42d, 0x019e03f0, 0x08010000, 0x00000000,
     0xc1e28aa3, 0xf79e03d8, 0x08010000, 0x00000000},
    // step 25
    {0xaed2c555, 0x02bac42d, 0x019e03f0, 0x08010000,
     0x8a3aae63, 0xc1e28aa3, 0xf79e03d8, 0x08010000},
    // step 26
    {0x785c69b9, 0xaed2c555, 0x02bac42d, 0x019e03f0,
     0x4a2ddf26, 0x8a3aae63, 0xc1e28aa3, 0xf79e03d8},
    // step 27
    {0xfe1d2c95, 0x785c69b9, 0xaed2c555, 0x02bac42d,
     0x7a17e51e, 0x4a2ddf26, 0x8a3aae63, 0xc1e28aa3},
    // step 28
    {0x3f250cbd, 0xfe1d2c95, 0x785c69b9, 0xaed2c555,
     0x8280f70f, 0x7a17e51e, 0x4a2ddf26, 0x8a3aae63},
    // step 29
    {0xcf262c00, 0x3f250cbd, 0xfe1d2c95, 0x785c69b9,
     0x3d301193, 0x8280f70f, 0x7a17e51e, 0x4a2ddf26},
    // step 30
    {0x98e73cd0, 0xcf262c00, 0x3f250cbd, 0xfe1d2c95,
     0xdec6d12e, 0x3d301193, 0x8280f70f, 0x7a17e51e},
    // step 31
    {0x491cb8f6, 0x98e73cd0, 0xcf262c00, 0x3f250cbd,
     0x3a548805, 0xdec6d12e, 0x3d301193, 0x8280f70f},
    // step 32
    {0x62e7db83, 0x491cb8f6, 0x98e73cd0, 0xcf262c00,
     0x9dbfc85f, 0x3a548805, 0xdec6d12e, 0x3d301193},
    // step 33
    {0x9cd684e5, 0x62e7db83, 0x491cb8f6, 0x98e73cd0,
     0xb96e4adb, 0x9dbfc85f, 0x3a548805, 0xdec6d12e},
    // step 34
    {0x0fbcdde1, 0x9cd684e5, 0x62e7db83, 0x491cb8f6,
     0xcc73705a, 0xb96e4adb, 0x9dbfc85f, 0x3a548805},
    // step 35
    {0x85fe5bad, 0x0fbcdde1, 0x9cd684e5, 0x62e7db83,
     0x61c6b59b, 0xcc73705a, 0xb96e4adb, 0x9dbfc85f},
    // step 36
    {0x9f6a019d, 0x85fe5bad, 0x0fbcdde1, 0x9cd684e5,
     0xd4585e37, 0x61c6b59b, 0xcc73705a, 0xb96e4adb},
    // step 37
    {0x420ba821, 0x9f6a019d, 0x85fe5bad, 0x0fbcdde1,
     0x700f6f2e, 0xd4585e37, 0x61c6b59b, 0xcc73705a},
    // step 38
    {0x88941a13, 0x420ba821, 0x9f6a019d, 0x85fe5bad,
     0x35d202b6, 0x700f6f2e, 0xd4585e37, 0x61c6b59b},
    // step 39
    {0xa8c9055c, 0x88941a13, 0x420ba821, 0x9f6a019d,
     0xa885ef05, 0x35d202b6, 0x700f6f2e, 0xd4585e37},
    // step 40
    {0x7b2222d8, 0xa8c9055c, 0x88941a13, 0x420ba821,
     0x50dcc333, 0xa885ef05, 0x35d202b6, 0x700f6f2e},
    // step 41
    {0x1c85894a, 0x7b2222d8, 0xa8c9055c, 0x88941a13,
     0x83176905, 0x50dcc333, 0xa885ef05, 0x35d202b6},
    // step 42
    {0xadf1bacb, 0x1c85894a, 0x7b2222d8, 0xa8c9055c,
     0x36ae17cd, 0x83176905, 0x50dcc333, 0xa885ef05},
    // step 43
    {0x2941cf79, 0xadf1bacb, 0x1c85894a, 0x7b2222d8,
     0xd78ef791, 0x36ae17cd, 0x83176905, 0x50dcc333},
    // step 44
    {0x631f2173, 0x2941cf79, 0xadf1bacb, 0x1c85894a,
     0x4a92fed5, 0xd78ef791, 0x36ae17cd, 0x83176905},
    // step 45
    {0xe1487b25, 0x631f2173, 0x2941cf79, 0xadf1bacb,
     0xa69ba27d, 0x4a92fed5, 0xd78ef791, 0x36ae17cd},
    // step 46 (output, after feed-forward)
    {0x51ff62fe, 0xe1487b25, 0x631f2173, 0x2941cf79,
     0xc9d76948, 0xa69ba27d, 0x4a92fed5, 0xd78ef791},
}};

// a2 state differences at each step.
constexpr StepDiffs TABLE1_A2_STATE = {{
    // step 0
    {0xe4ae4c2e, 0x989ee693, 0x8dd81c5e, 0x3abd607d,
     0x96c9d3bd, 0x589c4e77, 0x1a4afee7, 0xb9ba6518},
    // step 1
    {0x44c07e88, 0xe4ae4c2e, 0x989ee693, 0x8dd81c5e,
     0x3bf8cb8f, 0x96c9d3bd, 0x589c4e77, 0x1a4afee7},
    // step 2
    {0xdf5b8188, 0x44c07e88, 0xe4ae4c2e, 0x989ee693,
     0x8be0f80c, 0x3bf8cb8f, 0x96c9d3bd, 0x589c4e77},
    // step 3
    {0xa77d0c64, 0xdf5b8188, 0x44c07e88, 0xe4ae4c2e,
     0xa4df3450, 0x8be0f80c, 0x3bf8cb8f, 0x96c9d3bd},
    // step 4
    {0x8c10e073, 0xa77d0c64, 0xdf5b8188, 0x44c07e88,
     0x22d3a6e0, 0xa4df3450, 0x8be0f80c, 0x3bf8cb8f},
    // step 5
    {0x4ccf8586, 0x8c10e073, 0xa77d0c64, 0xdf5b8188,
     0x2395e8cf, 0x22d3a6e0, 0xa4df3450, 0x8be0f80c},
    // step 6
    {0x24ea3187, 0x4ccf8586, 0x8c10e073, 0xa77d0c64,
     0xf8738215, 0x2395e8cf, 0x22d3a6e0, 0xa4df3450},
    // step 7
    {0xf7c01d9b, 0x24ea3187, 0x4ccf8586, 0x8c10e073,
     0x559d8f9e, 0xf8738215, 0x2395e8cf, 0x22d3a6e0},
    // step 8
    {0xa427bd08, 0xf7c01d9b, 0x24ea3187, 0x4ccf8586,
     0x6ed9ba84, 0x559d8f9e, 0xf8738215, 0x2395e8cf},
    // step 9
    {0xf84c84fe, 0xa427bd08, 0xf7c01d9b, 0x24ea3187,
     0x95c5af06, 0x6ed9ba84, 0x559d8f9e, 0xf8738215},
    // step 10
    {0x63481667, 0xf84c84fe, 0xa427bd08, 0xf7c01d9b,
     0x22c4adf3, 0x95c5af06, 0x6ed9ba84, 0x559d8f9e},
    // step 11
    {0xdce54d54, 0x63481667, 0xf84c84fe, 0xa427bd08,
     0x1bfcda4c, 0x22c4adf3, 0x95c5af06, 0x6ed9ba84},
    // step 12
    {0xcc571014, 0xdce54d54, 0x63481667, 0xf84c84fe,
     0x1cf7298e, 0x1bfcda4c, 0x22c4adf3, 0x95c5af06},
    // step 13
    {0x3575964e, 0xcc571014, 0xdce54d54, 0x63481667,
     0x7d37b749, 0x1cf7298e, 0x1bfcda4c, 0x22c4adf3},
    // step 14
    {0xe4e81738, 0x3575964e, 0xcc571014, 0xdce54d54,
     0xcfef7092, 0x7d37b749, 0x1cf7298e, 0x1bfcda4c},
    // step 15
    {0xbf4df004, 0xe4e81738, 0x3575964e, 0xcc571014,
     0x86d918f0, 0xcfef7092, 0x7d37b749, 0x1cf7298e},
    // step 16
    {0x1e7defe0, 0xbf4df004, 0xe4e81738, 0x3575964e,
     0x9705ae54, 0x86d918f0, 0xcfef7092, 0x7d37b749},
    // step 17
    {0x354c6824, 0x1e7defe0, 0xbf4df004, 0xe4e81738,
     0xfa7d6024, 0x9705ae54, 0x86d918f0, 0xcfef7092},
    // step 18
    {0x00000000, 0x354c6824, 0x1e7defe0, 0xbf4df004,
     0x21f790ba, 0xfa7d6024, 0x9705ae54, 0x86d918f0},
    // step 19
    {0x40000000, 0x00000000, 0x354c6824, 0x1e7defe0,
     0xc0000000, 0x21f790ba, 0xfa7d6024, 0x9705ae54},
    // step 20
    {0x110a0120, 0x40000000, 0x00000000, 0x354c6824,
     0xeffe0000, 0xc0000000, 0x21f790ba, 0xfa7d6024},
    // step 21 = gamma
    {0x00000000, 0x110a0120, 0x40000000, 0x00000000,
     0xfffbef00, 0xeffe0000, 0xc0000000, 0x21f790ba},
    // step 22
    {0x00000000, 0x00000000, 0x110a0120, 0x40000000,
     0x00000000, 0xfffbef00, 0xeffe0000, 0xc0000000},
    // step 23
    {0xc0000000, 0x00000000, 0x00000000, 0x110a0120,
     0x00000000, 0x00000000, 0xfffbef00, 0xeffe0000},
    // step 24
    {0x00000000, 0xc0000000, 0x00000000, 0x00000000,
     0x01080020, 0x00000000, 0x00000000, 0xfffbef00},
    // step 25
    {0x00000000, 0x00000000, 0xc0000000, 0x00000000,
     0x00000000, 0x01080020, 0x00000000, 0x00000000},
    // step 26
    {0x00000000, 0x00000000, 0x00000000, 0xc0000000,
     0x00000000, 0x00000000, 0x01080020, 0x00000000},
    // step 27
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0xc0000000, 0x00000000, 0x00000000, 0x01080020},
    // step 28
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0xc0000000, 0x00000000, 0x00000000},
    // step 29
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0xc0000000, 0x00000000},
    // step 30
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x00000000, 0xc0000000},
    // steps 31-45: all zero
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    // step 46 (output)
    {0x087ff000, 0x00000000, 0x00000000, 0x00000000,
     0x087ff000, 0x00000000, 0x00000000, 0x00000000},
}};

// a1 message schedule differences.
// dW[0..5] = 0, dW[6] = 0xfffffffc (-4), dW[7..20] = 0, dW[21+] diverge.
constexpr MsgDiffs TABLE1_A1_MSG = {{
    0x00000000, 0x00000000, 0x00000000, 0x00000000,  // 0-3
    0x00000000, 0x00000000, 0xfffffffc, 0x00000000,  // 4-7
    0x00000000, 0x00000000, 0x00000000, 0x00000000,  // 8-11
    0x00000000, 0x00000000, 0x00000000, 0x00000000,  // 12-15
    0x00000000, 0x00000000, 0x00000000, 0x00000000,  // 16-19
    0x00000000,                                        // 20
    // Steps 21-45: depend on this specific y, not characteristic-level.
    // These are from the Table 1 instance only.
    0x08010000, 0xfffffffc, 0x600202c0, 0x00018000,  // 21-24
    0x3f1fbc7f, 0x8fffffe0, 0x24a01f87, 0x07f8c600,  // 25-28
    0xf44cf250, 0xf4c801e4, 0x1056b493, 0x9fea0a5b,  // 29-32
    0x64454da9, 0xdd1db292, 0x57b1ac05, 0x1fc0a691,  // 33-36
    0xd47b68b6, 0xbe389cfb, 0x292cedee, 0x0a04fce7,  // 37-40
    0x3209217e, 0x50f11f6b, 0xdbb5a87a, 0x482a79eb,  // 41-44
    0xdf42e43d,                                        // 45
}};

// Assemble the full Table 1 characteristic.
constexpr Characteristic TABLE1 = {
    // a1
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0xfffffffc, 0x00000000,
     0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000004, 0x00000000, 0x00000004,
     0xef800200, 0xfffffe00, 0x0ffffe00, 0xefbef7fc},
    // a2
    {0xa3a3e47f, 0x58cf0adb, 0x3fa82cc6, 0x0c907f06,
     0x3377c2cd, 0x1997456a, 0xa8bcf700, 0x2455c931,
     0xbffb159c, 0x504e97b0, 0x39e6d04a, 0x2a582f18,
     0x37fcb1a0, 0xd42be48e, 0x950d8d60, 0xed368ca3,
     0xe4ae4c2e, 0x989ee693, 0x8dd81c5e, 0x3abd607d,
     0x96c9d3bd, 0x589c4e77, 0x1a4afee7, 0xb9ba6518},
    // a1_state, a2_state
    TABLE1_A1_STATE,
    TABLE1_A2_STATE,
    // a1_msg (a2_msg left zero-initialized — instance-specific, not characteristic-level)
    TABLE1_A1_MSG,
    {},
    // beta: a1 state diff at step 21 (all zero)
    {0x00000000, 0x00000000, 0x00000000, 0x00000000,
     0x00000000, 0x00000000, 0x00000000, 0x00000000},
    // gamma: a2 state diff at step 21
    {0x00000000, 0x110a0120, 0x40000000, 0x00000000,
     0xfffbef00, 0xeffe0000, 0xc0000000, 0x21f790ba},
    // alpha: a1 state diff at step 0
    {0x00000000, 0x00000004, 0x00000000, 0x00000004,
     0xef800200, 0xfffffe00, 0x0ffffe00, 0xefbef7fc},
    // delta: a2 state diff at step 46
    {0x087ff000, 0x00000000, 0x00000000, 0x00000000,
     0x087ff000, 0x00000000, 0x00000000, 0x00000000},
    // absorption boundaries
    7,   // bwd_zero_from: a1 absorbed from step 7
    31,  // fwd_zero_from: a2 zero from step 31
    45,  // fwd_zero_to: a2 zero through step 45 (re-emerges at 46)
};

} // namespace diff46
