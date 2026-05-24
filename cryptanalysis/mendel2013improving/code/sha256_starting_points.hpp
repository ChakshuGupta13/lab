// Starting points for SHA-256 differential characteristic search.
//
// Shared library: includes starting points from multiple papers so that
// any paper directory can reference standard attack configurations.
//
// !!! IMPORTANT for SAT-based attacks !!!
// The starting points below transcribe each paper's *sparse* Table A
// (input to the paper's GnD/Phase-1 search). When fed to a SAT encoder
// they typically TIMEOUT well above their estimated complexity because
// the solver must re-derive Phase 1 from scratch.
//
// Most cryptanalysis papers ALSO publish a *dense* Table B — the fully
// solved characteristic (Phase-1 output, with u/n/0/1 conditions) — in
// their Appendix alongside an example collision. Encoding Table B
// instead of Table A is the difference between TIMEOUT and seconds:
// see the mendel2011 reproduction package for the empirical record.
//
// Verified reference cases:
//   - mendel2011 27-step collision: Table 6 SP times out at every n_fixed≤7;
//     Table 7 SP solves 0-fixed in ~40 s.
//   - mendel2011 32-step SFS:       Table 2 SP times out at 5M conflicts;
//     Table 3 SP solves in ~5.6 s (paper estimate "days on 32 nodes").
//   - Both implementations live in the mendel2011 reproduction package as
//     `starting_point_27_table7()` and `starting_point_32_sfs_table3()`.
//
// If you hit a wall implementing one of the papers below as SAT,
// transcribe the matching Table B from the paper before optimizing
// the SAT formulation. Also read Lessons 6 (u/n convention) and 7
// (paper-internal typos) before trusting your transcription.
//
// Currently included:
//   mendel2013::starting_point_28()      — 28-step real collision (Table 1, left)
//   mendel2013::starting_point_31()      — 31-step real collision (Table 1, mid)
//   mendel2013::starting_point_38_sfs()  — 38-step SFS (Table 1, right)
//
// Each starting point defines conditions on ∇A, ∇E, ∇W using the
// alternative SHA-256 description and generalized conditions.
//
// Symbols: '-' = equal (no diff), '?' = free, 'x' = unequal (diff, sign unknown).
// The search engine resolves '?' → '-'/'x' and 'x' → 'u'/'n'.
//
// Bit ordering: MSB-first (leftmost char = bit 31, rightmost = bit 0),
// matching WordCond::from_string().
//
// Sources:
//   Mendel-Nad-Schläffer 2013, EUROCRYPT 2013, Table 1.

#pragma once

#include "search.hpp"

namespace mendel2013 {

using namespace mendel2011;
using namespace gencond;

// ============================================================
// Table 1 (left): Starting point for 28-step collision
// ============================================================
//
// Local collision spans t=11 steps (step 8–18).
// Active message words: W_8, W_9, W_13, W_16, W_18.
// Expansion conditions: steps 20, 23, 24, 25.
// First 8 message words have no differences → real collision
// (IV matching via 2^{128} prefix freedom in W_0..W_7).
//
// This is the direct analogue of mendel2011 Table 6 (27-step)
// with V shifted from 7→8 and the local collision extended by 1 step.
//
// Verified against scratch_lc_search.cpp: (V=8, K=11) has unique optimal
// OBJ=9 with active={W8,W9,W13,W16,W18} + cancel={20,23,24,25}.

inline CharState starting_point_28() {
    CharState cs(28);

    // --- IV: standard SHA-256 IV, fully determined (no difference) ---
    cs.a(-1) = wc_from_constant(sha256::IV[0]);  // a = 6a09e667
    cs.a(-2) = wc_from_constant(sha256::IV[1]);  // b = bb67ae85
    cs.a(-3) = wc_from_constant(sha256::IV[2]);  // c = 3c6ef372
    cs.a(-4) = wc_from_constant(sha256::IV[3]);  // d = a54ff53a
    cs.e(-1) = wc_from_constant(sha256::IV[4]);  // e = 510e527f
    cs.e(-2) = wc_from_constant(sha256::IV[5]);  // f = 9b05688c
    cs.e(-3) = wc_from_constant(sha256::IV[6]);  // g = 1f83d9ab
    cs.e(-4) = wc_from_constant(sha256::IV[7]);  // h = 5be0cd19

    // --- State conditions ∇A and ∇E ---
    // Steps 0-7: no difference (all dash = equal in both messages)
    for (int i = 0; i <= 7; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // Steps 8-10: A and E all free (active region, full search)
    // (already '?' from constructor)

    // Step 11: A all dash, E all free
    cs.a(11) = wc_uniform(BC_DASH);
    // cs.e(11) stays '?'

    // Step 12: A all dash, E all free
    cs.a(12) = wc_uniform(BC_DASH);
    // cs.e(12) stays '?'

    // Step 13: A all dash, E all free
    cs.a(13) = wc_uniform(BC_DASH);
    // cs.e(13) stays '?'

    // Step 14: A all dash, E all free
    cs.a(14) = wc_uniform(BC_DASH);
    // cs.e(14) stays '?'

    // Steps 15-27: all dash (zero diff beyond active region)
    for (int i = 15; i < 28; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // --- Message conditions ∇W ---
    // Steps 0-7: no difference (first 8 free for IV matching)
    for (int i = 0; i <= 7; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Step 8: active (primary diff). Seed one 'x' at bit 16 to force
    // a non-trivial differential (prevents trivial all-dash solution).
    cs.w(8) = WordCond::from_string("???????????????x????????????????");

    // Step 9: active, all free
    // cs.w(9) stays '?'

    // Steps 10-12: no difference
    for (int i = 10; i <= 12; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Step 13: active, all free
    // cs.w(13) stays '?'

    // Steps 14-15: no difference
    cs.w(14) = wc_uniform(BC_DASH);
    cs.w(15) = wc_uniform(BC_DASH);

    // Step 16: active, all free
    // cs.w(16) stays '?'

    // Step 17: no difference
    cs.w(17) = wc_uniform(BC_DASH);

    // Step 18: active, all free
    // cs.w(18) stays '?'

    // Steps 19-27: no difference (within 28-step scope)
    for (int i = 19; i <= 27; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    return cs;
}

// ============================================================
// Table 1 (middle): Starting point for 31-step collision
// ============================================================
//
// Local collision spans t=14 steps (step 5–18).
// Active message words: W_5, W_6, W_7, W_8, W_9, W_16, W_18.
// Expansion conditions: steps 20–25.
// First 5 message words have no differences → needs 2-block approach
// (only 160 bits of freedom in W_0..W_4 < 256 bits needed for IV matching).
//
// Complexity: 2^{65.5} via 2-block MITM.

inline CharState starting_point_31() {
    CharState cs(31);

    // --- IV: free (SFS in block 1; 2-block approach matches later) ---
    for (int i = -4; i < 0; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // --- State conditions ∇A and ∇E ---
    // Steps 0-4: no difference
    for (int i = 0; i <= 4; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // Steps 5-7: A and E all free (active region)
    // (already '?' from constructor)

    // Step 8-10: A all free, E all free
    // (already '?' from constructor)

    // Step 11: A all dash, E all free
    cs.a(11) = wc_uniform(BC_DASH);

    // Steps 12-14: A all dash, E all free
    for (int i = 12; i <= 14; ++i)
        cs.a(i) = wc_uniform(BC_DASH);

    // Steps 15-30: all dash (zero diff)
    for (int i = 15; i < 31; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // --- Message conditions ∇W ---
    // Steps 0-4: no difference
    for (int i = 0; i <= 4; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Steps 5-9: active. Seed one 'x' at W[5] bit 16 to force
    // non-trivial differential (prevents trivial all-dash solution).
    cs.w(5) = WordCond::from_string("???????????????x????????????????");
    // cs.w(6)..cs.w(9) stay '?'

    // Steps 10-15: no difference
    for (int i = 10; i <= 15; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Step 16: active, all free
    // cs.w(16) stays '?'

    // Step 17: no difference
    cs.w(17) = wc_uniform(BC_DASH);

    // Step 18: active, all free
    // cs.w(18) stays '?'

    // Steps 19-30: no difference
    for (int i = 19; i <= 30; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    return cs;
}

// ============================================================
// Table 1 (right): Starting point for 38-step SFS collision
// ============================================================
//
// Local collision spans t=18 steps (step 7–24).
// Active message words: W_7, W_8, W_10, W_15, W_23, W_24.
// (6 active words — confirmed by paper; our local_collision.hpp search
// finds 5 with OBJ=11, missing W_8; the paper likely applies an
// additional constraint.)
// Expansion conditions: steps 20, 23, 25, 26, 27, 28, 29.
// No free prefix → semi-free-start only.

inline CharState starting_point_38_sfs() {
    CharState cs(38);

    // --- IV: free (SFS) ---
    for (int i = -4; i < 0; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // --- State conditions ∇A and ∇E ---
    // Steps 0-6: no difference
    for (int i = 0; i <= 6; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // Steps 7-10: A and E all free (active region)
    // (already '?' from constructor)

    // Steps 11-14: A all dash, E all free
    for (int i = 11; i <= 14; ++i)
        cs.a(i) = wc_uniform(BC_DASH);

    // Steps 15-20: E all free (wider active region for t=18)
    // (already '?' for E)

    // Steps 15-20: A transitions
    // Step 15-20 A likely has some conditions (search will determine)
    // For now: A free in active, dash outside
    for (int i = 21; i < 38; ++i)
        cs.a(i) = wc_uniform(BC_DASH);
    for (int i = 21; i < 38; ++i)
        cs.e(i) = wc_uniform(BC_DASH);

    // --- Message conditions ∇W ---
    // Steps 0-6: no difference
    for (int i = 0; i <= 6; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Step 7: active. Seed one 'x' at bit 16 to force non-trivial
    // differential (prevents trivial all-dash solution).
    cs.w(7) = WordCond::from_string("???????????????x????????????????");

    // Step 8: active, all free
    // cs.w(8) stays '?'

    // Step 9: no difference
    cs.w(9) = wc_uniform(BC_DASH);

    // Step 10: active, all free
    // cs.w(10) stays '?'

    // Steps 11-14: no difference
    for (int i = 11; i <= 14; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Step 15: active, all free
    // cs.w(15) stays '?'

    // Steps 16-22: no difference
    for (int i = 16; i <= 22; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Step 23: active, all free
    // cs.w(23) stays '?'

    // Step 24: active, all free
    // cs.w(24) stays '?'

    // Steps 25-37: no difference
    for (int i = 25; i <= 37; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    return cs;
}

} // namespace mendel2013
