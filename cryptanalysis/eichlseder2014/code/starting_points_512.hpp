// Starting points for SHA-512 38-step semi-free-start collision search.
// Encodes the differential characteristic from Eichlseder-Mendel-Schläffer 2014,
// Table 3: full generalized conditions validated against the Table 2 collision.
//
// Table 3 provides 835 pre-determined conditions (0, 1, u, n) beyond the
// structural x/dash encoding. These conditions were derived by the authors
// through their linearized model and nonlinear adjustment procedure.
//
// The starting point encodes:
//   - 'u'/'n': specific difference signs at active (x) positions
//   - '0'/'1': specific values at no-difference (dash) positions
//   - '-': undetermined equal bits (search resolves to 0/1)
//   - 'x': undetermined difference bits (only W[15] bit 0 remains)
//
// Bit ordering: MSB = bit 63, LSB = bit 0.

#pragma once

#include "search_512.hpp"

namespace eichlseder2014 {

// Build a WordCond64 with 'x' at positions where mask=1, '-' elsewhere.
// BC_DASH: allow (0,0) and (1,1)  → allow[0]=1, allow[1]=0, allow[2]=0, allow[3]=1
// BC_X:    allow (1,0) and (0,1)  → allow[0]=0, allow[1]=1, allow[2]=1, allow[3]=0
inline WordCond64 wc_from_x_mask(uint64_t mask) {
    WordCond64 wc;
    wc.allow[0] = ~mask;  // (0,0): dash bits
    wc.allow[1] =  mask;  // (1,0): x bits
    wc.allow[2] =  mask;  // (0,1): x bits
    wc.allow[3] = ~mask;  // (1,1): dash bits
    return wc;
}

// ============================================================
// 38-step SHA-512 semi-free-start collision starting point
// ============================================================
//
// Local collision: 18 steps (steps 7–24), differences in 6 message words:
//   W₇ (full word), W₈ (3 bits), W₁₀ (12 bits), W₁₅ (1 bit),
//   W₂₃ (8 bits, expansion-derived), W₂₄ (1 bit, expansion-derived).
//
// Cancellation conditions in message expansion:
//   ΔW₁₇ = 0: requires σ₁(ΔW₁₅) + ΔW₁₀ = 0
//   ΔW₂₂ = 0: requires ΔW₁₅ + σ₀(ΔW₇) = 0
//   ΔW₂₅ = 0: requires σ₁(ΔW₂₃) + σ₀(ΔW₁₀) = 0
//   ΔW₂₆ = 0: requires σ₁(ΔW₂₄) + ΔW₁₀ = 0
//   ΔW₃₀ = 0: requires ΔW₂₃ + σ₀(ΔW₁₅) = 0
//   ΔW₃₁ = 0: requires ΔW₂₄ + ΔW₁₅ = 0
//
// These cancellations are enforced by setting the target W words to dash.
// The search engine propagates conditions through the expansion; incompatible
// sign assignments cause contradictions (search backtracks).
//
// State conditions from verified Table 2 collision:
//   A: differences at steps 7–9, 12, 15–16 (40 bits total)
//   E: differences at steps 7–20 (215 bits total)
//   All other steps: zero difference (dash)
//
// The starting point marks A/E difference positions as 'x' (sign unknown)
// and non-difference positions as '-' (equal), derived from the collision's
// XOR differences. This reduces Phase 1 from 1488 to 463 bits.

inline CharState512 starting_point_38_sfs() {
    CharState512 cs(38);

    // --- IV (SFS: free to choose, but equal in both messages) ---
    for (int i = -4; i < 0; ++i) {
        cs.a(i) = wc_uniform_64(BC_DASH);
        cs.e(i) = wc_uniform_64(BC_DASH);
    }

    // --- A register: all steps default to dash, then overlay Table 3 ---
    for (int i = 0; i < 38; ++i)
        cs.a(i) = wc_uniform_64(BC_DASH);

    // Table 3 A conditions (validated against Table 2 collision: 0 conflicts)
    cs.a(7)  = WordCond64::from_string("-------------------------------------------------------------nun");
    cs.a(8)  = WordCond64::from_string("------------uuun-u---------n-----u------n----------------------u");
    cs.a(9)  = WordCond64::from_string("un--n------u---nu----u---n--nn-nuunn---n-nn-n-n-n-u-u--n--------");
    cs.a(12) = WordCond64::from_string("---------------------------------------------------------n------");
    cs.a(15) = WordCond64::from_string("u------u--------------------------------------------------------");
    cs.a(16) = WordCond64::from_string("---------------------------------------------------------------u");

    // --- E register: all steps default to dash, then overlay Table 3 ---
    for (int i = 0; i < 38; ++i)
        cs.e(i) = wc_uniform_64(BC_DASH);

    // Table 3 E conditions (validated against Table 2 collision: 0 conflicts)
    // Steps 5-6: sparse conditions at low bits
    cs.e(5)  = WordCond64::from_string("-------------------------------------------------------------1-0");
    cs.e(6)  = WordCond64::from_string("---------1----1101-------------------------------------------0-1");
    // Steps 7-16: dense conditions (core of local collision)
    cs.e(7)  = WordCond64::from_string("-0-------00--00010---01---00------01--1------------1--1--1---n-u");
    cs.e(8)  = WordCond64::from_string("-100-0-1unnn00nnun0-111-0111---00-111-0uu-1-0-1-00000011-11--u00");
    cs.e(9)  = WordCond64::from_string("-10010-00uu10u00n0101u0--0nn1u11--nn--n0---00-1-001n11uuu11--001");
    cs.e(10) = WordCond64::from_string("-u-u-0-u10nun-1u1uu-01uu-u0u011uu-1n1u1-0nu1n1n-0100n1u10n11-uu0");
    cs.e(11) = WordCond64::from_string("-11110100110011u01011n01110uu0101nun001uu10nn0u0unu111-u01010n0u");
    cs.e(12) = WordCond64::from_string("n1100--0--0u--n0-n0-0n1101uuu11---1--110u01101-100n0110u0n--1unn");
    cs.e(13) = WordCond64::from_string("un--n1101-1u0000nunnn0110n-00u01-10u01n0u-0u1u1uu0u-u100nn0-0011");
    cs.e(14) = WordCond64::from_string("10--0-10----101110111010010100----11--1-11-1-1-11-100--unn---001");
    cs.e(15) = WordCond64::from_string("n1--1-0u---101011010011111110100--00001-00-1-1-00-0-11-0-u--11-1");
    cs.e(16) = WordCond64::from_string("n-010-nu0-01nuuuuuuuuuuuuuuuu100-nuu1110uu111-u010-111-00n-00-un");
    // Steps 17-22: trailing conditions (difference cancellation zone)
    cs.e(17) = WordCond64::from_string("11----01----010--100nuuuuuuunuuuuunuuuu110-11101100010011011111n");
    cs.e(18) = WordCond64::from_string("0-----00----0---------------01111101111-01----1000001010000000nu");
    cs.e(19) = WordCond64::from_string("u----unn--------------------------------11----nuuuuuuuuuuuuuuuuu");
    cs.e(20) = WordCond64::from_string("00001000100010101010-------------1111unnnnnnnnnnnnnnnnnnnnnnnnnn");
    cs.e(21) = WordCond64::from_string("1----111-----------------------------111111111-11111111111111111");
    cs.e(22) = WordCond64::from_string("-------------------------------------000000000000000000000000000");

    // --- Message conditions: W₀..W₁₅ (direct message words) ---
    // W₀..W₆: no difference
    for (int i = 0; i <= 6; ++i)
        cs.w(i) = wc_uniform_64(BC_DASH);

    // Table 3 W conditions (enriched from x-mask to specific signs)
    cs.w(7)  = WordCond64::from_string("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuun");
    cs.w(8)  = WordCond64::from_string("----00-------------------------------------------------------nuu");
    cs.w(9)  = wc_uniform_64(BC_DASH);
    cs.w(10) = WordCond64::from_string("------------nuuuuuu-------------------------------------unnnn---");
    // W₁₁..W₁₂: expansion-derived single-bit conditions from Table 3
    cs.w(11) = WordCond64::from_string("---------------------------------------------------------------1");
    cs.w(12) = WordCond64::from_string("---------------------------------------------------------------0");
    for (int i = 13; i <= 14; ++i)
        cs.w(i) = wc_uniform_64(BC_DASH);
    // W₁₅: 1-bit difference at LSB (sign undetermined in Table 3)
    cs.w(15) = WordCond64::from_string("---------------------------------------------------------------x");

    // W₁₆..W₂₂: zero difference required (cancellation conditions)
    for (int i = 16; i <= 22; ++i)
        cs.w(i) = wc_uniform_64(BC_DASH);

    // W₂₃..W₂₄: expansion-derived active words (enriched signs from Table 3)
    cs.w(23) = WordCond64::from_string("unuuuuuu--------------------------------------------------------");
    cs.w(24) = WordCond64::from_string("---------------------------------------------------------------n");

    // W₂₅..W₃₇: zero difference required (cancellation conditions)
    for (int i = 25; i <= 37; ++i)
        cs.w(i) = wc_uniform_64(BC_DASH);

    // W₃₈..W₇₉: outside 38-step attack scope, left as free ('?')

    return cs;
}

} // namespace eichlseder2014
