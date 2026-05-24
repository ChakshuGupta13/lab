// Full-characteristic starting points for Mendel-Nad-Schlaffer 2013
// (EUROCRYPT 2013, "Improving Local Collisions: New Attacks on Reduced SHA-256").
//
// !!! IMPORTANT !!!
// Encode the dense Table B (full characteristic, with u/n/0/1 conditions)
// rather than the sparse Table A (mostly ?, the search-tool input). For SAT
// reproduction, this is the difference between TIMEOUT and seconds.
// The starting points in sha256_starting_points.hpp transcribe each
// paper's *sparse* Table 1 (input to the paper's GnD/Phase-1 search). When
// fed to a SAT encoder those typically TIMEOUT.
//
// This file transcribes the paper's *dense* Tables 3 / 4 / 5 — the fully
// solved characteristics produced by the paper's tool — for use as SAT
// starting points. Empirically the dense Table B
// substitution reduces SAT runtime from TIMEOUT to seconds.
//
// Conventions (u/n bit pairing):
//   This paper uses the codebase's `u`/`n` convention directly:
//     u = (m_bit = 1, m*_bit = 0)
//     n = (m_bit = 0, m*_bit = 1)
//   Verified 17 May 2026: Table 3 conditions match Table 6 example collision
//   bit-for-bit with NO swap, 0 mismatches. (Differs from Mendel 2011 which
//   needed a u<->n swap.)
//
// Bit ordering: MSB-first (leftmost char = bit 31, rightmost = bit 0),
// matching WordCond::from_string().
//
// IV / encoding notes:
//   - starting_point_28_full(): Table 6 uses standard SHA-256 IV (real
//     collision). Call enc.encode_iv(sha256::IV) when encoding.
//   - starting_point_31_full(): Table 7 is a semi-free-start collision
//     ("with the last 5 chaining variables set to 0"). Skip encode_iv().
//   - starting_point_38_sfs_full(): Table 8 is a semi-free-start collision.
//     Skip encode_iv().
//
// Verification: a Python script (not shipped) was used at transcription time
// to check each starting point against its corresponding example collision
// (paper Tables 6, 7, 8).
//
// Sources:
//   Mendel-Nad-Schläffer 2013 (EUROCRYPT 2013) Tables 3, 4, 5 (characteristics)
//                                                Tables 6, 7, 8 (example collisions).

#pragma once

#include "search.hpp"
#include "sha256_starting_points.hpp"

namespace mendel2013 {

// ============================================================
// Table 3 — fully solved characteristic for 28-step real collision
// ============================================================
//
// Local collision span: t=11 steps (8..18). Active W words: W_8, W_9,
// W_13, W_14, W_16, W_18. Conditions on E_5..E_16 plus A_6..A_12.
//
// Standard SHA-256 IV (h_0 same as in mendel2011 Table 8).
//
// Verified vs Table 6 example collision: 0 mismatches.
inline CharState starting_point_28_full() {
    CharState cs = starting_point_28();

    // step 6
    cs.a(6)  = WordCond::from_string("------------------------1-------");
    cs.e(6)  = WordCond::from_string("-------------0----------00------");
    // step 7
    cs.a(7)  = WordCond::from_string("------------------------1-------");
    cs.e(7)  = WordCond::from_string("----------1--1-10-------11--1---");
    // step 8
    cs.a(8)  = WordCond::from_string("-0uu-nuu-u-uu-nnn-uu---nu-n-----");
    cs.e(8)  = WordCond::from_string("-0uu-1nu-uun-u-0uu-n0---nnu11---");
    cs.w(8)  = WordCond::from_string("nnnu----n-nn-u-u-n-n----n-n-----");
    // step 9
    cs.a(9)  = WordCond::from_string("-0-------n--u----------n1-------");
    cs.e(9)  = WordCond::from_string("-1-1011-u1u0--0nn-0-u-n0010uu-0-");
    cs.w(9)  = WordCond::from_string("11--un1-un0nn--110-11u01uunnu---");
    // step 10
    cs.a(10) = WordCond::from_string("nnnnnn----u-u--------u--u--uu---");
    cs.e(10) = WordCond::from_string("0n010n1011101011n011u1110n0nu01u");
    // step 11
    cs.a(11) = WordCond::from_string("-1----------------------1-------");
    cs.e(11) = WordCond::from_string("-n-1un1-n111000n10n0110n10001-u0");
    // step 12
    cs.a(12) = WordCond::from_string("-1----------------------1-------");
    cs.e(12) = WordCond::from_string("0011n111n00u0n11u0uu10110uu10-00");
    // step 13
    cs.e(13) = WordCond::from_string("000100010n011nuuuuuuuu1n11011101");
    cs.w(13) = WordCond::from_string("------n--n----------n--1n--n----");
    // step 14
    cs.e(14) = WordCond::from_string("11-00u--0un0u000-00-u0nn-nnnu-0-");
    cs.w(14) = WordCond::from_string("---1----------------------------");
    // step 15
    cs.e(15) = WordCond::from_string("-----1---10-11001011-00--0001---");
    // step 16
    cs.e(16) = WordCond::from_string("-----1---01-1-------0-00-1111---");
    cs.w(16) = WordCond::from_string("1----u---0uun-----10un01uun-n---");
    // step 18 (W only — A and E are all-DASH at this step)
    cs.w(18) = WordCond::from_string("-----n----n-n------1-n01n-n-u---");

    return cs;
}

// ============================================================
// Table 4 — fully solved characteristic for 31-step SFS collision
// ============================================================
//
// Local collision span: t=14 steps (5..18). Active W words: W_5..W_9, W_16, W_18.
// First 5 message words have NO differences (the 2-block-collision freedom).
//
// SFS variant: Table 7 example uses h_0 = 532f13f5 6a28c3c0 e301fab5 0 0 0 0 0
// (chosen by paper to demonstrate the last-5-chaining-words = 0 case).
// When encoding for SAT, SKIP encode_iv() — solver chooses an IV consistent
// with the DASH conditions on A[-4..-1] / E[-4..-1].
//
// Verified vs Table 7 example collision: 0 mismatches (no u<->n swap).
// Completeness audit: A[8] and A[9] are all-DASH in paper but base SP has them
// FREE; explicit overrides added below (gap that the Python verifier cannot
// detect — `?` is trivially satisfied by any bit pair).
inline CharState starting_point_31_full() {
    CharState cs = starting_point_31();

    // step 3
    cs.a(3)  = WordCond::from_string("------------------------------0-");
    cs.e(3)  = WordCond::from_string("-0----------------0-------------");
    // step 4
    cs.a(4)  = WordCond::from_string("------------------------------00");
    cs.e(4)  = WordCond::from_string("-1---------1-----01---1-0--0--10");
    // step 5
    cs.a(5)  = WordCond::from_string("-nnn-n-n-11----n--nu-1-------0n-");
    cs.e(5)  = WordCond::from_string("0nnnn1uu-0-1101n-1nu--0-11-1-0n1");
    cs.w(5)  = WordCond::from_string("u---uunu-------n---n----------n-");
    // step 6
    cs.a(6)  = WordCond::from_string("unnnn-------------------------0-");
    cs.e(6)  = WordCond::from_string("n-n10111n--u11u00n10u1n-nn1n-1uu");
    cs.w(6)  = WordCond::from_string("nn1-n---nu-nn--1u--0-un0--n0-nn-");
    // step 7
    cs.a(7)  = WordCond::from_string("-------------------n--------n-0u");
    cs.e(7)  = WordCond::from_string("101u0nn10-11011u-n111n110un1-nnn");
    cs.w(7)  = WordCond::from_string("00nn0n101-n1nnn1u0nn-n011u-1n0--");
    // step 8
    cs.a(8)  = wc_uniform(BC_DASH);   // paper Table 4 has A[8] all-dash; base SP has FREE (gap fix)
    cs.e(8)  = WordCond::from_string("1-uu11110--0u10110n-10101010-0n0");
    cs.w(8)  = WordCond::from_string("0001u0001-000nuuun1n01nn-01nuuuu");
    // step 9
    cs.a(9)  = wc_uniform(BC_DASH);   // paper Table 4 has A[9] all-dash; base SP has FREE (gap fix)
    cs.e(9)  = WordCond::from_string("101100uu111111nu111001--011110nn");
    cs.w(9)  = WordCond::from_string("-----1---------un---0-----11un--");
    // step 10
    cs.a(10) = WordCond::from_string("----------------u------------u--");
    cs.e(10) = WordCond::from_string("1-00u1101001101un00--0001--u1n00");
    cs.w(10) = WordCond::from_string("---0--------------------------1-");
    // step 11
    cs.e(11) = WordCond::from_string("010100u0nu1uuuuuu1001000000n1u10");
    // step 12
    cs.e(12) = WordCond::from_string("111nuuuuuuuuuuuuu001111101100n00");
    // step 13
    cs.e(13) = WordCond::from_string("---101-11-1-----1----------0-0--");
    // step 14
    cs.e(14) = WordCond::from_string("---100---0011111u-------1----u--");
    // step 15
    cs.e(15) = WordCond::from_string("----------------0------------0--");
    // step 16
    cs.e(16) = WordCond::from_string("----------------1------------1--");
    cs.w(16) = WordCond::from_string("-------------unnnunnnnnnnnnnnn--");
    // step 18 (W only)
    cs.w(18) = WordCond::from_string("----------------n------------n--");

    return cs;
}

// ============================================================
// Table 5 — fully solved characteristic for 38-step SFS collision
// ============================================================
//
// Local collision span: t=18 steps (7..24). The paper's hardest result;
// estimated 8 hours on single CPU.
//
// SFS variant: Table 8 example uses h_0 = ba75b4ac c3c9fd45 fce04f3a ...
// (no constraint on chaining shape — solver picks any consistent IV).
// SKIP encode_iv() when encoding.
//
// Verified vs Table 8 example collision: 0 mismatches (no u<->n swap).
// Completeness audit vs base SP starting_point_38_sfs(): added DASH
// overrides for A[10] and A[17..20] (paper all-dash; base SP has FREE
// in those steps — the same class of gap caught by Adversary on Table 4).
inline CharState starting_point_38_sfs_full() {
    CharState cs = starting_point_38_sfs();

    // step 5
    cs.e(5)  = WordCond::from_string("--------------------1----1------");
    // step 6
    cs.e(6)  = WordCond::from_string("1--00--1-------1-10111---0-1--0-");
    // step 7
    cs.a(7)  = WordCond::from_string("-n----u-n-u---u-------n-nn------");
    cs.e(7)  = WordCond::from_string("nu-11-0uuun101-uuuuuuu1u-u-1--01");
    cs.w(7)  = WordCond::from_string("--nnnnn--nn--un----nuuu-nn------");
    // step 8
    cs.a(8)  = WordCond::from_string("----nn-n--n--un-uu-u-u-u-n------");
    cs.e(8)  = WordCond::from_string("n01nn-1n10000-1u1uuunu00n0-nn0n1");
    cs.w(8)  = WordCond::from_string("0000011011101--1100nuuuuuuuuuu00");
    // step 9
    cs.a(9)  = WordCond::from_string("u---u-n--nuuu---n-uuu--u-------n");
    cs.e(9)  = WordCond::from_string("000n00u10001n101nu0u000111-u11un");
    // step 10
    cs.a(10) = wc_uniform(BC_DASH);   // paper all-dash; base SP has FREE (gap fix)
    cs.e(10) = WordCond::from_string("00unn00110n1001u11u-101u111u0uun");
    cs.w(10) = WordCond::from_string("--------unnnnnnnu---------------");
    // step 11
    cs.e(11) = WordCond::from_string("1u1-11u11-n01n100n10u1-11u100011");
    cs.w(11) = WordCond::from_string("------1-------------------------");
    // step 12
    cs.e(12) = WordCond::from_string("0uu1u1u0u1uu1n01nn111u011n-01010");
    cs.w(12) = WordCond::from_string("----------1-----01--------------");
    // step 13
    cs.e(13) = WordCond::from_string("n0010uu01-00n1-01n0nu10u10-1-nuu");
    cs.w(13) = WordCond::from_string("----------------0--------------1");
    // step 14
    cs.e(14) = WordCond::from_string("101-110-1-0010-10--111-010---100");
    // step 15
    cs.a(15) = WordCond::from_string("----u----------n----------------");
    cs.e(15) = WordCond::from_string("0-1-u01----00--n-00----10----111");
    cs.w(15) = WordCond::from_string("-----------------------------u--");
    // step 16
    cs.a(16) = WordCond::from_string("-----------------------------u--");
    cs.e(16) = WordCond::from_string("----n-n1---01-un11------nuu-nu01");
    // step 17 (paper has A[17] all-dash; base SP FREE — gap fix)
    cs.a(17) = wc_uniform(BC_DASH);
    cs.e(17) = WordCond::from_string("-0--n-1---0nnnnn-nuu1---011-1-un");
    // step 18 (paper has A[18] all-dash; base SP FREE — gap fix)
    cs.a(18) = wc_uniform(BC_DASH);
    cs.e(18) = WordCond::from_string("----0-1---00000--000--1011101100");
    // step 19 (paper has A[19] all-dash; base SP FREE — gap fix)
    cs.a(19) = wc_uniform(BC_DASH);
    cs.e(19) = WordCond::from_string("0---u-00nuuuuuuu0001--0011011011");
    // step 20 (paper has A[20] all-dash; base SP FREE — gap fix)
    cs.a(20) = wc_uniform(BC_DASH);
    cs.e(20) = WordCond::from_string("1--1--11100111---1--0unnnnnnnn0-");
    // step 21
    cs.e(21) = WordCond::from_string("----1---11111111-----000000000--");
    // step 22
    cs.e(22) = WordCond::from_string("---------------------111111111--");
    // step 23 (W only)
    cs.w(23) = WordCond::from_string("----n---------un----------------");
    // step 24 (W only)
    cs.w(24) = WordCond::from_string("-----------------------------n--");

    return cs;
}

} // namespace mendel2013
