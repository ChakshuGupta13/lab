// Starting points for SHA-256 differential characteristic search.
// Encodes Tables 2 and 6 from Mendel-Nad-Schläffer 2011.
//
// Each starting point defines conditions on ∇A, ∇E, ∇W using the
// alternative SHA-256 description (§4.1) and generalized conditions (§4.2).
//
// Symbols: '-' = equal (no diff), '?' = free, 'x' = unequal (diff, sign unknown).
// The search engine resolves '?' → '-'/'x' and 'x' → 'u'/'n'.
//
// Bit ordering: MSB-first (leftmost char = bit 31, rightmost = bit 0),
// matching WordCond::from_string().

#pragma once

#include "search.hpp"

namespace mendel2011 {

// ============================================================
// Table 2: Starting point for 32-step semi-free-start collision
// ============================================================
//
// Active region: steps 2–17.  IV is free (SFS attack).
// Single-bit primary difference at bit 2 in step 2 (A, E, W).
// State diff propagates through steps 3–13, message expansion
// produces two x bits in W[17] at bits 27 and 16.

inline CharState starting_point_32_sfs() {
    CharState cs(32);

    // --- IV: steps -4..-1, both A and E are free (SFS) ---
    // CharState default is all '?', which is correct for SFS.
    // The paper shows '-' for IV in the starting point because
    // both messages share the same (chosen) IV → no difference.
    // For SFS: IV is free to choose but equal in both messages.
    for (int i = -4; i < 0; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // --- State conditions ∇A and ∇E ---
    // Steps 0-1: no difference
    cs.a(0)  = wc_uniform(BC_DASH);
    cs.a(1)  = wc_uniform(BC_DASH);
    cs.e(0)  = wc_uniform(BC_DASH);
    cs.e(1)  = wc_uniform(BC_DASH);

    // Step 2: single x at bit 2
    cs.a(2)  = WordCond::from_string("-----------------------------x--");
    cs.e(2)  = WordCond::from_string("-----------------------------x--");

    // Steps 3-6: all free (active region, full search)
    // (already '?' from constructor)

    // Step 7: A upper 16 free, lower 16 dash; E all free
    cs.a(7)  = WordCond::from_string("????????????????----------------");
    // cs.e(7) stays '?'

    // Step 8: A upper 16 free, lower 16 dash; E all free
    cs.a(8)  = WordCond::from_string("????????????????----------------");
    // cs.e(8) stays '?'

    // Step 9: A upper 15 free, x at bit 16, lower 16 dash; E all free
    cs.a(9)  = WordCond::from_string("???????????????x----------------");
    // cs.e(9) stays '?'

    // Step 10: A all dash; E all free
    cs.a(10) = wc_uniform(BC_DASH);
    // cs.e(10) stays '?'

    // Step 11: A all dash; E upper 16 free, lower 16 dash
    cs.a(11) = wc_uniform(BC_DASH);
    cs.e(11) = WordCond::from_string("????????????????----------------");

    // Step 12: A all dash; E upper 16 free, lower 16 dash
    cs.a(12) = wc_uniform(BC_DASH);
    cs.e(12) = WordCond::from_string("????????????????----------------");

    // Step 13: A all dash; E upper 15 free, x at bit 16, lower 16 dash
    cs.a(13) = wc_uniform(BC_DASH);
    cs.e(13) = WordCond::from_string("???????????????x----------------");

    // Steps 14-31: all dash (zero diff)
    for (int i = 14; i < 32; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // --- Message conditions ∇W ---
    // Steps 0-1: no difference
    cs.w(0)  = wc_uniform(BC_DASH);
    cs.w(1)  = wc_uniform(BC_DASH);

    // Step 2: single x at bit 2
    cs.w(2)  = WordCond::from_string("-----------------------------x--");

    // Steps 3-7: all free
    // (already '?' from constructor)

    // Step 8: upper 15 free, x at bit 16, lower 16 dash
    cs.w(8)  = WordCond::from_string("???????????????x----------------");

    // Steps 9-16: all dash
    for (int i = 9; i <= 16; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Step 17: two x bits at positions 27 and 16
    cs.w(17) = WordCond::from_string("----x----------x----------------");

    // Steps 18-31: all dash (within the 32-step attack scope)
    for (int i = 18; i <= 31; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Steps 32-63: free (outside attack scope; message expansion from
    // the x bits in W[17] may produce nonzero diffs here)
    // Left as '?' (default) — not constrained.

    return cs;
}

// ============================================================
// Table 6: Starting point for 27-step collision (standard IV)
// ============================================================
//
// Active region: steps 7–17.  IV is standard SHA-256 IV.
// Single-bit primary difference at bit 2 in W[7].
// State diff at steps 7–13 (A: 7–9, E: 7–13).
// Message diff at W[7], W[8], W[12], W[15], W[17].
//
// NOTE: The original TXT extraction garbled W[7] (wrong char count).
// Fresh pdftotext extraction gives "?????????????????????????????x??"
// (x at bit 2, bits 0-1 free). Known collision has XOR=0 at bits 0-1,
// so both '?' and '-' would be compatible — but the paper specifies '?'.

inline CharState starting_point_27() {
    CharState cs(27);

    // --- IV: standard SHA-256 IV, fully determined (no difference) ---
    // Alt-description mapping: A_{-1}=a=IV[0], A_{-2}=b=IV[1],
    //   A_{-3}=c=IV[2], A_{-4}=d=IV[3],
    //   E_{-1}=e=IV[4], E_{-2}=f=IV[5], E_{-3}=g=IV[6], E_{-4}=h=IV[7]
    cs.a(-1) = wc_from_constant(sha256::IV[0]);  // a = 6a09e667
    cs.a(-2) = wc_from_constant(sha256::IV[1]);  // b = bb67ae85
    cs.a(-3) = wc_from_constant(sha256::IV[2]);  // c = 3c6ef372
    cs.a(-4) = wc_from_constant(sha256::IV[3]);  // d = a54ff53a
    cs.e(-1) = wc_from_constant(sha256::IV[4]);  // e = 510e527f
    cs.e(-2) = wc_from_constant(sha256::IV[5]);  // f = 9b05688c
    cs.e(-3) = wc_from_constant(sha256::IV[6]);  // g = 1f83d9ab
    cs.e(-4) = wc_from_constant(sha256::IV[7]);  // h = 5be0cd19

    // --- State conditions ∇A and ∇E ---
    // Steps 0-6: no difference
    for (int i = 0; i <= 6; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // Steps 7-9: A all free, E all free
    // (already '?' from constructor)

    // Step 10: A all dash, E all free
    cs.a(10) = wc_uniform(BC_DASH);
    // cs.e(10) stays '?'

    // Step 11: A all dash, E all free
    cs.a(11) = wc_uniform(BC_DASH);
    // cs.e(11) stays '?'

    // Step 12: A all dash, E all free
    cs.a(12) = wc_uniform(BC_DASH);
    // cs.e(12) stays '?'

    // Step 13: A all dash, E all free
    cs.a(13) = wc_uniform(BC_DASH);
    // cs.e(13) stays '?'

    // Steps 14-26: all dash (zero diff)
    for (int i = 14; i < 27; ++i) {
        cs.a(i) = wc_uniform(BC_DASH);
        cs.e(i) = wc_uniform(BC_DASH);
    }

    // --- Message conditions ∇W ---
    // Steps 0-6: no difference
    for (int i = 0; i <= 6; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Step 7: all free except x at bit 2 (primary diff)
    cs.w(7)  = WordCond::from_string("?????????????????????????????x??");

    // Step 8: all free
    // (already '?' from constructor)

    // Step 9: no difference
    cs.w(9)  = wc_uniform(BC_DASH);

    // Steps 10-11: no difference
    cs.w(10) = wc_uniform(BC_DASH);
    cs.w(11) = wc_uniform(BC_DASH);

    // Step 12: all free
    // (already '?' from constructor)

    // Step 13: no difference
    cs.w(13) = wc_uniform(BC_DASH);

    // Step 14: no difference
    cs.w(14) = wc_uniform(BC_DASH);

    // Step 15: all free
    // (already '?' from constructor)

    // Step 16: no difference
    cs.w(16) = wc_uniform(BC_DASH);

    // Step 17: all free
    // (already '?' from constructor)

    // Steps 18-26: no difference (within 27-step scope)
    for (int i = 18; i <= 26; ++i)
        cs.w(i) = wc_uniform(BC_DASH);

    // Steps 27-63: free (outside attack scope; message expansion from
    // the active W words may produce nonzero diffs here)
    // Left as '?' (default) — not constrained.

    return cs;
}

// Table 7 — fully solved characteristic for 27-step collision
// (Mendel-Nad-Schlaffer 2011, Appendix A).
//
// Table 6 (starting_point_27) is the sparse SEARCH START handed to the GnD
// algorithm; Table 7 is its OUTPUT — a fully-determined characteristic with
// signed-bit conditions (u/n), value conditions (0/1), and equality (-).
// Using Table 7 as the SAT starting point skips Phase 1 entirely and lets
// CaDiCaL focus on finding satisfying message bits.
//
// Empirically (May 2026): substituting Table 7 for Table 6 breaks the
// 7-fixed message-word wall — even fully free messages (n_fixed=0) solve
// in ~40s, where Table 6 timed out at every n_fixed <= 7.
inline CharState starting_point_27_table7() {
    CharState cs = starting_point_27();

    // Bit position (MSB first):  3322222222221111111111
    //                            10987654321098765432109876543210
    cs.e(5)  = WordCond::from_string("----------------1--------1------");
    cs.e(6)  = WordCond::from_string("-1--------0--0-10-1----0-0------");
    cs.a(7)  = WordCond::from_string("-------unn--u------n---nn-uuuu--");
    cs.e(7)  = WordCond::from_string("101-11---u10u1-0nuu-uuuu1n---n0-");
    cs.w(7)  = WordCond::from_string("00---1--un-0u-nuuuuu1-nu0n101n--");
    cs.a(8)  = WordCond::from_string("nnnnn-nnnn--------nuu-----------");
    cs.e(8)  = WordCond::from_string("0n0n001001u-1u1n01un010n01n00110");
    cs.w(8)  = WordCond::from_string("-----u--n---n---------nn--------");
    cs.a(9)  = WordCond::from_string("----un--n--nu-------nu-u--------");
    cs.e(9)  = WordCond::from_string("-1n1n1011u011100nn100u10-10000u-");
    cs.e(10) = WordCond::from_string("u00000nuuu10uun01u00n00n110-u-u1");
    cs.e(11) = WordCond::from_string("0n000uuuuu01010111n-uun01n000n01");
    cs.e(12) = WordCond::from_string("01---1010u01u----111-010-0--110-");
    cs.w(12) = WordCond::from_string("------110-u-------n0--u--n-n--nn");
    cs.e(13) = WordCond::from_string("01-10u1nunuuu---1110-1nn11---01-");
    cs.e(14) = WordCond::from_string("-----1-01011----------00--------");
    cs.e(15) = WordCond::from_string("-----1-001000---------11--------");
    cs.w(15) = WordCond::from_string("0u1-nn-n-u-1u---11un0uu10u101u0-");
    cs.w(17) = WordCond::from_string("---0-1nnn---u-1-----10uu0-------");

    return cs;
}

// Table 3 — fully solved characteristic for 32-step SFS collision
// (Mendel-Nad-Schläffer 2011, Appendix A).
//
// Hybrid of paper Table 3 (full characteristic) and Table 2 (sparse SP):
// - All 16 A/E rows from Table 3 (with u/n/0/1 conditions on active region).
// - W upper-16 bits from Table 3; W lower-16 bits in W[3..7] left as '?'.
//   Reason: paper Table 3 marks 19 of those positions as '-' (equal) but
//   the paper's own Table 4 example collision has those bits differing
//   (verified vs Table 4 in Python, 17 May 2026). Following Table 2's
//   convention for those positions avoids the paper-internal contradiction.
//
// Empirically (May 2026): SAT solves in 5.6 s with this SP; paper estimated
// the same attack at "a few days on a cluster with 32 nodes".
//
// SFS usage: when encoding, SKIP encode_iv() (IV is solver-chosen for SFS;
// equality between the two copies is enforced by the SP's DASH conditions
// on A[-4..-1] / E[-4..-1]).
inline CharState starting_point_32_sfs_table3() {
    CharState cs = starting_point_32_sfs();

    // A/E rows from paper Table 3
    cs.e(0)  = WordCond::from_string("-----------------------------0--");
    cs.a(1)  = WordCond::from_string("--------------------------1-----");
    cs.e(1)  = WordCond::from_string("--0-0---1--1----1-0-0-------011-");
    cs.a(2)  = WordCond::from_string("--------------------------0--u--");
    cs.e(2)  = WordCond::from_string("--1-1-1000-1--11101101---1--1u0-");
    cs.w(2)  = WordCond::from_string("-----------------------------u--");

    cs.a(3)  = WordCond::from_string("10n10nnn1n0n-11n1u01u11000uu0n0n");
    cs.e(3)  = WordCond::from_string("-1n1n10un0un101-n1n1n0110un0u0n0");
    cs.w(3)  = WordCond::from_string("uu-un-----un---n????????????????");

    cs.a(4)  = WordCond::from_string("-----n----------0---------0-1---");
    cs.e(4)  = WordCond::from_string("-0n0n1nuuun0-1u1unnnuu011n000nn1");
    cs.w(4)  = WordCond::from_string("1n---1u--uu1u-uu????????????????");

    cs.a(5)  = WordCond::from_string("----------------n---------1-----");
    cs.e(5)  = WordCond::from_string("0u1nn1n-1010-00001u0101-11101110");
    cs.w(5)  = WordCond::from_string("01-1-un0-1-1n-nn????????????????");

    cs.a(6)  = WordCond::from_string("-------------n--u--------u---n--");
    cs.e(6)  = WordCond::from_string("00u01un0000000n111u00100101uu11u");
    cs.w(6)  = WordCond::from_string("n----nnuu-n-nu---???????????????");

    cs.a(7)  = wc_uniform(BC_DASH);
    cs.e(7)  = WordCond::from_string("-n10u000u1un0101nn10n00001n000u1");
    cs.w(7)  = WordCond::from_string("1n0001un10u0nnn-????????????????");

    cs.a(8)  = wc_uniform(BC_DASH);
    cs.e(8)  = WordCond::from_string("-10-1n0-0--1-01-0-1-0----n011-10");
    cs.w(8)  = WordCond::from_string("----u-------unnn----------------");

    cs.a(9)  = WordCond::from_string("----u----------u----------------");
    cs.e(9)  = WordCond::from_string("-0--u00-1-01-1--1---1----n1---0-");

    cs.e(10) = WordCond::from_string("---nunn------n--n--------u-u-u--");
    cs.e(11) = WordCond::from_string("---0-10----100--0--------1-0-0--");
    cs.e(12) = WordCond::from_string("---0011----011--1--------0-1-1--");
    cs.e(13) = WordCond::from_string("---un------unnnn----------------");
    cs.e(14) = WordCond::from_string("---00------00000----------------");
    cs.e(15) = WordCond::from_string("---11------11111----------------");

    cs.w(17) = WordCond::from_string("----n----------n----------------");

    return cs;
}

} // namespace mendel2011
