// Generalized bit conditions for differential cryptanalysis.
// Based on De Cannière & Rechberger 2006, Table II.
// Shared between DC06 (SHA-1) and Mendel 2011 (SHA-256) implementations.
//
// Each bit condition constrains a pair (x_i, x_i*) of corresponding bits
// from two messages.  The condition is a 4-bit mask indicating which of
// the four possible pairs are allowed:
//   bit 0: (0,0)   bit 1: (1,0)   bit 2: (0,1)   bit 3: (1,1)
//
// All 16 possible masks have a named symbol.  The hex value of each
// symbol IS its mask, so operations reduce to bitwise logic.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <random>

// ---- Cross-compiler intrinsics ----
#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX  // prevent <windows.h> min/max macros (pulled by <intrin.h>)
#endif
#include <intrin.h>
inline int gc_popcount32(uint32_t x) { return (int)__popcnt(x); }
inline int gc_popcount64(uint64_t x) { return (int)__popcnt64(x); }
inline int gc_ctz32(uint32_t x) { unsigned long idx; _BitScanForward(&idx, x); return (int)idx; }
inline int gc_ctz64(uint64_t x) { unsigned long idx; _BitScanForward64(&idx, x); return (int)idx; }
#define GC_UNREACHABLE() __assume(false)
#else
inline int gc_popcount32(uint32_t x) { return __builtin_popcount(x); }
inline int gc_popcount64(uint64_t x) { return __builtin_popcountll(x); }
inline int gc_ctz32(uint32_t x) { return __builtin_ctz(x); }
inline int gc_ctz64(uint64_t x) { return __builtin_ctzll(x); }
#define GC_UNREACHABLE() __builtin_unreachable()
#endif

namespace gencond {

// ---- Per-bit condition ----

enum BitCond : uint8_t {
    BC_HASH = 0x0,  // #  contradiction
    BC_0    = 0x1,  // 0  both zero
    BC_U    = 0x2,  // u  x=1, x*=0  (positive diff)
    BC_3    = 0x3,  // 3  x*=0
    BC_N    = 0x4,  // n  x=0, x*=1  (negative diff)
    BC_5    = 0x5,  // 5  x=0
    BC_X    = 0x6,  // x  different
    BC_7    = 0x7,  // 7  not both one
    BC_1    = 0x8,  // 1  both one
    BC_DASH = 0x9,  // -  equal
    BC_A    = 0xA,  // A  x=1
    BC_B    = 0xB,  // B  not(n)
    BC_C    = 0xC,  // C  x*=1
    BC_D    = 0xD,  // D  not(u)
    BC_E    = 0xE,  // E  not both zero
    BC_FREE = 0xF,  // ?  free
};

inline constexpr BitCond bc_meet(BitCond a, BitCond b) {
    return BitCond(a & b);
}
inline constexpr bool bc_is_contradiction(BitCond c) { return c == BC_HASH; }
inline constexpr bool bc_is_free(BitCond c) { return c == BC_FREE; }
inline constexpr BitCond bc_complement(BitCond c) { return BitCond(0xF ^ c); }

// Whether condition allows equal pairs only (no difference)
inline constexpr bool bc_is_equal(BitCond c) {
    return (c & 0x6) == 0;  // bits 1,2 (the diff pairs) are clear
}

inline char bc_to_char(BitCond c) {
    constexpr char tbl[16] = {
        '#','0','u','3','n','5','x','7',
        '1','-','A','B','C','D','E','?'
    };
    return tbl[c & 0xF];
}

inline BitCond bc_from_char(char ch) {
    switch (ch) {
        case '#': return BC_HASH; case '0': return BC_0;
        case 'u': return BC_U;   case '3': return BC_3;
        case 'n': return BC_N;   case '5': return BC_5;
        case 'x': return BC_X;   case '7': return BC_7;
        case '1': return BC_1;   case '-': return BC_DASH;
        case 'A': return BC_A;   case 'B': return BC_B;
        case 'C': return BC_C;   case 'D': return BC_D;
        case 'E': return BC_E;   case '?': return BC_FREE;
        default:  return BC_HASH;
    }
}

// Create condition from an observed pair of bit values (0 or 1).
inline constexpr BitCond bc_from_pair(int x, int xstar) {
    return BitCond(1u << (x + 2 * xstar));
}

// Number of allowed pairs (0 = contradiction, 4 = free).
inline constexpr int bc_popcount(BitCond c) {
    // Lookup table for 4-bit popcount (0..15). Avoids non-constexpr intrinsics.
    constexpr int T[] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
    return T[c & 0xF];
}

// Whether exactly one pair is allowed (fully determined).
inline constexpr bool bc_is_fixed(BitCond c) {
    return bc_popcount(c) == 1;
}

// Whether the condition is '-' (equal, allows (0,0) and (1,1)).
inline constexpr bool bc_is_dash(BitCond c) { return c == BC_DASH; }

// Whether the condition is 'x' (unequal, allows (1,0) and (0,1)).
inline constexpr bool bc_is_x(BitCond c) { return c == BC_X; }


// ---- Per-word condition (parameterized by word type W) ----
//
// Bit-sliced representation: four W masks, one per pair type.
// allow[k] bit i = 1 means position i permits pair k.
// W = uint32_t for SHA-256/SHA-1, uint64_t for SHA-512.

template<typename W = uint32_t>
struct WordCondT {
    static constexpr int BITS = sizeof(W) * 8;
    W allow[4];  // [0]=(0,0)  [1]=(1,0)  [2]=(0,1)  [3]=(1,1)

    // --- Popcount helper (dispatches 32-bit vs 64-bit) ---
    static int popcnt(W mask) {
        if constexpr (sizeof(W) <= 4)
            return gc_popcount32(static_cast<uint32_t>(mask));
        else
            return gc_popcount64(static_cast<uint64_t>(mask));
    }

    // --- Factories ---

    static WordCondT free() {
        W m = ~W(0);
        return {{m, m, m, m}};
    }
    static WordCondT contradiction() {
        return {{0, 0, 0, 0}};
    }

    // From a concrete pair of values (strongest possible conditions).
    static WordCondT from_pair(W val, W val_star) {
        return {{
            W(~val & ~val_star),   // (0,0)
            W( val & ~val_star),   // (1,0)
            W(~val &  val_star),   // (0,1)
            W( val &  val_star),   // (1,1)
        }};
    }

    // Parse from a BITS-character string (MSB first, LSB last).
    static WordCondT from_string(const char* s) {
        WordCondT wc = free();
        int len = static_cast<int>(strlen(s));
        for (int i = 0; i < BITS && i < len; ++i)
            wc.set(BITS - 1 - i, bc_from_char(s[i]));
        return wc;
    }

    // --- Accessors ---

    BitCond get(int b) const {
        return BitCond(
            ((allow[0] >> b) & 1)       |
            (((allow[1] >> b) & 1) << 1) |
            (((allow[2] >> b) & 1) << 2) |
            (((allow[3] >> b) & 1) << 3)
        );
    }

    void set(int b, BitCond c) {
        W mask = ~(W(1) << b);
        for (int k = 0; k < 4; ++k)
            allow[k] = (allow[k] & mask) | (W((c >> k) & 1) << b);
    }

    // --- Operations ---

    // Meet (intersection): tighten conditions.
    WordCondT meet(const WordCondT& o) const {
        return {{
            W(allow[0] & o.allow[0]), W(allow[1] & o.allow[1]),
            W(allow[2] & o.allow[2]), W(allow[3] & o.allow[3])
        }};
    }

    // Positions with no allowed pair (contradiction).
    W contradiction_mask() const {
        return ~(allow[0] | allow[1] | allow[2] | allow[3]);
    }
    bool is_consistent() const { return contradiction_mask() == 0; }

    // Positions where all four pairs are allowed (?).
    W free_mask() const {
        return allow[0] & allow[1] & allow[2] & allow[3];
    }

    // Number of non-free (constrained) positions.
    int hw() const { return popcnt(~free_mask()); }

    // Positions that MUST differ (only (1,0) or (0,1) allowed, never (0,0)/(1,1)).
    W must_diff_mask() const {
        return ~(allow[0] | allow[3]) & (allow[1] | allow[2]);
    }

    // Positions where at least one differing pair is allowed.
    W may_diff_mask() const { return allow[1] | allow[2]; }

    // Check whether a concrete pair (val, val*) conforms to this condition.
    bool conforms(W val, W val_star) const {
        WordCondT actual = from_pair(val, val_star);
        for (int k = 0; k < 4; ++k)
            if (actual.allow[k] & ~allow[k]) return false;
        return true;
    }

    bool operator==(const WordCondT& o) const {
        return allow[0]==o.allow[0] && allow[1]==o.allow[1] &&
               allow[2]==o.allow[2] && allow[3]==o.allow[3];
    }
    bool operator!=(const WordCondT& o) const { return !(*this == o); }

    // --- Counting ---

    int count_free() const { return popcnt(free_mask()); }

    // Positions with condition '-' (equal: allows only (0,0) and (1,1)).
    W dash_mask() const {
        return allow[0] & ~allow[1] & ~allow[2] & allow[3];
    }
    int count_dash() const { return popcnt(dash_mask()); }

    // Positions with condition 'x' (unequal: allows only (1,0) and (0,1)).
    W x_mask() const {
        return ~allow[0] & allow[1] & allow[2] & ~allow[3];
    }
    int count_x() const { return popcnt(x_mask()); }

    // Total number of constrained (non-free) positions.
    int count_constrained() const { return BITS - count_free(); }

    // --- Impose / refine ---

    // Impose condition c on bit b. Returns false if contradiction results.
    bool impose(int b, BitCond c) {
        BitCond cur = get(b);
        BitCond refined = bc_meet(cur, c);
        set(b, refined);
        return !bc_is_contradiction(refined);
    }

    // Meet with another WordCondT, returns false if any bit contradicts.
    bool impose_word(const WordCondT& o) {
        for (int k = 0; k < 4; ++k)
            allow[k] &= o.allow[k];
        return is_consistent();
    }

    // --- Sampling ---

    // Generate a random conforming (val, val*) pair.
    // Precondition: is_consistent() (no contradicted bits).
    template <typename RNG>
    std::pair<W, W> sample(RNG& rng) const {
        W val = 0, val_star = 0;
        for (int b = 0; b < BITS; ++b) {
            BitCond c = get(b);
            // Collect allowed pairs
            int pairs[4], np = 0;
            for (int k = 0; k < 4; ++k)
                if ((c >> k) & 1) pairs[np++] = k;
            // Pick one uniformly at random
            int pick = pairs[std::uniform_int_distribution<int>(0, np - 1)(rng)];
            val |= W(pick & 1) << b;
            val_star |= W((pick >> 1) & 1) << b;
        }
        return {val, val_star};
    }

    // --- Formatting ---

    // BITS-character string, MSB first.
    std::string to_string() const {
        std::string s(BITS, '?');
        for (int b = BITS - 1; b >= 0; --b)
            s[BITS - 1 - b] = bc_to_char(get(b));
        return s;
    }

    void print(const char* label = nullptr) const {
        if (label) std::printf("%s", label);
        std::string s = to_string();
        std::printf("%s", s.c_str());
    }
};

// Backward-compatible alias: SHA-256 / SHA-1 use 32-bit words.
using WordCond = WordCondT<uint32_t>;
// SHA-512: 64-bit words.
using WordCond64 = WordCondT<uint64_t>;

} // namespace gencond
