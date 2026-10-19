#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define N 256
static const int64_t Q = 8380417, ROOT = 1753;
static int64_t zetas[N];

static int64_t powmod(int64_t b, int64_t e, int64_t m) {
    int64_t r = 1; b %= m;
    while (e > 0) { if (e & 1) r = r * b % m; b = b * b % m; e >>= 1; }
    return r;
}
static int bitrev8(int x) { int r = 0; for (int i = 0; i < 8; i++) { r = (r << 1) | (x & 1); x >>= 1; } return r; }

static void gen_zetas(void) { for (int i = 0; i < N; i++) zetas[i] = powmod(ROOT, bitrev8(i), Q); }

static void ntt(int64_t a[N], const unsigned char *fault) {
    int k = 0;
    for (int len = 128; len > 0; len >>= 1)
        for (int start = 0; start < N; start += 2 * len) {
            ++k;
            int64_t zeta = (fault && fault[k]) ? 0 : zetas[k];
            for (int j = start; j < start + len; ++j) {
                int64_t t = zeta * a[j + len] % Q;
                a[j + len] = ((a[j] - t) % Q + Q) % Q;
                a[j] = (a[j] + t) % Q;
            }
        }
}

static int dk_rank(const unsigned char *fault) {
    static int64_t M[N][N];
    for (int j = 0; j < N; j++) {
        int64_t e0[N] = {0}, e1[N];
        e0[j] = 1; memcpy(e1, e0, sizeof e0);
        ntt(e0, NULL); ntt(e1, fault);
        for (int i = 0; i < N; i++) M[i][j] = ((e0[i] - e1[i]) % Q + Q) % Q;
    }
    int rank = 0;
    for (int col = 0; col < N && rank < N; col++) {
        int piv = -1;
        for (int r = rank; r < N; r++) if (M[r][col]) { piv = r; break; }
        if (piv < 0) continue;
        for (int c = 0; c < N; c++) { int64_t tmp = M[rank][c]; M[rank][c] = M[piv][c]; M[piv][c] = tmp; }
        int64_t inv = powmod(M[rank][col], Q - 2, Q);
        for (int c = 0; c < N; c++) M[rank][c] = M[rank][c] * inv % Q;
        for (int r = 0; r < N; r++)
            if (r != rank && M[r][col]) {
                int64_t f = M[r][col];
                for (int c = 0; c < N; c++) M[r][c] = ((M[r][c] - f * M[rank][c]) % Q + Q) % Q;
            }
        rank++;
    }
    return rank;
}

static int len_of(int k) {
    int len = 128, lo = 1;
    while (len > 0) { if (k >= lo && k < lo + N / (2 * len)) return len; lo += N / (2 * len); len >>= 1; }
    return -1;
}

int main(void) {
    gen_zetas();
    unsigned char none[N + 1] = {0};
    printf("null-fault rank (expect 0): %d\n", dk_rank(none));

    printf("\n== single twiddle fault: rank vs len_l (Theorem 1) ==\n");
    int all_ok = 1;
    int probe[] = {1, 2, 4, 8, 16, 32, 64, 128, 3, 7, 200, 255};
    for (unsigned t = 0; t < sizeof probe / sizeof *probe; t++) {
        int k = probe[t];
        unsigned char f[N + 1] = {0}; f[k] = 1;
        int r = dk_rank(f), L = len_of(k), ok = (r == L);
        all_ok &= ok;
        printf("  k=%3d  layer-len=%3d  rank=%3d  %s\n", k, L, r, ok ? "OK" : "**MISMATCH**");
    }

    printf("\n== one fault per layer (saturating): rank vs N-1 (Theorem 2) ==\n");
    unsigned char sat[N + 1] = {0};
    int firsts[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (unsigned t = 0; t < sizeof firsts / sizeof *firsts; t++) sat[firsts[t]] = 1;
    int rs = dk_rank(sat), oks = (rs == N - 1);
    all_ok &= oks;
    printf("  K={1,2,4,...,128}  rank=%d  expect %d  %s\n", rs, N - 1, oks ? "OK" : "**MISMATCH**");

    printf("\n== two same-layer faults: rank vs 2*len (cor:samelayer) ==\n");
    unsigned char sl[N + 1] = {0}; sl[2] = 1; sl[3] = 1;
    int r2 = dk_rank(sl), ok2 = (r2 == 2 * 64);
    all_ok &= ok2;
    printf("  K={2,3} (layer len 64)  rank=%d  expect %d  %s\n", r2, 2 * 64, ok2 ? "OK" : "**MISMATCH**");

    printf("\n%s\n", all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return all_ok ? 0 : 1;
}
