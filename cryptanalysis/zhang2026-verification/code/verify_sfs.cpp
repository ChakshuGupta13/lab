// Verify the SFS collision pairs from Zhang et al. 2026, Tables 9 and 11.
// Table 9: 37-step SFS collision pair
// Table 11: 36-step SFS collision pair

#include "../../../common/sha256.hpp"
#include <cstdio>
#include <cstring>

using namespace sha256;

// Reduced-step compression: steps 0..r-1, then feedforward.
static State compress_reduced(const State& cv, const Word* msg, int r) {
    Word W[64];
    for (int i = 0; i < 16; ++i) W[i] = msg[i];
    for (int i = 16; i < r; ++i)
        W[i] = sigma_1(W[i-2]) + W[i-7] + sigma_0(W[i-15]) + W[i-16];

    State s = cv;
    compress_range(s, 0, r - 1, W);

    // feedforward
    for (int i = 0; i < 8; ++i) s[i] += cv[i];
    return s;
}

static void print8(const char* label, const State& s) {
    printf("%s: ", label);
    for (int i = 0; i < 8; ++i) printf("%08x ", s[i]);
    printf("\n");
}

int main() {
    // === Table 9: 37-step SFS collision ===
    printf("=== Table 9: 37-step SFS collision ===\n");
    State cv37 = {0x0f4252be, 0xe2d5d87a, 0x56b40fde, 0x6ab7e678,
                  0x3dfb0dcf, 0x6ac73c9d, 0x587ec5f5, 0x67bbd7dc};

    Word M37[16] = {0x5a69a8e4, 0x45ff466e, 0xafec8126, 0x2d74afe7,
                    0x54780c76, 0x94b9dae7, 0x675ce76b, 0x107ffeb9,
                    0xbe7baa67, 0x2653bae8, 0x45b576c8, 0x0de40fc1,
                    0x2d9ea187, 0x26b93c1b, 0x31f1ac39, 0x24de0094};

    Word M37p[16] = {0x5a69a8e4, 0x45ff466e, 0xafec8126, 0x2d74afe7,
                     0x54780c76, 0x94b9dae7, 0x873ee7a6, 0x108002b9,
                     0xbe7baa67, 0x28d3bae9, 0x45b576c8, 0x0de40fc1,
                     0x2d9ea187, 0x26b93c1b, 0x31f1a839, 0x24de0094};

    State expected37 = {0xd08a6613, 0x7d52c71f, 0xe2594bf6, 0x231a4580,
                        0x76c30bc4, 0xd0bec97d, 0x1c832e47, 0x70298870};

    State h37  = compress_reduced(cv37, M37,  37);
    State h37p = compress_reduced(cv37, M37p, 37);

    print8("CV    ", cv37);
    print8("H(M)  ", h37);
    print8("H(M') ", h37p);
    print8("Expect", expected37);

    bool match37 = (h37 == h37p);
    bool correct37 = (h37 == expected37);
    printf("M != M': ");
    bool diff37 = false;
    for (int i = 0; i < 16; ++i) if (M37[i] != M37p[i]) { diff37 = true; break; }
    printf("%s\n", diff37 ? "YES" : "NO (identical!)");
    printf("H(M) == H(M'): %s\n", match37 ? "YES (COLLISION!)" : "NO");
    printf("H(M) == expected: %s\n", correct37 ? "YES" : "NO");

    // Show which words differ
    printf("Differing words: ");
    for (int i = 0; i < 16; ++i)
        if (M37[i] != M37p[i]) printf("W%d ", i);
    printf("\n\n");

    // === Table 11: 36-step SFS collision ===
    printf("=== Table 11: 36-step SFS collision ===\n");
    State cv36 = {0xe66d2bd8, 0xe3edba21, 0xba127056, 0x6fee2e22,
                  0xfa0bde22, 0x579c8cad, 0x28035798, 0xcc37783b};

    Word M36[16] = {0x723e142c, 0x707ad9d2, 0x1dab79ad, 0x44273be9,
                    0xed39e9ee, 0x7ff9c7c2, 0x40d1026e, 0x1bf9c78c,
                    0x82230057, 0x311511c4, 0x1bc6e51b, 0xded9f32d,
                    0x956ed153, 0x4b561ffe, 0x2d929a18, 0xdde8a8ab};

    Word M36p[16] = {0x723e142c, 0x707ad9d2, 0x1dab79ad, 0x44273be9,
                     0xed39e9ee, 0x8001bfc2, 0x40d1026e, 0x89eda209,
                     0x9a224047, 0x311511c4, 0x1bc6e51b, 0xded9f32d,
                     0x956ed153, 0x4b55e000, 0x2d929a18, 0xdde8a8ab};

    State expected36 = {0xe43637a7, 0xa7ed5162, 0x9a328c44, 0xd9e5591e,
                        0x3a71407d, 0xfae30a7e, 0x66c15de7, 0x9daa5143};

    State h36  = compress_reduced(cv36, M36,  36);
    State h36p = compress_reduced(cv36, M36p, 36);

    print8("CV    ", cv36);
    print8("H(M)  ", h36);
    print8("H(M') ", h36p);
    print8("Expect", expected36);

    bool match36 = (h36 == h36p);
    bool correct36 = (h36 == expected36);
    printf("M != M': ");
    bool diff36 = false;
    for (int i = 0; i < 16; ++i) if (M36[i] != M36p[i]) { diff36 = true; break; }
    printf("%s\n", diff36 ? "YES" : "NO (identical!)");
    printf("H(M) == H(M'): %s\n", match36 ? "YES (COLLISION!)" : "NO");
    printf("H(M) == expected: %s\n", correct36 ? "YES" : "NO");

    printf("Differing words: ");
    for (int i = 0; i < 16; ++i)
        if (M36[i] != M36p[i]) printf("W%d ", i);
    printf("\n");

    // Overall
    printf("\n=== Summary ===\n");
    printf("37-step SFS: %s\n", (match37 && correct37) ? "VERIFIED" : "FAILED");
    printf("36-step SFS: %s\n", (match36 && correct36) ? "VERIFIED" : "FAILED");

    return (match37 && correct37 && match36 && correct36) ? 0 : 1;
}
