#include "params.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int PQCLEAN_MLDSA44_CLEAN_crypto_sign_keypair(uint8_t *pk, uint8_t *sk);

int pqc_fault_calls_remaining;
static const uint8_t *seed_bytes;
static int seed_used;

void pqc_reset(const uint8_t *next) {
    seed_bytes = next;
    seed_used = 0;
}

int PQCLEAN_randombytes(uint8_t *output, size_t length) {
    if (seed_used || length != SEEDBYTES) return -1;
    memcpy(output, seed_bytes, SEEDBYTES);
    seed_used = 1;
    return 0;
}

int main(void) {
    const uint8_t seed[SEEDBYTES] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    uint8_t clean[PQCLEAN_MLDSA44_CLEAN_CRYPTO_PUBLICKEYBYTES];
    uint8_t faulted[PQCLEAN_MLDSA44_CLEAN_CRYPTO_PUBLICKEYBYTES];
    uint8_t secret[PQCLEAN_MLDSA44_CLEAN_CRYPTO_SECRETKEYBYTES];
    size_t changed = 0;
    pqc_fault_calls_remaining = 0;
    pqc_reset(seed);
    if (PQCLEAN_MLDSA44_CLEAN_crypto_sign_keypair(clean, secret) != 0) return 1;
    pqc_fault_calls_remaining = 1;
    pqc_reset(seed);
    if (PQCLEAN_MLDSA44_CLEAN_crypto_sign_keypair(faulted, secret) != 0) return 2;
    if (pqc_fault_calls_remaining != 0) return 3;
    for (size_t index = 0; index < sizeof clean; index++) changed += clean[index] != faulted[index];
    const int rho_equal = memcmp(clean, faulted, SEEDBYTES) == 0;
    printf("{\"implementation\":\"pqclean-ml-dsa-44-clean\",\"fault\":\"first_s1_polynomial_one_power_of_two_twiddle_per_layer\",\"seedPrefixEqual\":%s,\"publicKeysDiffer\":%s,\"changedPublicKeyBytes\":%zu}\n",
           rho_equal ? "true" : "false", changed ? "true" : "false", changed);
    return changed ? 0 : 4;
}
