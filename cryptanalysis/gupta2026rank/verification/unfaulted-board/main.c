#define MLK_CONFIG_PARAMETER_SET 512
#define MLK_CONFIG_NAMESPACE_PREFIX unfaulted
#define MLK_CONFIG_NO_RANDOMIZED_API

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mlkem_native.h"

enum { TEST_CASES = 3 };

static uint8_t public_key[MLKEM512_PUBLICKEYBYTES];
static uint8_t secret_key[MLKEM512_SECRETKEYBYTES];
static uint8_t ciphertext[MLKEM512_CIPHERTEXTBYTES];
static uint8_t keygen_coins[2 * MLKEM_SYMBYTES];
static uint8_t encaps_coins[MLKEM_SYMBYTES];
static uint8_t encaps_secret[MLKEM_BYTES];
static uint8_t decaps_secret[MLKEM_BYTES];

#include "mlkem_native.c"

int main(void)
{
  puts("MLKEM512_UNFAULTED_READY");
  fflush(stdout);
  if (getchar() != 't')
  {
    puts("MLKEM512_FAIL_COMMAND");
    return 1;
  }

  for (unsigned case_index = 0; case_index < TEST_CASES; case_index++)
  {
    memset(keygen_coins, (int)case_index, sizeof(keygen_coins));
    memset(encaps_coins, (int)case_index + 16, sizeof(encaps_coins));
    if (unfaulted_keypair_derand(public_key, secret_key, keygen_coins) != 0)
    {
      puts("MLKEM512_FAIL_KEYGEN");
      return 1;
    }
    if (unfaulted_enc_derand(ciphertext, encaps_secret, public_key, encaps_coins) != 0)
    {
      puts("MLKEM512_FAIL_ENCAPS");
      return 1;
    }
    if (unfaulted_dec(decaps_secret, ciphertext, secret_key) != 0)
    {
      puts("MLKEM512_FAIL_DECAPS");
      return 1;
    }
    if (memcmp(encaps_secret, decaps_secret, sizeof(encaps_secret)) != 0)
    {
      puts("MLKEM512_FAIL_SECRET_MISMATCH");
      return 1;
    }
    printf("MLKEM512_CASE_%u_OK\n", case_index);
  }

  puts("MLKEM512_UNFAULTED_PASS");
  return 0;
}