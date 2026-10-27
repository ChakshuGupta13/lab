#ifndef MLK_CONFIG_PARAMETER_SET
#define MLK_CONFIG_PARAMETER_SET 512
#endif
#define MLK_CONFIG_NAMESPACE_PREFIX pilot
#define MLK_CONFIG_NO_RANDOMIZED_API

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if MLK_CONFIG_PARAMETER_SET == 512
enum { DEGREE = 256, CALLS = 8, COINS = 64, PK_BYTES = 800, SK_BYTES = 1632, LEVEL = 512 };
#elif MLK_CONFIG_PARAMETER_SET == 768
enum { DEGREE = 256, CALLS = 8, COINS = 64, PK_BYTES = 1184, SK_BYTES = 2400, LEVEL = 768 };
#elif MLK_CONFIG_PARAMETER_SET == 1024
enum { DEGREE = 256, CALLS = 8, COINS = 64, PK_BYTES = 1568, SK_BYTES = 3168, LEVEL = 1024 };
#else
#error unsupported MLK_CONFIG_PARAMETER_SET
#endif

static unsigned phase;
static unsigned active_call;
static unsigned call_counts[2];
static unsigned fault_hits;
static int fault_enabled;
static int replace_enabled;
static int16_t replace_c0;
static int16_t replace_c1;
static int16_t inputs[2][CALLS][DEGREE];
static int16_t outputs[2][CALLS][DEGREE];

void pqc_ntt_begin(int16_t *coefficients)
{
  active_call = call_counts[phase]++;
  if (active_call >= CALLS)
  {
    fprintf(stderr, "unexpected NTT call\n");
    exit(1);
  }
  if (replace_enabled && active_call == 0)
  {
    coefficients[0] = replace_c0;
    coefficients[1] = replace_c1;
  }
  memcpy(inputs[phase][active_call], coefficients, sizeof(inputs[0][0]));
}

int16_t pqc_twiddle(unsigned index, int16_t original)
{
  if (fault_enabled && phase == 1 && active_call == 0 && index != 0 &&
      index <= 64 && (index & (index - 1)) == 0)
  {
    fault_hits++;
    return 0;
  }
  return original;
}

void pqc_ntt_end(int16_t *coefficients)
{
  memcpy(outputs[phase][active_call], coefficients, sizeof(outputs[0][0]));
}

#include "mlkem_native.c"

static void print_hex(const uint8_t *bytes, size_t length)
{
  putchar('"');
  for (size_t offset = 0; offset < length; offset++)
  {
    printf("%02x", bytes[offset]);
  }
  putchar('"');
}

static void print_coefficients(const int16_t *coefficients)
{
  putchar('[');
  for (unsigned offset = 0; offset < DEGREE; offset++)
  {
    printf("%s%d", offset == 0 ? "" : ",", coefficients[offset]);
  }
  putchar(']');
}

int main(int argc, char **argv)
{
  uint8_t coins[COINS];
  uint8_t correct_pk[PK_BYTES];
  uint8_t faulted_pk[PK_BYTES];
  uint8_t secret_key[SK_BYTES];
  if ((argc != 2 && argc != 3 && argc != 5) || strlen(argv[1]) != 2 * COINS ||
      (argc == 3 && strcmp(argv[2], "null") != 0 && strcmp(argv[2], "inputs") != 0) ||
      (argc == 5 && strcmp(argv[2], "replace") != 0))
  {
    fprintf(stderr, "usage: pilot COINS_HEX [null|inputs|replace C0 C1]\n");
    return 1;
  }
  if (argc == 5)
  {
    replace_enabled = 1;
    replace_c0 = (int16_t)atoi(argv[3]);
    replace_c1 = (int16_t)atoi(argv[4]);
  }
  for (unsigned offset = 0; offset < COINS; offset++)
  {
    unsigned value;
    if (sscanf(argv[1] + 2 * offset, "%2x", &value) != 1)
    {
      fprintf(stderr, "invalid test coins\n");
      return 1;
    }
    coins[offset] = (uint8_t)value;
  }
  int inputs_mode = argc == 3 && strcmp(argv[2], "inputs") == 0;
  fault_enabled = argc != 3 || inputs_mode;
  if (pilot_keypair_derand(correct_pk, secret_key, coins) != 0)
  {
    fprintf(stderr, "correct key generation failed\n");
    return 1;
  }
  phase = 1;
  if (pilot_keypair_derand(faulted_pk, secret_key, coins) != 0)
  {
    fprintf(stderr, "faulted key generation failed\n");
    return 1;
  }
  if (inputs_mode)
  {
    printf("{\"calls\":%u,\"inputs\":[", call_counts[0]);
    for (unsigned call = 0; call < call_counts[0]; call++)
    {
      if (call != 0)
      {
        putchar(',');
      }
      print_coefficients(inputs[0][call]);
    }
    printf("],\"correct_pk\":");
    print_hex(correct_pk, PK_BYTES);
    printf(",\"faulted_pk\":");
    print_hex(faulted_pk, PK_BYTES);
    printf("}\n");
    return 0;
  }
  int unchanged_inputs = memcmp(inputs[0], inputs[1], sizeof(inputs[0])) == 0;
  int unchanged_other_outputs =
      memcmp(outputs[0][1], outputs[1][1],
             (CALLS - 1) * sizeof(outputs[0][0])) == 0;
  printf("{\"public\":{\"level\":%d,\"target\":0,\"faults\":", LEVEL);
  printf("%s", fault_enabled ? "[1,2,4,8,16,32,64]" : "[]");
  printf(",\"replacement\":0,\"correct_pk\":");
  print_hex(correct_pk, PK_BYTES);
  printf(",\"faulted_pk\":");
  print_hex(faulted_pk, PK_BYTES);
  printf("},\"private\":{\"input\":");
  print_coefficients(inputs[0][0]);
  printf(",\"correct_ntt\":");
  print_coefficients(outputs[0][0]);
  printf(",\"faulted_ntt\":");
  print_coefficients(outputs[1][0]);
  printf(",\"calls\":[%u,%u],\"fault_hits\":%u,", call_counts[0],
         call_counts[1], fault_hits);
  printf("\"unchanged_inputs\":%s,\"unchanged_other_outputs\":%s}}\n",
         unchanged_inputs ? "true" : "false",
         unchanged_other_outputs ? "true" : "false");
  return 0;
}