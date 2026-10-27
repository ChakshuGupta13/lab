#include <stddef.h>
#include <stdint.h>

#include "api.h"

enum {
  STACK_TOP = 0x20400000,
  CPACR_ADDRESS = 0xe000ed88,
  FPU_ACCESS_BITS = 0x00f00000,
  SEMIHOST_EXIT_EXTENDED = 0x20,
  SEMIHOST_APPLICATION_EXIT = 0x20026
};

unsigned char keygen_public_key[CRYPTO_PUBLICKEYBYTES];
unsigned char keygen_secret_key[CRYPTO_SECRETKEYBYTES];
volatile uint32_t keygen_status;
volatile uint32_t keygen_random_calls;

void *memcpy(void *destination, const void *source, size_t length)
{
  unsigned char *output = destination;
  const unsigned char *input = source;
  for (size_t offset = 0; offset < length; offset++) output[offset] = input[offset];
  return destination;
}

void *memset(void *destination, int value, size_t length)
{
  unsigned char *output = destination;
  for (size_t offset = 0; offset < length; offset++) output[offset] = (unsigned char)value;
  return destination;
}

int memcmp(const void *left, const void *right, size_t length)
{
  const unsigned char *first = left;
  const unsigned char *second = right;
  for (size_t offset = 0; offset < length; offset++) {
    if (first[offset] != second[offset]) return first[offset] < second[offset] ? -1 : 1;
  }
  return 0;
}

int PQCLEAN_randombytes(uint8_t *output, size_t length)
{
  uint32_t state = 0x9e3779b9u + 0x6d2b79f5u * keygen_random_calls++;
  for (size_t offset = 0; offset < length; offset++) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    output[offset] = (uint8_t)state;
  }
  return 0;
}

__attribute__((noinline)) void keygen_before(void)
{
  __asm__ volatile("" ::: "memory");
}

__attribute__((noinline)) void keygen_after(void)
{
  __asm__ volatile("" ::: "memory");
}

__attribute__((noreturn)) void probe_reset(void)
{
  *(volatile uint32_t *)CPACR_ADDRESS |= FPU_ACCESS_BITS;
  __asm__ volatile("dsb\nisb" ::: "memory");
  keygen_status = 1;
  keygen_before();
  crypto_kem_keypair(keygen_public_key, keygen_secret_key);
  keygen_status = 2;
  keygen_after();
  uint32_t exit_reason[2] = {SEMIHOST_APPLICATION_EXIT, 0};
  register uint32_t operation __asm__("r0") = SEMIHOST_EXIT_EXTENDED;
  register uint32_t argument __asm__("r1") = (uint32_t)exit_reason;
  __asm__ volatile("bkpt 0xab" : "+r"(operation), "+r"(argument) :: "memory");
  for (;;) __asm__ volatile("wfi");
}

__attribute__((section(".isr_vector"), used))
const uintptr_t keygen_vectors[] = {STACK_TOP, (uintptr_t)probe_reset};