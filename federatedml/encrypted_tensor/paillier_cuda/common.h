#ifndef PAILLIER_CUDA_COMMON
#define PAILLIER_CUDA_COMMON
#include <gmp.h>
#include "cgbn/cgbn.h"
#include <stdint.h>
#include <cuda.h>

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define PAILLIER_TPI 32
#define CPH_BITS 2048 // cipher bits
#define MAX_RAND_SEED 4294967295U
#define WINDOW_BITS 5
// helpful typedefs for the kernel
typedef cgbn_context_t<PAILLIER_TPI>    context_t;
typedef cgbn_env_t<context_t, CPH_BITS> env_cph_t;
typedef cgbn_mem_t<CPH_BITS> gpu_cph;

#define PLAIN_BITS 1024

#ifdef PLAIN_INT32
  typedef uint32_t plain_t;
#else
  #ifdef PLAIN_INT64
  typedef uint64_t plain_t;
  #else
    #ifdef PLAIN_BITS
    typedef cgbn_mem_t<PLAIN_BITS> plain_t;
    #endif
  #endif
#endif
struct FixedPointNumber {
  char encoding[PLAIN_BITS/8];
  int32_t exponent;
  int32_t base;
};

enum MemcpyType {
HostToHost = 0,
HostToDevice,
DeviceToHost,
DeviceToDevice
};

class PaillierPublicKey {
public:
  cgbn_mem_t<CPH_BITS> g;
  cgbn_mem_t<CPH_BITS> n;
  cgbn_mem_t<CPH_BITS> nsquare;
  cgbn_mem_t<CPH_BITS> max_int;
};


class PaillierPrivateKey {
public:
  cgbn_mem_t<CPH_BITS> p;
  cgbn_mem_t<CPH_BITS> q;
  cgbn_mem_t<CPH_BITS> psquare;
  cgbn_mem_t<CPH_BITS> qsquare;
  cgbn_mem_t<CPH_BITS> q_inverse;
  cgbn_mem_t<CPH_BITS> p_inverse;
  cgbn_mem_t<CPH_BITS> hp;
  cgbn_mem_t<CPH_BITS> hq;
};

struct PaillierEncryptedNumber {
  char cipher[CPH_BITS/8]; // expected size: CPH_BITS/8 bytes
  uint32_t exponent;
  uint32_t base;
};

void invert(mpz_t rop, mpz_t a, mpz_t b) {
  mpz_invert(rop, a, b);
}

inline void cudaMallocAndSet(void **addr, size_t size) {
  cudaMalloc(addr, size);
  cudaMemset(*addr, 0, size);
}

inline void dumpMem(char *addr, size_t size) {
  printf("dumping memory at 0x%x\n", addr);
  printf("0x");
  for (int64_t i = size - 1; i >= 0; i--) printf("%02x", addr[i] & 0xff);
  printf("\n");
}


inline void extractPen(gpu_cph *dst, PaillierEncryptedNumber *src, uint32_t count, MemcpyType type) {
  for (int i = 0; i < count; i++) {
    if (type == HostToHost)
      memcpy(dst + i, src[i].cipher, sizeof(gpu_cph));
    else if (type == HostToDevice)
      cudaMemcpy(dst + i, src[i].cipher, sizeof(gpu_cph), cudaMemcpyHostToDevice);
    else if (type == DeviceToHost)
      cudaMemcpy(dst + i, src[i].cipher, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    else if (type == DeviceToDevice)
      cudaMemcpy(dst + i, src[i].cipher, sizeof(gpu_cph), cudaMemcpyDeviceToDevice);
  }
}

inline void penFromBuffer(PaillierEncryptedNumber *dst, gpu_cph *src, uint32_t count, MemcpyType type) {
  for (int i = 0; i < count; i++) {
    if (type == HostToHost)
      memcpy((dst + i)->cipher, src + i, sizeof(gpu_cph));
    else if (type == HostToDevice)
      cudaMemcpy((dst + i)->cipher, src + i, sizeof(gpu_cph), cudaMemcpyHostToDevice);
    else if (type == DeviceToHost)
      cudaMemcpy((dst + i)->cipher, src + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    else if (type == DeviceToDevice)
      cudaMemcpy((dst + i)->cipher, src + i, sizeof(gpu_cph), cudaMemcpyDeviceToDevice);
  }
}

template<uint32_t _BITS>
inline void set_ui64(cgbn_mem_t<_BITS> *target, uint64_t value) {
  cudaMemcpy((void *)target, &value, sizeof(uint64_t), cudaMemcpyHostToDevice);
}

inline void store2dev(void *address,  mpz_t z, unsigned int BITS) {
  size_t words;
  if(mpz_sizeinbase(z, 2)>BITS) {
    exit(1);
  }
  mpz_export((uint32_t *)address, &words, -1, sizeof(uint32_t), 0, 0, z);
  while(words<(BITS+31)/32)
    ((uint32_t *)address)[words++]=0;
}

inline void store2gmp(mpz_t z, void *address, unsigned int BITS ) {
  mpz_import(z, (BITS+31)/32, -1, sizeof(uint32_t), 0, 0, (uint32_t *)address);
}

#endif