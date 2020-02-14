#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <sys/time.h>
#include "cgbn/cgbn.h"
#include "samples/utility/gpu_support.h"
#include <curand.h>
#include <curand_kernel.h>

#include <chrono>
#include <random>
#include <cassert>

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define CPH_BITS 2048 // cipher bits
#define MAX_RAND_SEED 4294967295U
#define WINDOW_BITS 5
#define LOW_BOUND 5

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, CPH_BITS> env_cph_t;
typedef cgbn_mem_t<CPH_BITS> gpu_cph;

void store2dev(void *address,  mpz_t z, unsigned int BITS) {
  size_t words;
  if(mpz_sizeinbase(z, 2)>BITS) {
    exit(1);
  }
  mpz_export((uint32_t *)address, &words, -1, sizeof(uint32_t), 0, 0, z);
  while(words<(BITS+31)/32)
    ((uint32_t *)address)[words++]=0;
}

void store2gmp(mpz_t z, void *address, unsigned int BITS ) {
  mpz_import(z, (BITS+31)/32, -1, sizeof(uint32_t), 0, 0, (uint32_t *)address);
}

void getprimeover(mpz_t rop, int bits, int &seed_start){
  gmp_randstate_t state;
  gmp_randinit_default(state);
  gmp_randseed_ui(state, seed_start);
  seed_start++;
  mpz_t rand_num;
  mpz_init(rand_num);
  mpz_urandomb(rand_num, state, bits);
  mpz_setbit(rand_num, bits-1);
  mpz_nextprime(rop, rand_num); 
  mpz_clear(rand_num);
}

void invert(mpz_t rop, mpz_t a, mpz_t b) {
  mpz_invert(rop, a, b);
}

class PaillierPublicKey {
 public:
  cgbn_mem_t<CPH_BITS> g;
  cgbn_mem_t<CPH_BITS> n;
  cgbn_mem_t<CPH_BITS> nsquare;
  cgbn_mem_t<CPH_BITS> max_int;
};


// template<unsigned int BITS>
class PaillierPrivateKey {
 public:
  cgbn_mem_t<CPH_BITS> p;
  cgbn_mem_t<CPH_BITS> q;
  cgbn_mem_t<CPH_BITS> psquare;
  cgbn_mem_t<CPH_BITS> qsquare;
  cgbn_mem_t<CPH_BITS> q_inverse;
  cgbn_mem_t<CPH_BITS> hp;
  cgbn_mem_t<CPH_BITS> hq;
};

// template<unsigned int _BITS, unsigned int _TPI>
__device__ __forceinline__ 
void mont_modular_power(env_cph_t &bn_env, env_cph_t::cgbn_t &result, 
		const env_cph_t::cgbn_t &x, const env_cph_t::cgbn_t &power, 
		const env_cph_t::cgbn_t &modulus) {
/************************************************************************************
* calculate x^power mod modulus with montgomery multiplication.
* input: x, power, modulus.
* output: result
* requirement: x < modulus and modulus is an odd number.
*/

  env_cph_t::cgbn_t         t, starts;
  int32_t      index, position, leading;
  uint32_t     mont_inv;
  env_cph_t::cgbn_local_t   odd_powers[1<<WINDOW_BITS-1];

  // find the leading one in the power
  leading=CPH_BITS-1-cgbn_clz(bn_env, power);
  if(leading>=0) {
    // convert x into Montgomery space, store in the odd powers table
    mont_inv=cgbn_bn2mont(bn_env, result, x, modulus);
    
    // compute t=x^2 mod modulus
    cgbn_mont_sqr(bn_env, t, result, modulus, mont_inv);
    
    // compute odd powers window table: x^1, x^3, x^5, ...
    cgbn_store(bn_env, odd_powers, result);
    #pragma nounroll
    for(index=1;index<(1<<WINDOW_BITS-1);index++) {
      cgbn_mont_mul(bn_env, result, result, t, modulus, mont_inv);
      cgbn_store(bn_env, odd_powers+index, result);
    }

    // starts contains an array of bits indicating the start of a window
    cgbn_set_ui32(bn_env, starts, 0);

    // organize p as a sequence of odd window indexes
    position=0;
    while(true) {
      if(cgbn_extract_bits_ui32(bn_env, power, position, 1)==0)
        position++;
      else {
        cgbn_insert_bits_ui32(bn_env, starts, starts, position, 1, 1);
        if(position+WINDOW_BITS>leading)
          break;
        position=position+WINDOW_BITS;
      }
    }

    // load first window.  Note, since the window index must be odd, we have to
    // divide it by two before indexing the window table.  Instead, we just don't
    // load the index LSB from power
    index=cgbn_extract_bits_ui32(bn_env, power, position+1, WINDOW_BITS-1);
    cgbn_load(bn_env, result, odd_powers+index);
    position--;
    
    // Process remaining windows 
    while(position>=0) {
      cgbn_mont_sqr(bn_env, result, result, modulus, mont_inv);
      if(cgbn_extract_bits_ui32(bn_env, starts, position, 1)==1) {
        // found a window, load the index
        index=cgbn_extract_bits_ui32(bn_env, power, position+1, WINDOW_BITS-1);
        cgbn_load(bn_env, t, odd_powers+index);
        cgbn_mont_mul(bn_env, result, result, t, modulus, mont_inv);
      }
      position--;
    }
    
    // convert result from Montgomery space
    cgbn_mont2bn(bn_env, result, result, modulus, mont_inv);
  }
  else {
    // p=0, thus x^p mod modulus=1
    cgbn_set_ui32(bn_env, result, 1);
  }
}

__device__  __forceinline__ void l_func(env_cph_t &bn_env, env_cph_t::cgbn_t &out, 
		env_cph_t::cgbn_t &cipher_t, env_cph_t::cgbn_t &x_t, env_cph_t::cgbn_t &xsquare_t, 
		env_cph_t::cgbn_t &hx_t) {
/****************************************************************************************
* calculate L(cipher_t^(x_t - 1) mod xsquare_t) * hx_t. 
* input: cipher_t, x_t, xsquare-t, hx_t
* out:   out
*/
  env_cph_t::cgbn_t  tmp, tmp2, cipher_lt;
  env_cph_t::cgbn_wide_t  tmp_wide;
  cgbn_sub_ui32(bn_env, tmp2, x_t, 1);
  
  if(cgbn_compare(bn_env, cipher_t, xsquare_t) >= 0) {
    cgbn_rem(bn_env, cipher_lt, cipher_t, xsquare_t);
    mont_modular_power(bn_env,tmp,cipher_lt,tmp2,xsquare_t);
  } else {
    mont_modular_power(bn_env, tmp, cipher_t, tmp2, xsquare_t);
  }
 
  cgbn_sub_ui32(bn_env, tmp, tmp, 1);
  cgbn_div(bn_env, tmp, tmp, x_t);
  cgbn_mul_wide(bn_env, tmp_wide, tmp, hx_t);
  cgbn_rem_wide(bn_env, tmp, tmp_wide, x_t);
 
  cgbn_set(bn_env, out, tmp);
}


__global__ void setup_kernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ __noinline__ void apply_obfuscator(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, 
		gpu_cph *ciphers, gpu_cph *obfuscators, int count, curandState *state ) {
/******************************************************************************************
* obfuscate the encrypted text, obfuscator = cipher * r^n mod n^2
* in:
*   ciphers: encrypted text from simple raw encryption
*   state:   GPU random generator state.
* out:
*   obfuscators: obfused encryption text.
*/
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int tid= idx/TPI;
  if(tid>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t     bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n, nsquare,cipher, r, tmp;
  env_cph_t::cgbn_wide_t tmp_wide;

  curandState localState = state[idx];
  unsigned int rand_r = curand_uniform(&localState) * MAX_RAND_SEED;                  
  state[idx] = localState;

  cgbn_set_ui32(bn_env, r, rand_r); // TODO: new rand or reuse
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);
  cgbn_load(bn_env, cipher, &ciphers[tid]);
  mont_modular_power(bn_env, tmp, r, n, nsquare);
  cgbn_mul_wide(bn_env, tmp_wide, cipher, tmp);
  cgbn_rem_wide(bn_env, r, tmp_wide, nsquare);
  cgbn_store(bn_env, obfuscators + tid, r);   // store r into sum
}


__global__ void raw_encrypt(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, 
		gpu_cph *plains, gpu_cph *ciphers,int count) {
/*************************************************************************************
* simple encrption cipher = 1 + plain * n mod n^2
* in:
*   plains: plain text(2048 bits)
* out:
*   ciphers: encrypted result.
*/
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t      bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n, nsquare, plain,  tmp, max_int, cipher;
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);      
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);
  cgbn_load(bn_env, max_int, &gpu_pub_key[0].max_int);
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_sub(bn_env, tmp, n, max_int); 
  cgbn_mul(bn_env, cipher, n, plain);
  cgbn_add_ui32(bn_env, cipher, cipher, 1);
  cgbn_rem(bn_env, cipher, cipher, nsquare);

  cgbn_store(bn_env, ciphers + tid, cipher);   // store r into sum
}

__global__ __noinline__ void raw_encrypt_with_obfs(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, 
		gpu_cph *plains, gpu_cph *ciphers, int count, curandState *state) {
/*******************************************************************************
* encryption and obfuscation in one function, with less memory copy.
* in:
*   plains: plain text.
*   state: random generator state.
* out:
*   ciphers: encrpted text.
*/
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int tid= idx/TPI;
  if(tid>=count)
    return;
  context_t     bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t     bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n, nsquare, plain,  tmp, max_int, cipher; 
  env_cph_t::cgbn_wide_t tmp_wide;
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);      
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);
  cgbn_load(bn_env, max_int, &gpu_pub_key[0].max_int);
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_sub(bn_env, tmp, n, max_int); 
  cgbn_mul(bn_env, cipher, n, plain);
  cgbn_add_ui32(bn_env, cipher, cipher, 1);
  cgbn_rem(bn_env, cipher, cipher, nsquare);

  env_cph_t::cgbn_t r; 

  curandState localState = state[idx];
  unsigned int rand_r = curand_uniform(&localState) * MAX_RAND_SEED;                  
  state[idx] = localState;

  cgbn_set_ui32(bn_env, r, rand_r); // TODO: new rand or reuse

  mont_modular_power(bn_env,tmp, r, n, nsquare);

  cgbn_mul_wide(bn_env, tmp_wide, cipher, tmp);
  cgbn_rem_wide(bn_env, r, tmp_wide, nsquare);
  cgbn_store(bn_env, ciphers + tid, r);   // store r into sum
}


__global__ __noinline__ void raw_add(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, gpu_cph *ciphers_r, 
		gpu_cph *ciphers_a, gpu_cph *ciphers_b,int count) {
/**************************************************************************************
* add under encrypted text.
* in: 
*   ciphers_a, ciphers_b: encrypted a and b.
* out:
*   ciphers_r: encrypted result.
*/
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t          bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  nsquare, r, a, b;
  env_cph_t::cgbn_wide_t r_wide;
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);      
  cgbn_load(bn_env, a, ciphers_a + tid);      
  cgbn_load(bn_env, b, ciphers_b + tid);
  cgbn_mul_wide(bn_env, r_wide, a, b);
  cgbn_rem_wide(bn_env, r, r_wide, nsquare);
  cgbn_store(bn_env, ciphers_r + tid, r);
}

__global__ void raw_mul(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, gpu_cph *ciphers_r, 
		gpu_cph *ciphers_a, gpu_cph *plains_b,int count) {
/****************************************************************************************
* multiplication under encrypted text. b * a.
* in:
*   ciphers_a, plains_b: encrypted a and b.
* out:
*   ciphers_r: encrypted result.
*/
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t      bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n,max_int, nsquare, r, cipher, plain, neg_c, neg_scalar,tmp;               

  cgbn_load(bn_env, n, &gpu_pub_key[0].n);      
  cgbn_load(bn_env, max_int, &gpu_pub_key[0].max_int);      
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);      
  cgbn_load(bn_env, cipher, ciphers_a + tid);      
  cgbn_load(bn_env, plain, plains_b + tid);

  cgbn_sub(bn_env, tmp, n, max_int); 
  if(cgbn_compare(bn_env, plain, tmp) >= 0 ) {
    // Very large plaintext, take a sneaky shortcut using inverses
    cgbn_modular_inverse(bn_env,neg_c, cipher, nsquare);
    cgbn_sub(bn_env, neg_scalar, n, plain);
    mont_modular_power(bn_env, r, neg_c, neg_scalar, nsquare);
  } else {
    mont_modular_power(bn_env, r, cipher, plain, nsquare);
  }
  cgbn_store(bn_env, ciphers_r + tid, r);
}

  
__global__ void raw_decrypt(PaillierPrivateKey *gpu_priv_key, PaillierPublicKey *gpu_pub_key,
	   	cgbn_error_report_t *report, gpu_cph *plains, gpu_cph *ciphers, int count) {
/*************************************************************************************
* decryption
* in:
*   ciphers: encrypted text. 2048 bits.
* out:
*   plains: decrypted plain text.
*/
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);
  env_cph_t          bn_env(bn_context.env<env_cph_t>());
  env_cph_t::cgbn_t  mp, mq, tmp, q_inverse, n, p, q, hp, hq, psquare, qsquare, cipher;  
  cgbn_load(bn_env, cipher, ciphers + tid);
  cgbn_load(bn_env, q_inverse, &gpu_priv_key[0].q_inverse);
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);
  cgbn_load(bn_env, p, &gpu_priv_key[0].p);
  cgbn_load(bn_env, q, &gpu_priv_key[0].q);
  cgbn_load(bn_env, hp, &gpu_priv_key[0].hp);
  cgbn_load(bn_env, hq, &gpu_priv_key[0].hq);
  cgbn_load(bn_env, psquare, &gpu_priv_key[0].psquare);
  cgbn_load(bn_env, qsquare, &gpu_priv_key[0].qsquare);
  
  l_func(bn_env, mp, cipher, p, psquare, hp); 
  l_func(bn_env, mq, cipher, q, qsquare, hq); 
  
  cgbn_sub(bn_env, tmp, mp, mq);
  cgbn_mul(bn_env, tmp, tmp, q_inverse); 
  cgbn_rem(bn_env, tmp, tmp, p);
  cgbn_mul(bn_env, tmp, tmp, q);
  cgbn_add(bn_env, tmp, mq, tmp);
  cgbn_rem(bn_env, tmp, tmp, n);
  
  cgbn_store(bn_env, plains + tid, tmp);
}

void print_buffer_in_hex(char *addr, int count) {
  printf("dumping memory in hex\n");
  for (int i = 0; i < count; i++)
    printf('%x', addr + i);
  printf("\n");
}

extern "C" {
PaillierPublicKey* gpu_pub_key;
PaillierPrivateKey* gpu_priv_key;
cgbn_error_report_t* err_report;

void init_pub_key(void *n, void *g, void *nsquare, void *max_int) {
  cudaMalloc(&gpu_pub_key, sizeof(PaillierPublicKey));
  cudaMemcpy((void *)&gpu_pub_key->g, g, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_pub_key->n, n, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_pub_key->nsquare, nsquare, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_pub_key->max_int, max_int, CPH_BITS/8, cudaMemcpyHostToDevice);
}

void init_priv_key(void *p, void *q, void *psquare, void *qsquare, void *q_inverse,
                   void *hp, void *hq) {
  cudaMalloc(&gpu_priv_key, sizeof(PaillierPrivateKey));
  cudaMemcpy((void *)&gpu_priv_key->p, p, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->q, q, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->psquare, psquare, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->qsquare, qsquare, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->q_inverse, q_inverse, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->hp, hp, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->hq, hq, CPH_BITS/8, cudaMemcpyHostToDevice);
}

void init_err_report() {
  CUDA_CHECK(cgbn_error_report_alloc(&err_report));
}

void reset() {
  CUDA_CHECK(cgbn_error_report_free(err_report));
  cudaFree(gpu_pub_key);
  cudaFree(gpu_priv_key);
}

void call_raw_encrypt(uint32_t *addr, int count, char *res) {
  gpu_cph *plains_on_gpu;
  gpu_cph *ciphers;
  int TPB = 128;
  int IPB = TPB/TPI;
  cudaMalloc((void **)&plains_on_gpu, sizeof(gpu_cph) * count);
  cudaMalloc((void **)&ciphers, sizeof(gpu_cph) * count);
  cudaMemset((void *)plains_on_gpu, 0, sizeof(gpu_cph) * count);

  for (int i = 0; i < count; i++)
    cudaMemcpy(plains_on_gpu + i, addr + i, sizeof(uint32_t), cudaMemcpyHostToDevice);

  int block_size = (count + IPB - 1)/IPB;
  int thread_size = TPB;

  raw_encrypt<<<block_size, thread_size>>>(gpu_pub_key, err_report, plains_on_gpu, ciphers, count);
  for (int i = 0; i < count; i++)
    cudaMemcpy(res + i * sizeof(gpu_cph), ciphers + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);

  print_buffer_in_hex(res, sizeof(gpu_cph));
  cudaFree(plains_on_gpu);
  cudaFree(ciphers);

  return;
}
}