#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "common.h"
#include "samples/utility/gpu_support.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <cstdlib>

// #include <chrono>
#include <cassert>

template<uint32_t _BITS, uint32_t _TPI>
__device__ __forceinline__ 
void mont_modular_power(cgbn_env_t<cgbn_context_t<_TPI>, _BITS> &bn_env,   \
    typename cgbn_env_t<cgbn_context_t<_TPI>, _BITS>::cgbn_t &result,      \
	  const typename cgbn_env_t<cgbn_context_t<_TPI>, _BITS>::cgbn_t &x,     \
    const typename cgbn_env_t<cgbn_context_t<_TPI>, _BITS>::cgbn_t &power, \
	  const typename cgbn_env_t<cgbn_context_t<_TPI>, _BITS>::cgbn_t &modulus) {
// calculate x^power mod modulus with montgomery multiplication.
// input: x, power, modulus.
// output: result
// requirement: x < modulus and modulus is an odd number.

  typename cgbn_env_t<cgbn_context_t<_TPI>, _BITS>::cgbn_t  t, starts;
  int32_t      index, position, leading;
  uint32_t     mont_inv;
  typename cgbn_env_t<cgbn_context_t<_TPI>, _BITS>::cgbn_local_t odd_powers[1<<WINDOW_BITS-1];

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

template<uint32_t _BITS, uint32_t _TPI>
__global__ __noinline__ 
void mul(cgbn_mem_t<_BITS> *a, cgbn_mem_t<_BITS> *b, cgbn_mem_t<_BITS> *res, \
    cgbn_error_report_t *report, const uint32_t count) {

  typedef cgbn_context_t<_TPI> mul_context;
  typedef cgbn_env_t<mul_context, _BITS> env_mul_t;
  uint32_t tid = (blockIdx.x * blockDim.x + threadIdx.x)/_TPI;
  if (tid >= count)
    return;
  
  mul_context bn_context(cgbn_report_monitor, report, tid);
  env_mul_t bn_env(bn_context);
  typename cgbn_env_t<cgbn_context_t<_TPI>, _BITS>::cgbn_t oprand1, oprand2, tmp;
  
  cgbn_load(bn_env, oprand1, a + tid);
  cgbn_load(bn_env, oprand2, b + tid);
  cgbn_mul(bn_env, tmp, oprand1, oprand2);
  cgbn_store(bn_env, res + tid, tmp);
  
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
    mont_modular_power<CPH_BITS, PAILLIER_TPI>(bn_env, tmp, cipher_lt, tmp2, xsquare_t);
  } else {
    mont_modular_power<CPH_BITS, PAILLIER_TPI>(bn_env, tmp, cipher_t, tmp2, xsquare_t);
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
  int tid= idx/PAILLIER_TPI;
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
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/PAILLIER_TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t      bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n, nsquare, plain, cipher;
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);      
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_mul(bn_env, cipher, n, plain);
  cgbn_add_ui32(bn_env, cipher, cipher, 1);
  cgbn_rem(bn_env, cipher, cipher, nsquare);

  cgbn_store(bn_env, ciphers + tid, cipher);   // store r into sum
}

__global__ __noinline__ void raw_encrypt_with_obfs(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, 
		gpu_cph *plains, gpu_cph *ciphers, int count, uint32_t *rand_vals) {
/*******************************************************************************
* encryption and obfuscation in one function, with less memory copy.
* in:
*   plains: plain text.
*   state: random generator state.
* out:
*   ciphers: encrpted text.
*/
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int tid= idx/PAILLIER_TPI;
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

  // curandState localState = state[idx];
  uint32_t rand_r = rand_vals[tid];
  // state[idx] = localState;

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
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/PAILLIER_TPI;
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
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/PAILLIER_TPI;
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


__global__ void raw_matmul(PaillierPublicKey *pub_key, cgbn_error_report_t *report, gpu_cph *ciphers_a, \
gpu_cph *plains_b, gpu_cph *ciphers_res, const uint32_t P, const uint32_t Q, const uint32_t R) {
  // size: P * Q, Q * R
  int col = blockIdx.x * blockDim.x + threadIdx.y;
  int row = blockIdx.y * blockDim.y + threadIdx.z;

  int tid = (row * R) + col;

  if (row >= P || col >= R)
   return;
  context_t bn_context(cgbn_report_monitor, report, tid);
  env_cph_t bn_env(bn_context.env<env_cph_t>());
  env_cph_t::cgbn_t n, max_int, nsquare, r, cipher, plain, neg_c, neg_scalar, tmp, tmp_res;
  env_cph_t::cgbn_wide_t tmp_wide;

  cgbn_set_ui32(bn_env, tmp_res, 1);
  cgbn_load(bn_env, n, &pub_key[0].n);
  cgbn_load(bn_env, max_int, &pub_key[0].max_int);
  cgbn_load(bn_env, nsquare, &pub_key[0].nsquare);

  for (int i = 0; i < Q; i++) {
    cgbn_load(bn_env, cipher, ciphers_a + row * Q + i);
    cgbn_load(bn_env, plain, plains_b + col * R + i); // remember b is flattened vertically.

    cgbn_sub(bn_env, tmp, n, max_int);
    if (cgbn_compare(bn_env, plain, tmp) >= 0) {
      cgbn_modular_inverse(bn_env, neg_c, cipher, nsquare);
      cgbn_sub(bn_env, neg_scalar, n, plain);
      mont_modular_power(bn_env, r, neg_c, neg_scalar, nsquare);
    } else {
      mont_modular_power(bn_env, r, cipher, plain, nsquare);
    }
    cgbn_mul_wide(bn_env, tmp_wide, tmp_res, r);
    cgbn_rem_wide(bn_env, tmp_res, tmp_wide, nsquare);
  }
  cgbn_store(bn_env, ciphers_res + row * R + col, tmp_res);
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
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/PAILLIER_TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);
  env_cph_t          bn_env(bn_context.env<env_cph_t>());
  env_cph_t::cgbn_t  mp, mq, tmp, p_inverse, n, p, q, hp, hq, psquare, qsquare, cipher, q_inverse;  
  cgbn_load(bn_env, cipher, ciphers + tid);
  cgbn_load(bn_env, q_inverse, &gpu_priv_key[0].q_inverse);
  cgbn_load(bn_env, p_inverse, &gpu_priv_key[0].p_inverse);
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);
  cgbn_load(bn_env, p, &gpu_priv_key[0].p);
  cgbn_load(bn_env, q, &gpu_priv_key[0].q);
  cgbn_load(bn_env, hp, &gpu_priv_key[0].hp);
  cgbn_load(bn_env, hq, &gpu_priv_key[0].hq);
  cgbn_load(bn_env, psquare, &gpu_priv_key[0].psquare);
  cgbn_load(bn_env, qsquare, &gpu_priv_key[0].qsquare);
  
  l_func(bn_env, mp, cipher, p, psquare, hp); 
  l_func(bn_env, mq, cipher, q, qsquare, hq); 
  
  if (cgbn_compare(bn_env, mp, mq) > 0) {
    cgbn_sub(bn_env, tmp, mp, mq);
    cgbn_mul(bn_env, tmp, tmp, q_inverse); 
    cgbn_rem(bn_env, tmp, tmp, p);
    cgbn_mul(bn_env, tmp, tmp, q);
    cgbn_add(bn_env, tmp, mq, tmp);
    cgbn_rem(bn_env, tmp, tmp, n);
  } else {
    cgbn_sub(bn_env, tmp, mq, mp);
    cgbn_mul(bn_env, tmp, tmp, p_inverse); 
    cgbn_rem(bn_env, tmp, tmp, q);
    cgbn_mul(bn_env, tmp, tmp, p);
    cgbn_add(bn_env, tmp, mp, tmp);
    cgbn_rem(bn_env, tmp, tmp, n);
  }
  
  cgbn_store(bn_env, plains + tid, tmp);
}

void print_buffer_in_hex(char *addr, int count) {
  printf("dumping memory in hex\n");
  for (int i = 0; i < count; i++)
    printf("%x", *(addr + i) & 0xff); // remove padding.
  printf("\n");
}

void print_num_hex(char *addr, int count) {
  printf("dumping memory in hex, little endine\n");
  bool leading = false;
  for (int i = count - 1; i >= 0; i--) {
    if (*(addr + i) >= 0 && leading == false) {
      leading = true;
      printf("%x", *(addr + i) & 0xff); // remove padding.
    } else if (leading == true) {
      printf("%x", *(addr + i) & 0xff); // remove padding.
    }
  }
  printf("\n");
}