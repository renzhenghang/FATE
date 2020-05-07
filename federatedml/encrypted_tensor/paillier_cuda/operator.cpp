// #include "opreator.h"
#include "paillier.cu"

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
                   void *hp, void *hq, void *p_inverse) {
  cudaMalloc(&gpu_priv_key, sizeof(PaillierPrivateKey));
  cudaMemcpy((void *)&gpu_priv_key->p, p, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->q, q, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->psquare, psquare, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->qsquare, qsquare, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->q_inverse, q_inverse, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->hp, hp, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->hq, hq, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->p_inverse, p_inverse, CPH_BITS/8, cudaMemcpyHostToDevice);
}

void init_err_report() {
  cgbn_error_report_alloc(&err_report);
}

void reset() {
  cgbn_error_report_free(err_report);
  cudaFree(gpu_pub_key);
  cudaFree(gpu_priv_key);
}

void call_raw_encrypt_obfs(gpu_cph *plains_on_gpu, const uint32_t count,  \
  gpu_cph *ciphers_on_gpu, uint32_t* rand_vals_gpu) {
  // all parameters on gpu

  int TPB = 128;
  int IPB = TPB/PAILLIER_TPI;
  int block_size = (count + IPB - 1)/IPB;
  int thread_size = TPB;
  if (rand_vals_gpu != NULL) {
    raw_encrypt_with_obfs<<<block_size, thread_size>>>(gpu_pub_key, err_report, \
      plains_on_gpu, ciphers_on_gpu, count, rand_vals_gpu);
  }
  else
    raw_encrypt<<<block_size, thread_size>>>(gpu_pub_key, err_report, plains_on_gpu,\
       ciphers_on_gpu, count);

}

void call_raw_add(gpu_cph *cipher_a, gpu_cph *cipher_b, gpu_cph *cipher_res, const uint32_t count) {
  
  int TPB = 128;
  int IPB = TPB/PAILLIER_TPI;

  int block_size = (count + IPB - 1) / IPB;
  int thread_size = TPB;

  raw_add<<<block_size, thread_size>>>(gpu_pub_key, err_report, cipher_res, cipher_a, cipher_b, count);

}

void call_raw_mul(gpu_cph *cipher_a, gpu_cph *plain_b, gpu_cph *cipher_res, const uint32_t count) {
  // a is cipher, b is plain
  
  int TPB = 128;
  int IPB = TPB/PAILLIER_TPI;
  
  int block_size = (count + IPB - 1) / IPB;
  int thread_size = TPB;

  raw_mul<<<block_size, thread_size>>>(gpu_pub_key, err_report, cipher_res, cipher_a, plain_b, count);
}

void call_raw_decrypt(gpu_cph *cipher_gpu, const uint32_t count, gpu_cph *res) {
  
  int TPB = 128;
  int IPB = TPB/PAILLIER_TPI;
  int block_size = (count + IPB - 1) / IPB;
  int thread_size = TPB;

  raw_decrypt<<<block_size, thread_size>>>(gpu_priv_key, gpu_pub_key, err_report, res, \
  cipher_gpu, count);
}


void call_raw_matmul(gpu_cph *cipher_gpu, gpu_cph *plain_b, gpu_cph *cipher_res, const uint32_t P,\
   const uint32_t Q, const uint32_t R) {
  dim3 threadPerBlock(PAILLIER_TPI, 4, 4); // TODO: remove hardcoded.
  uint32_t x_dim = ceil((double)P/(double)threadPerBlock.x);
  uint32_t y_dim = ceil((double)R/(double)threadPerBlock.y);

  dim3 blockPerGrid(x_dim, y_dim);

  raw_matmul<<<blockPerGrid, threadPerBlock>>>(gpu_pub_key, err_report, cipher_gpu, plain_b, \
    cipher_res, P, Q, R);
}


void cipher_align(PaillierEncryptedNumber *a, PaillierEncryptedNumber *b, const uint32_t count) {
  // align exponent before executing "encrypted add" operation
  // parameters:
  //   a: PEN array on cpu, b: same as a
  // steps:
  //   1. figure out whose exponent is bigger
  //   2. update exponent
  //   3. perform raw mul
  //   4. copy back to PaillierEncryptedNumber
  int *map = (int *) malloc(sizeof(int) * count);
  gpu_cph *cof;
  cudaMallocAndSet((void **)&cof, sizeof(gpu_cph) * count);
  // 1
  for (int i = 0; i < count; i++) {
    map[i] = a[i].exponent < b[i].exponent ? 0 : 1;
    uint64_t diff = (uint64_t) pow(a[i].base, abs((int)a[i].exponent- (int)b[i].exponent));
    //cudaMemcpy(cof + i, &diff, sizeof(uint64_t), cudaMemcpyHostToDevice);
    set_ui64<CPH_BITS>(cof + i, diff);
    if (a[i].exponent < b[i].exponent)
      a[i].exponent = b[i].exponent;
    else b[i].exponent = a[i].exponent;
  }
  gpu_cph *encoding;
  gpu_cph *res;
  
  cudaMalloc(&encoding, sizeof(gpu_cph) * count);
  cudaMalloc(&res, sizeof(gpu_cph) * count);
  for (int i = 0; i < count; i++) {
    if (map[i] == 0)
      cudaMemcpy(encoding + i, a + i, sizeof(gpu_cph), cudaMemcpyHostToDevice);
    else
      cudaMemcpy(encoding + i, b + i, sizeof(gpu_cph), cudaMemcpyHostToDevice);
  }
  // 2
  call_raw_mul(encoding, cof, res, count);
  // 3
  for (int i = 0; i < count; i++) {
    if (map[i] == 0)
      cudaMemcpy((a + i)->cipher, res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    else
      cudaMemcpy((b + i)->cipher, res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
  }
  
  //..
  cudaFree(encoding);
  cudaFree(res);
  free(map);
  cudaFree(cof);
 
}

void pen_increase_exponent_to(PaillierEncryptedNumber *a, const uint32_t exponent, \
   const uint32_t count) {
  printf("enter pen\n");
  printf("count: %d\n", count);
  gpu_cph *cof;
  gpu_cph *cipher_gpu;
  gpu_cph *cipher_res;
  cudaMallocAndSet((void **)&cof, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_gpu, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * count);
  uint32_t base = a[0].base;
  for (int i = 0; i < count; i++) {
    uint32_t diff = exponent >= a[i].exponent ? exponent - a[i].exponent : 0;
    uint64_t tmp = (uint64_t) pow(base, diff);
    // cudaMemcpy(cof + i, &tmp, sizeof(plain_t), cudaMemcpyHostToDevice);
    set_ui64<CPH_BITS>(cof + i, diff);
  }
  
  extractPen(cipher_gpu, a, count, HostToDevice);
  call_raw_mul(cipher_gpu, cof, cipher_res, count);
  for (int i = 0; i < count; i++) {
    cudaMemcpy((a + i)->cipher, cipher_res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    a[i].exponent = exponent;
  }

  cudaFree(cipher_gpu);
  cudaFree(cipher_res);
  cudaFree(cof);
}

// void fpn_increase_exponent_to(FixedPointNumber *a, const uint32_t exponent, const uint32_t count) {
//   plain_t *fpn_gpu;
//   plain_t *cof;
//   plain_t *res;

//   cudaMallocAndSet((void **)&fpn_gpu, sizeof(plain_t) * count);
//   cudaMallocAndSet((void **)&res, sizeof(plain_t) * count);
//   cudaMallocManaged((void **)&cof, sizeof(plain_t) * count);

//   for (int i = 0; i < count; i++)
//     cudaMemcpy(fpn_gpu + i, &a[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice);
  
//   uint32_t base = a[0].base;
//   for (int i = 0; i < count; i++)
//     cof[i] = (plain_t) pow(base, exponent - a[i].exponent);
//   uint32_t thread_size = 1024;
//   uint32_t block_size = ceil((double)count/(double)thread_size);
//   fpn_mul<<<block_size, thread_size>>>(fpn_gpu, cof, count, res);
//   for (int i = 0; i < count; i++) {
//     cudaMemcpy(&a[i].encoding, res + i, sizeof(plain_t), cudaMemcpyDeviceToHost);
//     a[i].exponent = exponent;
//   }

//   cudaFree(fpn_gpu);
//   cudaFree(cof);
//   cudaFree(res);

// }


void cipher_add_cipher(PaillierEncryptedNumber *a, PaillierEncryptedNumber *b, \
  PaillierEncryptedNumber *r, const uint32_t count) {
  // perform encrypted add on PEN
  // parameters:
  //   a, b: add numbers, on cpu. c: result on cpu
  // steps:
  //   1. align
  //   2. perform raw add
  //   3. copy to cpu
  cipher_align(a, b, count);
  gpu_cph *cipher_a;
  gpu_cph *cipher_b;
  gpu_cph *cipher_res;
  cudaMallocAndSet((void **)&cipher_a, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_b, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * count);

  extractPen(cipher_a, a, count, HostToDevice);
  extractPen(cipher_b, b, count, HostToDevice);

  call_raw_add(cipher_a, cipher_b, cipher_res, count);
  penFromBuffer(r, cipher_res, count, DeviceToHost);

  for (int i = 0; i < count; i++) {
    r[i].exponent = a[i].exponent;
    r[i].base = a[i].base;
  }
  
}


void plain_mul_cipher(FixedPointNumber *b, PaillierEncryptedNumber *a, \
   PaillierEncryptedNumber *r, const int count) {
  // perform encrypted multiplication
  // parameters:
  //   b: coefficients, plain text on cpu
  //   a: encrypted num of arrays
  //   r: result, all on cpu
  // steps:
  //   1. perform raw mul
  //   2. add exponent together.
  //   3. copy to cpu
  gpu_cph *plain_gpu;
  gpu_cph *cipher_gpu;
  gpu_cph *cipher_res;
  cudaMallocAndSet((void **)&plain_gpu, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_gpu, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * count);

  extractPen(cipher_gpu, a, count, HostToDevice);
  for (int i = 0; i < count; i++)
    cudaMemcpy(plain_gpu + i, &((b + i)->encoding), sizeof(plain_t), cudaMemcpyHostToDevice);
  
  call_raw_mul(cipher_gpu, plain_gpu, cipher_res, count);
  for (int i = 0; i < count; i++) {
    cudaMemcpy((r + i)->cipher, cipher_res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    (r + i)->base = (a + i)->base;
    (r + i)->exponent = (a + i)->exponent + (b + i)->exponent;
  }

  cudaFree(plain_gpu);
  cudaFree(cipher_gpu);
  cudaFree(cipher_res);
}


void encrypt(FixedPointNumber *plain, gpu_cph *r, const uint32_t count, const bool obf) {
  // encrypt function.
  // parameters:
  //   plain: in cpu
  //   r : in cpu
  // steps:
  //   1. copy encoding to gpu
  //   2. perform raw encrypt
  //   3. copy back to result(on cpu)
  gpu_cph *raw_plain_gpu;
  gpu_cph *raw_cipher_gpu;
  unsigned int *obfs = NULL;
  cudaMallocAndSet((void **)&raw_plain_gpu, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&raw_cipher_gpu, sizeof(gpu_cph) * count);
  memset(r, 0, sizeof(gpu_cph) * count);
  cudaMemset(raw_plain_gpu, 0, sizeof(gpu_cph) * count);
  dumpMem(plain[0].encoding, sizeof(plain_t));
  
  for (int i = 0; i < count; i++) {
    cudaMemcpy(raw_plain_gpu + i, &plain[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice);
  }
  
  if (obf) {
    cudaMallocManaged(&obfs, sizeof(unsigned int) * count);
    for (int i = 0; i < count; i++) obfs[i] = rand();
  }
   
  call_raw_encrypt_obfs(raw_plain_gpu, count, raw_cipher_gpu, obfs);

  cudaMemcpy(r, raw_cipher_gpu, sizeof(gpu_cph) * count, cudaMemcpyDeviceToHost);

  cudaFree(raw_plain_gpu);
  cudaFree(raw_cipher_gpu);
  if (obf) cudaFree(obfs);
}

// void encrypt_async(FixedPointNumber *plain, gpu_cph *r, const uint32_t count, const bool obf) {
//   gpu_cph *raw_plain_gpu;
//   gpu_cph *raw_cipher_gpu;
//   unsigned int *obfs = NULL;
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);
//   cudaMalloc(&raw_plain_gpu, sizeof(gpu_cph) * count);
//   cudaMalloc(&raw_cipher_gpu, sizeof(gpu_cph) * count);
//   memset(r, 0, sizeof(gpu_cph) * count);
//   cudaMemset(raw_plain_gpu, 0, sizeof(gpu_cph) * count);
  
//   for (int i = 0; i < count; i++) {
//     cudaMemcpyAsync(raw_plain_gpu + i, &plain[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice, stream);
//   }
  
//   if (obf) {
//     cudaMallocManaged(&obfs, sizeof(unsigned int) * count);
//     for (int i = 0; i < count; i++) obfs[i] = rand();
//   }
   
//   // call_raw_encrypt_obfs(raw_plain_gpu, count, raw_cipher_gpu, obfs);
//   int TPB = 128;
//   int IPB = TPB/TPI;
//   int block_size = (count + IPB - 1)/IPB;
//   int thread_size = TPB;

//   if (obf)
//     raw_encrypt_with_obfs<<<block_size, thread_size, 0, stream>>>(gpu_pub_key, err_report, \
//       raw_plain_gpu, raw_cipher_gpu, count, obfs);
//   else
//     raw_encrypt<<<block_size, thread_size, 0, stream>>>(gpu_pub_key, err_report, raw_plain_gpu,\
//       raw_cipher_gpu, count);

//   cudaMemcpyAsync(r, raw_cipher_gpu, sizeof(gpu_cph) * count, cudaMemcpyDeviceToHost, stream);

//   cudaFree(raw_plain_gpu);
//   cudaFree(raw_cipher_gpu);
//   if (obf) cudaFree(obfs);
//   cudaStreamDestroy(stream);
// }


void decrypt(PaillierEncryptedNumber *cipher, gpu_cph *r, const uint32_t count) {
  // perform decrypt
  // parameters:
  //   cipher: in cpu
  //   r : in cpu
  // steps:
  //   1. copy to gpu
  //   2. perform raw decrypt
  //   3. copy back to cpu
  gpu_cph *raw_cipher_gpu;
  gpu_cph *res_gpu;
  cudaMallocAndSet((void **)&raw_cipher_gpu, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&res_gpu, sizeof(gpu_cph) * count);

  for (int i = 0; i < count; i++)
    cudaMemcpy(raw_cipher_gpu + i, cipher[i].cipher, sizeof(gpu_cph), cudaMemcpyHostToDevice);
  
  call_raw_decrypt(raw_cipher_gpu, count, res_gpu);
  cudaMemcpy(r, res_gpu, sizeof(gpu_cph) * count, cudaMemcpyDeviceToHost);

  cudaFree(raw_cipher_gpu);
  cudaFree(res_gpu);
}


void sum(PaillierEncryptedNumber *cipher, PaillierEncryptedNumber *res, const uint32_t count) {
  // sum
  // parameters:
  //  cipher: in cpu
  //  r: in cpu
  // steps:
  //  1. copy to gpu
  //  2. align
  //  3. perform raw add on half
  //  4. loop until only one left

  // if count is odd, add one
  printf("count: %d\n", count);
  int32_t num_elem = count % 2 == 1 ? count + 1 : count;
  gpu_cph *ciphers_buf[2];
  gpu_cph *inc;
  cudaMallocAndSet((void **)&ciphers_buf[0], sizeof(gpu_cph) * num_elem);
  cudaMallocAndSet((void **)&ciphers_buf[1], sizeof(gpu_cph) * num_elem);
  cudaMallocAndSet((void **)&inc, sizeof(gpu_cph) * count);

  uint32_t max_exponent = 0;
  for (int i = 0; i < count; i++)
    max_exponent = max_exponent < cipher[i].exponent ? cipher[i].exponent : max_exponent;
  for (int i = 0; i < count; i++) {
    uint64_t tmp = (uint64_t) pow(cipher[i].base, max_exponent - cipher[i].exponent);
    //cudaMemcpy(inc + i, &tmp, sizeof(uint64_t), cudaMemcpyHostToDevice);
    set_ui64<CPH_BITS>(inc + i, tmp);
  }

  extractPen(ciphers_buf[0], cipher, count, HostToDevice);
  call_raw_mul(ciphers_buf[0], inc, ciphers_buf[1], count);
  
  if (count % 2 == 1)
    cudaMemset(ciphers_buf[1] + num_elem - 1, 1, 1);
    
  uint32_t dst_index = 0;
  gpu_cph *dst_buf;
  gpu_cph *src_buf;
  for (int i = num_elem / 2; i >= 1; i /= 2) {
    dst_buf = ciphers_buf[dst_index % 2];
    src_buf = ciphers_buf[(dst_index % 2 + 1) % 2];
    printf("check it %d\n", i);
    call_raw_add(src_buf, src_buf + i, dst_buf, i);
    dst_index += 1;
  }

  cudaMemcpy(res->cipher, dst_buf, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
  res->base = cipher[0].base;
  res->exponent = max_exponent;

  cudaFree(ciphers_buf[0]);
  cudaFree(ciphers_buf[1]);
  cudaFree(inc);
}

void matmul(PaillierEncryptedNumber *cipher_a, FixedPointNumber *plain_b, PaillierEncryptedNumber *r,\
   const uint32_t P, const uint32_t Q, const uint32_t R) {
  // perform matrix multiplication.
  // parameters:
  //  cipher_a: ciphers in cpu
  //  plain_b: plains in cpu
  //  r: result in cpu
  //  cipher_a has shape P * Q
  //  plain_b has shape Q * R, b is vertically flattened.
  // steps:
  //  1. copy cipher_a to GPU, plain_b to GPU
  //  2. align ciphers
  //  3. call_raw_matmul
  //  4. copy back to CPU with corresponding exponent
  gpu_cph *cipher_gpu = NULL;
  gpu_cph *plain_gpu = NULL;
  gpu_cph *cipher_res = NULL;
  
  // find the largest exponent
  uint32_t max_exponent = 0;
  for (int i = 0; i < P * Q; i++)
    max_exponent = max_exponent < cipher_a[i].exponent ? cipher_a[i].exponent : max_exponent;
  
  for (int i = 0; i < Q * R; i++)
    max_exponent = max_exponent < plain_b[i].exponent ? plain_b[i].exponent : max_exponent;
  
  // increase exponent
//   pen_increase_exponent_to(cipher_a, max_exponent, P * Q);
//   fpn_increase_exponent_to(plain_b, max_exponent, Q * R);

  cudaMallocAndSet((void **)&cipher_gpu, sizeof(gpu_cph) * P * Q);
  cudaMallocAndSet((void **)&plain_gpu, sizeof(gpu_cph) * Q * R);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * P * R);

  extractPen(cipher_gpu, cipher_a, P * Q, HostToDevice);
  for (int i = 0; i < Q * R; i++)
    cudaMemcpy(plain_gpu + i, &plain_b[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice);
  
  call_raw_matmul(cipher_gpu, plain_gpu, cipher_res, P, Q, R);

  for (int i = 0; i < P * R; i++) {
    cudaMemcpy(r[i].cipher, cipher_res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    r[i].exponent = 2 * max_exponent;
    r[i].base = cipher_a[0].base;
  }

}

void batch_matmul(PaillierEncryptedNumber *a, FixedPointNumber *b, PaillierEncryptedNumber *r, \
  uint32_t *size_a, uint32_t *size_b, const uint32_t dim) {
  // perform matmul
  // size_a: list of dimensions. i.e., 3,2,6,4
  // size_b: list of dimentions. i.e., 1,2,9,4
  // dim, dimentions of a and b, i.e. 4
  gpu_cph *cipher_a;
  gpu_cph *plain_b;
  gpu_cph *cipher_res;

  uint32_t max_exponent = 0;
  uint32_t count_a = 1;
  uint32_t count_b = 1;
  uint32_t count_res = 1;
  uint32_t P = size_a[dim - 2];
  uint32_t Q = size_a[dim - 1];
  uint32_t R = size_b[dim - 2];
  cudaStream_t streams[8];
  const uint32_t NUM_STREAMS = 8;
  for (int i = 0; i < NUM_STREAMS; i++) cudaStreamCreate(&streams[i]);

  for (int i = 0; i < dim; i++) count_a *= size_a[i];
  for (int i = 0; i < dim; i++) count_b *= size_b[i];
  for (int i = 0; i < dim - 2; i++) count_res *= size_a[i];
  count_res *= P * R;

  uint32_t stride_a = P * Q;
  uint32_t stride_b = Q * R;
  uint32_t stride_res = P * R;

  cudaMallocAndSet((void **)&cipher_a, sizeof(gpu_cph) * count_a);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * count_res);
  cudaMallocAndSet((void **)&plain_b, sizeof(gpu_cph) * count_b);
  
  for (int i = 0; i < count_a; i++)
    max_exponent = max_exponent < a[i].exponent ? a[i].exponent : max_exponent;
  for (int i = 0; i < count_b; i++)
    max_exponent = max_exponent < b[i].exponent ? b[i].exponent : max_exponent;
  
  // pen_increase_exponent_to()
  // fpn_increase_exponent_to()

  extractPen(cipher_a, a, count_a, HostToDevice);
  for (int i = 0; i < count_b; i++)
    cudaMemcpy(plain_b + i, &b[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice);

  uint32_t a_start = 0;
  uint32_t b_start = 0;
  uint32_t res_start = 0;

  dim3 threadPerBlock(PAILLIER_TPI, 4, 4); // TODO: remove hardcoded.
  uint32_t x_dim = ceil((double)P/(double)threadPerBlock.x);
  uint32_t y_dim = ceil((double)R/(double)threadPerBlock.y);

  dim3 blockPerGrid(x_dim, y_dim);

  for (int i = 0; i < dim; i++) {
    uint32_t dim_a = size_a[i];
    uint32_t dim_b = size_b[i];
    uint32_t loop_dim = dim_a < dim_b ? dim_b : dim_a;
    bool brdcst_a = dim_a == 1, brdcst_b = dim_b == 1;
    
    for (int j = 0; j < loop_dim; j++) {
      // call raw matmul
      cudaStreamSynchronize(streams[j % NUM_STREAMS]);
      raw_matmul<<<blockPerGrid, threadPerBlock, 0, streams[j % NUM_STREAMS]>>>(gpu_pub_key, \
      err_report, cipher_a + a_start, plain_b + b_start, cipher_res + res_start, P, Q, R);
      if (!brdcst_a || j == loop_dim - 1) a_start += stride_a;
      if (!brdcst_b || j == loop_dim - 1) b_start += stride_b;
      res_start += stride_res;
    }
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < count_res; i++) {
    cudaMemcpy(r[i].cipher, cipher_res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    r[i].exponent = max_exponent * 2;
    r[i].base = a[0].base;
  }

  for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
  cudaFree(cipher_a);
  cudaFree(plain_b);
  cudaFree(cipher_res);
}

}// extern "C"
