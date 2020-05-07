#ifndef PAILLIER_CUDA_OPERATOR
#define PAILLIER_CUDA_OPERATOR
#include "./common.h"

extern "C" {
void init_pub_key(void *n, void *g, void *nsquare, void *max_int);
void init_priv_key(void *p, void *q, void *psquare, void *qsquare, void *q_inverse, void *hp, void *hq);
void init_err_report();
void reset();
// low level
void call_raw_encrypt_obfs(gpu_cph *plains_on_gpu, const uint32_t count, gpu_cph *ciphers_on_gpu, uint32_t* rand_vals_gpu);
void call_raw_add(gpu_cph *cipher_a, gpu_cph *cipher_b, gpu_cph *cipher_res, const uint32_t count);
// void call_raw_mul(gpu_cph *cipher_a, plain_t *plain_b, gpu_cph *cipher_res, const uint32_t count);
void call_raw_decrypt(gpu_cph *cipher_gpu, const uint32_t count, gpu_cph *res);
// void call_raw_matmul(gpu_cph *cipher_gpu, plain_t *plain_b, gpu_cph *cipher_res, const uint32_t P, const uint32_t Q, const uint32_t R);

// high level
void cipher_align(PaillierEncryptedNumber *a, PaillierEncryptedNumber *b, const uint32_t count);
void pen_increase_exponent_to(PaillierEncryptedNumber *a, const uint32_t exponent, const uint32_t count);
void cipher_add_cipher(PaillierEncryptedNumber *a, PaillierEncryptedNumber *b, PaillierEncryptedNumber *r, const uint32_t count);
void plain_mul_cipher(FixedPointNumber *b, PaillierEncryptedNumber *a, PaillierEncryptedNumber *r, const int count);
void encrypt(FixedPointNumber *plain, gpu_cph *r, const uint32_t count, const bool obf);
void decrypt(PaillierEncryptedNumber *cipher, gpu_cph *r, const uint32_t count);
void sum(PaillierEncryptedNumber *cipher, PaillierEncryptedNumber *res, const uint32_t count);
void matmul(PaillierEncryptedNumber *cipher_a, FixedPointNumber *plain_b, PaillierEncryptedNumber *r, const uint32_t P, const uint32_t Q, const uint32_t R);
void batch_matmul(PaillierEncryptedNumber *a, FixedPointNumber *b, PaillierEncryptedNumber *r, uint32_t *size_a, uint32_t *size_b, const uint32_t dim);

}

#endif