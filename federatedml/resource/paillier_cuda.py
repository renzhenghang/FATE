import ctypes
from ctypes import c_char_p, c_int32, create_string_buffer
from functools import wraps
import random

CPH_BITS = 2048
CPH_BYTES = CPH_BITS // 8
_key_init = False

def _load_cuda_lib():
    lib = ctypes.CDLL('./paillier.so')
    return lib

_cuda_lib = _load_cuda_lib()

def check_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _key_init:
            return func(*args, **kwargs)
        # TODO: Raise Error
    return wrapper

def init_gpu_keys(pub_key, priv_key):
    global _key_init
    if _key_init:
        print('key initiated, return.')
    c_n = c_char_p(pub_key.n.to_bytes(CPH_BYTES, 'little'))
    c_g = c_char_p(pub_key.g.to_bytes(CPH_BYTES, 'little'))
    c_nsquare = c_char_p(pub_key.nsquare.to_bytes(CPH_BYTES, 'little'))
    c_max_int = c_char_p(pub_key.max_int.to_bytes(CPH_BYTES, 'little'))

    _cuda_lib.init_pub_key(c_n, c_g, c_nsquare, c_max_int)

    c_p = c_char_p(priv_key.p.to_bytes(CPH_BYTES, 'little'))
    c_q = c_char_p(priv_key.q.to_bytes(CPH_BYTES, 'little'))
    c_psquare = c_char_p(priv_key.psquare.to_bytes(CPH_BYTES, 'little'))
    c_qsquare = c_char_p(priv_key.qsquare.to_bytes(CPH_BYTES, 'little'))
    c_q_inverse = c_char_p(priv_key.q_inverse.to_bytes(CPH_BYTES, 'little'))
    c_hp = c_char_p(priv_key.hp.to_bytes(CPH_BYTES, 'little'))
    c_hq = c_char_p(priv_key.hq.to_bytes(CPH_BYTES, 'little'))

    _cuda_lib.init_priv_key(c_p, c_q, c_psquare, c_qsquare, c_q_inverse, c_hp, c_hq)

    _key_init = True

def init_err_report():
    _cuda_lib.init_err_report()

def get_bytes(int_array, length):
    res = b''
    for a in int_array:
        res += a.to_bytes(length, 'little')
    
    return res


def get_int(byte_array, count, length):
    res = []
    for i in range(count):
        res.append(int.from_bytes(byte_array[i * length: (i + 1) * length], 'little'))
    return res


@check_key
def raw_encrypt_gpu(values):
    global _cuda_lib
    res_p = create_string_buffer(len(values) * CPH_BYTES)
    c_count = c_int32(len(values))
    array_t = c_int32 * len(values)
    c_array = array_t(*values)
    _cuda_lib.call_raw_encrypt(c_array, c_count, res_p)
    res = get_int(res_p.raw, len(values), CPH_BYTES)

    return res

@check_key
def raw_encrypt_obfs_gpu(values, rand_vals):
    global _cuda_lib
    res_p = create_string_buffer(len(values) * CPH_BYTES)
    c_count = c_int32(len(values))
    array_t = c_int32 * len(values)
    c_input = array_t(*values)
    c_rand_vals = array_t(*rand_vals)
    _cuda_lib.call_raw_encrypt_obfs(c_input, c_count, res_p, c_rand_vals)
    res = get_int(res_p.raw, len(values), CPH_BYTES)

    return res

@check_key
def raw_add_gpu(ciphers_a, ciphers_b, res_p):
    global _cuda_lib
    ins_num = len(ciphers_a) # TODO: check len(ciphers_a) == len(ciphers_b)
    in_a = get_bytes(ciphers_a, CPH_BYTES)
    in_b = get_bytes(ciphers_b, CPH_BYTES)

    c_count = c_int32(ins_num)

    _cuda_lib.call_raw_add(in_a, in_b, res_p, c_count)

@check_key
def raw_mul_gpu(ciphers_a, plains_b, res_p):
    global _cuda_lib
    ins_num = len(ciphers_a) # TODO: check len(ciphers_a) == len(plains_b)
    in_a = get_bytes(ciphers_a, CPH_BYTES)
    in_b = get_bytes(plains_b, 4)

    c_count = c_int32(ins_num)

    _cuda_lib.call_raw_mul(in_a, in_b, res_p, c_count)

@check_key
def raw_decrypt_gpu(ciphers):
    global _cuda_lib
    res_p = create_string_buffer(len(ciphers) * 4)
    ins_num = len(ciphers)
    in_cipher = get_bytes(ciphers, CPH_BYTES)

    c_count = c_int32(ins_num)

    _cuda_lib.call_raw_decrypt(in_cipher, c_count, res_p)

    return get_int(res_p.raw, ins_num, 4)


"""
def gen_instance(ins_num):
    return [random.randint(1, 2 ** 16 - 1) for i in range(ins_num)]
@check_key
def test_key(pub_key, priv_key):
    global _cuda_lib
    print('==========pub_key============')
    print('n', hex(pub_key.n))
    print('g', hex(pub_key.g))
    print('nsquare', hex(pub_key.nsquare))
    print('max_int', hex(pub_key.max_int))
    print('==========priv_key==========')
    print('p', hex(priv_key.p))
    print('q', hex(priv_key.q))
    print('psquare', hex(priv_key.psquare))
    print('qsquare', hex(priv_key.qsquare))
    print('q_inverse', hex(priv_key.q_inverse))
    print('hp', hex(priv_key.hp))
    print('hq', hex(priv_key.hq))
    res_p = create_string_buffer(CPH_BITS//8)
    _cuda_lib.test_key_cp(res_p)

def test_raw_encrypt(ins_num, pub_key, priv_key):

    test_list = gen_instance(ins_num)
    standard_cipher = [pub_key.raw_encrypt(t) for t in test_list]

    res_p = create_string_buffer(ins_num * 2048 // 8)
    print('standard_cipher:', hex(standard_cipher[0]))
    raw_encrypt_gpu(test_list, res_p)
    print('gpu res:', res_p.raw.hex())

def test_raw_encrypt_obfs(ins_num, pub_key, priv_key):
    test_list = gen_instance(ins_num)
    rand_vals = gen_instance(ins_num)
    standard_cipher = [pub_key.raw_encrypt(test_list[i], rand_vals[i]) for i in range(ins_num)]

    print('standard_cipher:', hex(standard_cipher[0]))
    res_p = create_string_buffer(ins_num * 2048 // 8)
    raw_encrypt_obfs_gpu(test_list, rand_vals, res_p)
    gpu_cipher = get_int(res_p.raw, ins_num, 2048 // 8)
    print('gpu cipher:', hex(gpu_cipher[0]))

def test_raw_decrypt(ins_num, pub_key, priv_key):
    test_list = gen_instance(ins_num)
    rand_vals = gen_instance(ins_num)

    enc_res = create_string_buffer(ins_num * 2048 // 8)
    dec_res = create_string_buffer(ins_num * 32 // 8)

    raw_encrypt_obfs_gpu(test_list, rand_vals, enc_res)

    enc_int = get_int(enc_res.raw, ins_num, 2048 // 8)

    raw_decrypt_gpu(enc_int, dec_res)

    dec_int = get_int(dec_res.raw, ins_num, 32 // 8)
    print('plains: ', test_list[0])
    print('dec_res: ', dec_int[0])


def test_raw_mul(ins_num, pub_key, priv_key):
    test_list1 = gen_instance(ins_num)
    rand_vals1 = gen_instance(ins_num)
    test_list2 = gen_instance(ins_num) # plains b

    enc_res1 = create_string_buffer(ins_num * 2048 // 8)

    raw_encrypt_obfs_gpu(test_list1, rand_vals1, enc_res1)

    ciphers_1 = get_int(enc_res1.raw, ins_num, 2048 // 8) # ciphers a


    raw_mul_buf = create_string_buffer(ins_num * 2048 // 8)
    raw_mul_gpu(ciphers_1, test_list2, raw_mul_buf)

    raw_mul_res = get_int(raw_mul_buf.raw, ins_num, 2048 // 8)
    dec_res_buf = create_string_buffer(ins_num * 2048 // 8)

    raw_decrypt_gpu(raw_mul_res, dec_res_buf)

    dec_res = get_int(dec_res_buf.raw, ins_num, 32 // 8)

    std_res = [test_list1[i] * test_list2[i] for i in range(ins_num)]

    print(dec_res[:2])
    print(std_res[:2])

def test_raw_add(ins_num, pub_key, priv_key):
    test_list1 = gen_instance(ins_num)
    rand_vals1 = gen_instance(ins_num)
    test_list2 = gen_instance(ins_num)
    rand_vals2 = gen_instance(ins_num)

    enc_res1 = create_string_buffer(ins_num * 2048 // 8)
    enc_res2 = create_string_buffer(ins_num * 2048 // 8)

    raw_encrypt_obfs_gpu(test_list1, rand_vals1, enc_res1)
    raw_encrypt_obfs_gpu(test_list2, rand_vals2, enc_res2)

    ciphers_1 = get_int(enc_res1.raw, ins_num, 2048 // 8)
    ciphers_2 = get_int(enc_res2.raw, ins_num, 2048 // 8)


    raw_add_buf = create_string_buffer(ins_num * 2048 // 8)
    raw_add_gpu(ciphers_1, ciphers_2, raw_add_buf)

    raw_add_res = get_int(raw_add_buf.raw, ins_num, 2048 // 8)
    dec_res_buf = create_string_buffer(ins_num * 32 // 8)

    raw_decrypt_gpu(raw_add_res, dec_res_buf)

    dec_res = get_int(dec_res_buf.raw, ins_num, 32 // 8)

    std_res = [test_list1[i] + test_list2[i] for i in range(ins_num)]

    print(dec_res[:2])
    print(std_res[:2])

if __name__ == '__main__':
    from ..secureprotol.fate_paillier import PaillierPublicKey, PaillierPrivateKey, PaillierKeypair
    pub_key, priv_key = PaillierKeypair.generate_keypair(1024)
    init_gpu_keys(pub_key, priv_key)
    test_raw_add(10, pub_key, priv_key)
"""