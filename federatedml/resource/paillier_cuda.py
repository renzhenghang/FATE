import ctypes
from ctypes import c_char_p, c_int32, create_string_buffer
from functools import wraps

CPH_BITS = 2048
_key_init = False

def _load_cuda_lib():
    lib = ctypes.CDLL('./paillier.so')
    return lib

_cuda_lib = _load_cuda_lib()

def check_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _key_init:
            func(*args, **kwargs)
    return wrapper

def init_gpu_keys(pub_key, priv_key):
    global _key_init
    if _key_init:
        print('key initiated, return.')
    c_n = c_char_p(pub_key.n.to_bytes(CPH_BITS//8, 'little'))
    c_g = c_char_p(pub_key.g.to_bytes(CPH_BITS//8, 'little'))
    c_nsquare = c_char_p(pub_key.nsquare.to_bytes(CPH_BITS//8, 'little'))
    c_max_int = c_char_p(pub_key.max_int.to_bytes(CPH_BITS//8, 'little'))

    _cuda_lib.init_pub_key(c_n, c_g, c_nsquare, c_max_int)

    c_p = c_char_p(priv_key.p.to_bytes(CPH_BITS//8, 'little'))
    c_q = c_char_p(priv_key.q.to_bytes(CPH_BITS//8, 'little'))
    c_psquare = c_char_p(priv_key.psquare.to_bytes(CPH_BITS//8, 'little'))
    c_qsquare = c_char_p(priv_key.qsquare.to_bytes(CPH_BITS//8, 'little'))
    c_q_inverse = c_char_p(priv_key.q_inverse.to_bytes(CPH_BITS//8, 'little'))
    c_hp = c_char_p(priv_key.hp.to_bytes(CPH_BITS//8, 'little'))
    c_hq = c_char_p(priv_key.hq.to_bytes(CPH_BITS//8, 'little'))

    _cuda_lib.init_priv_key(c_p, c_q, c_psquare, c_qsquare, c_q_inverse, c_hp, c_hq)

    _key_init = True

def init_err_report():
    _cuda_lib.init_err_report()

@check_key
def raw_encrypt_gpu(values, res_p):
    global _cuda_lib
    c_count = c_int32(len(values))
    array_t = c_int32 * len(values)
    c_array = array_t(*values)
    _cuda_lib.call_raw_encrypt(c_array, c_count, res_p)

@check_key
def raw_encrypt_obfs_gpu(values, rand_vals, res_p):
    global _cuda_lib
    c_count = c_int32(len(values))
    array_t = c_int32 * len(values)
    c_input = array_t(*values)
    c_rand_vals = array_t(*rand_vals)
    _cuda_lib.call_raw_encrypt_obfs(c_input, c_count, res_p, c_rand_vals)

    
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

def test_raw_encrypt(ins_num):
    from ..secureprotol.fate_paillier import PaillierPublicKey, PaillierPrivateKey, PaillierKeypair
    import random
    pub_key, priv_key = PaillierKeypair.generate_keypair(1024)
    init_gpu_keys(pub_key, priv_key)
    
    test_list = []
    standard_cipher = []
    for i in range(0, ins_num):
        t = random.randint(0, 2**32 - 1)
        test_list.append(t)
        standard_cipher.append(pub_key.raw_encrypt(t))
    res_p = create_string_buffer(ins_num * 2048 // 8)
    print('standard_cipher:', hex(standard_cipher[0]))
    raw_encrypt_gpu(test_list, res_p)
    print('gpu res:', repr(res_p.raw))

def test_raw_encrypt_obfs(ins_num):
    from ..secureprotol.fate_paillier import PaillierPublicKey, PaillierPrivateKey, PaillierKeypair
    import random
    pub_key, priv_key = PaillierKeypair.generate_keypair(1024)
    init_gpu_keys(pub_key, priv_key)
    
    test_list = []
    standard_cipher = []
    rand_vals = []
    for i in range(0, ins_num):
        t = random.randint(1, 2 ** 32 - 1)
        r = random.randint(1, 2 ** 32 - 1)
        test_list.append(t)
        rand_vals.append(r)
        standard_cipher.append(pub_key.raw_encrypt(t, r))

    print('standard_cipher:', hex(standard_cipher[0]))
    res_p = create_string_buffer(ins_num * 2048 // 8)
    raw_encrypt_obfs_gpu(test_list, rand_vals, res_p)
    gpu_cipher = []
    for i in range(0, ins_num):
        gpu_cipher.append(int.from_bytes(res_p.raw[i * 2048: (i + 1) * 2048], 'little'))

    print('gpu cipher:', hex(gpu_cipher[0]))

@check_key
def raw_decrypt_gpu(value):
    print('raw_decrypt')

if __name__ == '__main__':
    test_raw_encrypt_obfs(1)