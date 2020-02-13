import ctypes
from ctypes import c_char_p, c_int32
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
def raw_encrypt_gpu(values):
    c_count = c_int32(len(values))
    array_t = c_int32 * len(values)
    c_array = array_t(*values)
    _cuda_lib.call_raw_encrypt.restype = array_t
    res_p = _cuda_lib.call_raw_encrypt(c_array, c_count)

    ciphers = list(res_p)

    return ciphers
    

@check_key
def raw_decrypt_gpu(value):
    print('raw_decrypt')

if __name__ == '__main__':
    from ..secureprotol.fate_paillier import PaillierPublicKey, PaillierPrivateKey, PaillierKeypair
    import random
    pub_key, priv_key = PaillierKeypair.generate_keypair(1024)
    init_gpu_keys(pub_key, priv_key)
    test_list = []
    standard_cipher = []
    for i in range(0, 10):
        t = random.randint(0, 2**32 - 1)
        test_list.append(t)
        standard_cipher.append(pub_key.raw_encrypt(t))

    print('standard_cipher:', standard_cipher[:2])
    ciphers = raw_encrypt_gpu(test_list)
    print('')
    print(ciphers)