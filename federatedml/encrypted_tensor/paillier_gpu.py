import ctypes
from ctypes import c_char_p, c_int32, c_int64, create_string_buffer, \
    Structure, c_bool, c_uint64, c_char, c_byte
import ctypes.util
from functools import wraps
import random
from .paillier_exception import *
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
from federatedml.secureprotol.fixedpoint import FixedPointNumber

CPH_BITS = 2048
PLN_BITS = 1024
CPH_BYTES = CPH_BITS // 8
PLN_BYTES = PLN_BITS // 8
_BASE = 16
_key_init = False
_pub_key = None
_priv_key = None

class c_FixedPointNumber(Structure):
    _fields_ = [
        ('encoding', c_byte * PLN_BYTES),
        ('exponent', c_int32),
        ('base', c_int32)
    ]
    def __init__(self, fpn):
        c_pln = (c_byte * PLN_BYTES).from_buffer(
            create_string_buffer(fpn.encoding.to_bytes(PLN_BYTES, 'little'))
        )
        super(c_FixedPointNumber, self).__init__(encoding=c_pln, exponent=fpn.exponent, base=fpn.BASE)

class c_PaillierEncryptedNumber(Structure):
    _fields_ = [
        ('cipher', c_byte * CPH_BYTES),
        ('exponent', c_int32),
        ('base', c_int32)
    ]
    def __init__(self, pen=None):
        if pen is not None:
            c_cipher = (c_byte * CPH_BYTES).from_buffer(create_string_buffer(pen.ciphertext(be_secure=False).to_bytes(CPH_BYTES, 'little')))
            super(c_PaillierEncryptedNumber, self).__init__(cipher=c_cipher, \
                exponent=pen.exponent, base=_BASE)
def _load_cuda_lib():
    path = '/home/zhenghang/work_dir/eggroll/lib/paillier_gpu.so'
    # print(path)
    if path is None:
        raise CudaLibLoadErr('Cuda Lib Not Found, please check LD_LIBRARY_PATH')
    lib = ctypes.CDLL(path)
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
    global _pub_key
    global _priv_key
    if _key_init:
        print('key initiated, return.')
    c_n = c_char_p(pub_key.n.to_bytes(CPH_BYTES, 'little'))
    c_g = c_char_p(pub_key.g.to_bytes(CPH_BYTES, 'little'))
    c_nsquare = c_char_p(pub_key.nsquare.to_bytes(CPH_BYTES, 'little'))
    c_max_int = c_char_p(pub_key.max_int.to_bytes(CPH_BYTES, 'little'))

    _cuda_lib.init_pub_key(c_n, c_g, c_nsquare, c_max_int)
    print('n in cpu:', hex(pub_key.n))

    c_p = c_char_p(priv_key.p.to_bytes(CPH_BYTES, 'little'))
    c_q = c_char_p(priv_key.q.to_bytes(CPH_BYTES, 'little'))
    c_psquare = c_char_p(priv_key.psquare.to_bytes(CPH_BYTES, 'little'))
    c_qsquare = c_char_p(priv_key.qsquare.to_bytes(CPH_BYTES, 'little'))
    c_q_inverse = c_char_p(priv_key.q_inverse.to_bytes(CPH_BYTES, 'little'))
    c_hp = c_char_p(priv_key.hp.to_bytes(CPH_BYTES, 'little'))
    c_hq = c_char_p(priv_key.hq.to_bytes(CPH_BYTES, 'little'))
    c_p_inverse = c_char_p(priv_key.p_inverse.to_bytes(CPH_BYTES, 'little'))

    _cuda_lib.init_priv_key(c_p, c_q, c_psquare, c_qsquare, c_q_inverse, c_hp, c_hq, c_p_inverse)
    _pub_key = pub_key
    _priv_key = priv_key

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


@check_key
def encrypt(values, obf=True):
    # values: list of Fixed Point Number
    global _cuda_lib
    global _pub_key
    # [print(v.encoding) for v in values]
    print('cpu encoding', values[0].encoding)
    fpn_list = [
        c_FixedPointNumber(v) for v in values
    ]

    fpn_array_t = c_FixedPointNumber * len(values)
    fpn_array = fpn_array_t(*fpn_list)


    pen_buffer = create_string_buffer(CPH_BYTES * len(values))

    _cuda_lib.encrypt(fpn_array, pen_buffer, c_int32(len(values)), c_bool(obf))

    cipher_list = get_int(pen_buffer.raw, len(values), CPH_BYTES)
    pen_list = [
        PaillierEncryptedNumber(_pub_key, cipher_list[i], values[i].exponent) \
            for i in range(len(values))
    ]

    return pen_list

@check_key
def decrypt(values):
    global _cuda_lib
    global _pub_key
    pen_list = [
        c_PaillierEncryptedNumber(v) for v in values
    ]
    pen_array = (c_PaillierEncryptedNumber * len(values))(*pen_list)
    plain_buffer = create_string_buffer(256 * len(values))
    _cuda_lib.decrypt(pen_array, plain_buffer, len(values))

    plains = get_int(plain_buffer.raw, len(values), 256)

    fpn_list = [
        FixedPointNumber(plains[i], values[i].exponent, _pub_key.n, _pub_key.max_int) for i in range(len(values))
    ]
    
    return fpn_list

@check_key
async def encrypt_async(value, obf=True, i=0):
    # values: list of Fixed Point Number
    global _cuda_lib
    # [print(v.encoding) for v in values]
    fpn_list = [
        c_FixedPointNumber(value)
    ]

    fpn_array_t = c_FixedPointNumber * 1
    fpn_array = fpn_array_t(*fpn_list)


    pen_buffer = create_string_buffer(CPH_BYTES * 1)

    print('start kernel', i)
    _cuda_lib.encrypt_async(fpn_array, pen_buffer, c_int32(1), c_bool(obf))
    print('finish kernel', i)

    cipher_list = get_int(pen_buffer.raw, 1, CPH_BYTES)
    pen = PaillierEncryptedNumber(_pub_key, cipher_list[0], value.exponent)

    return pen


@check_key
def add_impl(a_list, b_list):
    global _cuda_lib

    a_pen_list = [
        c_PaillierEncryptedNumber(v) for v in a_list
    ]
    a_pen_array = (c_PaillierEncryptedNumber * len(a_list))(*a_pen_list)

    b_pen_list = [
        c_PaillierEncryptedNumber(v) for v in b_list
    ]
    b_pen_array = (c_PaillierEncryptedNumber * len(b_list))(*b_pen_list)

    res_pen_array = (c_PaillierEncryptedNumber * len(b_list))()
    _cuda_lib.cipher_add_cipher(a_pen_array, b_pen_array, res_pen_array, len(a_list))

    res_list = [PaillierEncryptedNumber(None, ciphertext=int.from_bytes(bytearray(v.cipher), 'little'),\
         exponent=v.exponent) for v in res_pen_array]

    return res_list

@check_key
def mul_impl(a_list, b_list):
    global _cuda_lib
    a_pen_list = [
        c_PaillierEncryptedNumber(v) for v in a_list
    ]
    a_pen_array = (c_PaillierEncryptedNumber * len(a_list))(*a_pen_list)

    b_fpn_list = [
        c_FixedPointNumber(v) for v in b_list
    ] 
    b_fpn_array = (c_FixedPointNumber * len(b_list))(*b_fpn_list)

    res_pen_array = (c_PaillierEncryptedNumber * len(b_list))()
    _cuda_lib.plain_mul_cipher(b_fpn_array, a_pen_array, res_pen_array, len(a_list))

    res_list = [
        PaillierEncryptedNumber(None, ciphertext=int.from_bytes(bytearray(v.cipher), 'little'), \
            exponent=v.exponent) for v in res_pen_array
    ]
    return res_list


@check_key
def sum_impl(a_list):
    global _cuda_lib

    a_pen_list = [c_PaillierEncryptedNumber(v) for v in a_list]
    a_pen_array = (c_PaillierEncryptedNumber * len(a_list))(*a_pen_list)

    res_pen = (c_PaillierEncryptedNumber * 1)()
    print('list:', len(a_list))
    _cuda_lib.sum(a_pen_array, res_pen, c_int32(len(a_list)))

    res = [PaillierEncryptedNumber(None, ciphertext=int.from_bytes(bytearray(v.cipher), 'little'), \
        exponent=v.exponent) for v in res_pen]

    return res[0]


@check_key
def matmul_impl(a, b, a_shape, b_shape):
    global _cuda_lib
    
    p = a_shape[0]
    q = a_shape[1]
    r = b_shape[1]
    a_pen_list = [c_PaillierEncryptedNumber(v) for v in a]
    a_pen_array = (c_PaillierEncryptedNumber * len(a))(*a_pen_list)
    b_fpn_array = (c_FixedPointNumber * len(b))(*[c_FixedPointNumber(v) for v in b])

    res_pen = (c_PaillierEncryptedNumber * (p * r))()

    _cuda_lib.matmul(a_pen_array, b_fpn_array, res_pen, p, q, r)

    r_list = [PaillierEncryptedNumber(None, ciphertext=int.from_bytes(bytearray(v.cipher), 'little'), \
        exponent=v.exponent) for v in res_pen]

    return r_list
