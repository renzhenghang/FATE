
from eggroll.roll_paillier_tensor import paillier_gpu

from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierEncryptedNumber
from federatedml.secureprotol.fixedpoint import FixedPointNumber
from federatedml.secureprotol import gmpy_math
import asyncio
import random
import numpy as np
import threading

def load(data):
    return data


def dump(data):
    return data


def load_pub_key(pub):
    return pub


def dump_pub_key(pub):
    return pub


def load_prv_key(priv):
    return priv


def dump_prv_key(priv):
    return priv


def num2Mng(data, pub):
    shape = data.shape
    d_list = data.flatten()
    res = paillier_gpu.encrypt(d_list, obf=False) # Fast Encrypt
    return np.reshape(res, shape)

def brdcst(data1, data2):
    shape_1 = data1.shape
    shape_2 = data2.shape
    brd_1 = data1
    brd_2 = data2

    def apply_brdcst(data_s, data_g):
        # data_s has shorter shape length
        diff = len(data_g.shape) - len(data_s.shape)
        shape_s = data_s.shape
        shape_g = data_g.shape
        shape_g_trunc = shape_g[diff:]
        for i in reversed(range(len(shape_s))):
            if shape_s[i] == 1:
                data_s = np.repeat(data_s, shape_g_trunc[i], axis=i)
            elif shape_g_trunc[i] == 1:
                data_g = np.repeat(data_g, shape_s[i], axis=i+diff)
            elif shape_g_trunc[i] != shape_s[i]:
                raise ValueError("shape cannot align")
        for i in reversed(range(diff)):
            data_s = np.expand_dims(data_s, axis=0)
            data_s = np.repeat(data_s, shape_g[i], axis=0)

        return data_s, data_g
    
    if len(shape_1) < len(shape_2):
        return apply_brdcst(brd_1, brd_2)
    else:
        return apply_brdcst(brd_2, brd_1)


def add(x, y, pub):
    """
    if x and y are paillier tensor, align, add.
    """
    # steps:
    #   flatten the numpy array
    #   align
    #   add together
    x_shape = x.shape
    y_shape = y.shape
    # TODO: brdcst
    if x_shape == y_shape:
        x_list = x.flatten()
        y_list = y.flatten()
        res = paillier_gpu.add_impl(x_list, y_list)
        return np.reshape(res, x_shape)
    else:
        brd_x, brd_y = brdcst(x, y)
        return add(brd_x, brd_y, pub)


def scalar_mul(x, s, pub):
    """
    scala multiplication of x(vector) and s(scala)
    x: numpy array of PaillierEncryptedNumber
    y: FixedPointNumber
    """
    x_shape = x.shape
    x_flatten = np.flatten(x)
    s_array = np.array([s for _ in range(len(x_flatten))])
    
    res = paillier_gpu.mul_impl(x_flatten, s_array)

    return np.reshape(res, x_shape)


def mul(x, y, pub):
    """
    return the result of x * s (sematic of "*" explained in numpy)
    x: numpy array of PaillierEncryptedNumber
    y: numpy array of FixedPointNumber
    """
    x_shape = x.shape
    y_shape = y.shape
    if x_shape == y_shape:
        x_flatten = np.flatten(x)
        y_flatten = np.flatten(y)
        res = paillier_gpu.mul_impl(x_flatten, y_flatten)
        return np.reshape(res, x_shape)
    else:
        brd_x, brd_y = brdcst(x, y)
        return mul(brd_x, brd_y, pub)


def vdot(x, v, pub):
    """
    return the dot product of two vectors
    x: numpy array of FixedPointNumber
    y: numpy array of PaillierEncryptedNumber
    """
    x_flatten = x.flatten()
    v_flatten = v.flatten()
    mul_res = paillier_gpu.mul_impl(v_flatten, x_flatten)

    return paillier_gpu.sum_impl(mul_res)


def matmul(x, y, _pub):
    """
    return the matrix multiplication of x and y
    x: numpy ndarray of PaillierEncryptedNumber
    y: numpy ndarray of FixedPointNumber
    """
    if x.shape[-1] != y.shape[-2]:
        pass # TODO: REPORT ERROR
    res = paillier_gpu.matmul_impl(x.flatten(), y.flatten(order='F'), x.shape, y.shape)

    return res


def transe(data):
    return data.T


def mean(data, pub):
    # return np.array([data.mean(axis=0)])
    d_flatten = data.flatten()
    res = paillier_gpu.sum_impl(d_flatten)
    n = len(d_flatten)
    return res * 1/n


def hstack(x, y, pub):
    return np.hstack((x, y))


def decryptdecode(data, pub, priv):
    if isinstance(data, np.ndarray):
        d_flatten = list(data.flatten())
        d_shape = data.shape
    else:
        d_flatten = data
        d_shape = (len(data),)

    res = paillier_gpu.decrypt(d_flatten)
    res_decode = [r.decode() for r in res]

    return np.reshape(res_decode, d_shape)


def encrypt_and_obfuscate(data, pub, obfs=False):
    data = np.vectorize(FixedPointNumber.encode)(data, pub.n, pub.max_int)
    if isinstance(data, np.ndarray):
        d_flatten = data.flatten()
        d_shape = data.shape
    else:
        d_flatten = data
        d_shape = (len(data),)
    res = paillier_gpu.encrypt(d_flatten, obfs)

    return np.reshape(res, d_shape)

async def encrypt_obf_async(data, pub, obfs=False):
    d_flatten = data.flatten()
    d_shape = data.shape
    # thread_list = 
    res = await asyncio.gather(*[paillier_gpu.encrypt_async(v) for v in d_flatten])
    return np.reshape(res, d_shape)

def keygen():
    pub, priv = PaillierKeypair().generate_keypair()
    paillier_gpu.init_gpu_keys(pub, priv)

    return pub, priv
