import ctypes
from ctypes import c_char_p

dll = ctypes.CDLL('./paillier.so')

n = 2 ** 2048 - 1
g = 2 ** 2048 - 1
max_int = 2 ** 2048 - 1
nsquare = 2 ** 2048 - 1

c_n = c_char_p(n.to_bytes(256, 'little'))
c_g = c_char_p(g.to_bytes(256, 'little'))
c_max_int = c_char_p(max_int.to_bytes(256, 'little'))
c_nsquare = c_char_p(nsquare.to_bytes(256, 'little'))
