
import ctypes

libtest = ctypes.cdll.LoadLibrary("libmolecular_integral.so")

def debug_print():
    a = libtest.debug_print()
    print(a)