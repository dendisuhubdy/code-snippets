from __future__ import absolute_import, print_function

import tvm
import numpy as np

def main():
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    print((type(C)))

if __name__=="__main__":
    main()
