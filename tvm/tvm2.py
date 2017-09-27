from __future__ import absolute_import, print_function

import tvm
import numpy as np

def main():
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    print((type(C)))

    ctx = tvm.gpu(0)
    print(ctx)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    tvm.fadd_cuda(a, b, c)

    dev_module = fadd_cuda.imported_modules[0]
    print("-----CUDA code-----")
    print(dev_module.get_source())

if __name__=="__main__":
    main()
