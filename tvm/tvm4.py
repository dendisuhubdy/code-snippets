
from __future__ import absolute_import, print_function

import tvm
import numpy as np

from tvm.contrib import cc
from tvm.contrib import util

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

s = tvm.create_schedule(C.op)

bx, tx = s[C].split(C.op.axis[0], factor=64)

s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

fadd_cuda = tvm.build(s, [A, B, C], "cuda", target_host="llvm", name="myadd")

ctx = tvm.gpu(0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd_cuda(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

dev_module = fadd_cuda.imported_modules[0]
print("-----CUDA code-----")
print(dev_module.get_source())

fadd1 = tvm.module.load(temp.relpath("myadd.so"))
fadd1_dev = tvm.module.load(temp.relpath("myadd.ptx"))
fadd1.import_module(fadd1_dev)
fadd1(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

fadd_cuda.export_library(temp.relpath("myadd_pack.so"))
fadd2 = tvm.module.load(temp.relpath("myadd_pack.so"))
fadd2(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

fadd_cl = tvm.build(s, [A, B, C], "opencl", name="myadd")
print("------opencl code------")
print(fadd_cl.imported_modules[0].get_source())
ctx = tvm.cl(0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd_cl(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
