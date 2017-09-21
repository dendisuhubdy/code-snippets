import pyopencl as cl
import numpy as np
import numpy.linalg as la

def main():
    a = np.random.rand(256**3).astype(np.float32)

    ctx = cl.create_some_context()

    queue = cl.CommandQueue(ctx)

    a_dev = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=a.nbytes)
    cl.enqueue_copy(queue, a_dev, a)

    prg = cl.Program(ctx, """
    __kernel void twice(__global float* a)
    {
        a[get_global_id(0)] *= 2;
    }
    """).build()

    prg.twice(queue, a.shape, (1,), a_dev)
    result = np.empty_like(a)
    cl.enqueue_copy(queue, result, a_dev)
    print(la.norm(result -2*a))

if __name__=="__main__":
    main()
