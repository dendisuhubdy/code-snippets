import torch
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple

a = torch.randn(1,4,4).cuda()
b = torch.zeros(a.size()).cuda()

kernel = '''
extern "C"
__global__ void flip(float *dst, const float *src, int w, int total)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i >= total)
      return;
   dst[i] = src[(i / w) * w + (w - (i % w) - 1)];
}
'''
program = Program(kernel.encode(), 'flip.cu'.encode())
ptx = program.compile()

m = function.Module()
m.load(bytes(ptx.encode()))

f = m.get_function('flip')

Stream = namedtuple('Stream', ['ptr'])
s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

f(grid=(1,1,1), block=(1024,1,1), args=[b.data_ptr(), a.data_ptr(), a.size(-1), a.numel()],
  stream=s)

print(a)
print(b)
