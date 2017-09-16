from pynvrtc.interface import NVRTCInterface, NVRTCException

src = ... ## Populate CUDA source code

inter = NVRTCInterface()

try:
    prog = inter.nvrtcCreateProgram(src, 'simple.cu', [], []);
    inter.nvrtcCompileProgram(prog, ['-ftz=true'])
    ptx = inter.nvrtcGetPTX(prog)
except NVRTCException as e:
    print('Error: %s' % repr(e))
