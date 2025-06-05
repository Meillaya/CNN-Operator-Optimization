from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_op_lib',
    ext_modules=[
        CUDAExtension('custom_op_lib', [
            'csrc/add.cpp',
            'csrc/add_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }) 