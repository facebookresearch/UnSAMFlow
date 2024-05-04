#!/usr/bin/env python3
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ["-std=c++11"]

nvcc_args = [
    "-gencode",
    "arch=compute_50,code=sm_50",
    "-gencode",
    "arch=compute_52,code=sm_52",
    "-gencode",
    "arch=compute_60,code=sm_60",
    "-gencode",
    "arch=compute_61,code=sm_61",
    "-gencode",
    "arch=compute_61,code=compute_61",
    "-ccbin",
    "/usr/bin/gcc-5",
]

setup(
    name="correlation_cuda",
    ext_modules=[
        CUDAExtension(
            "correlation_cuda",
            ["correlation_cuda.cc", "correlation_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
                "cuda-path": ["/usr/local/cuda-9.0"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
