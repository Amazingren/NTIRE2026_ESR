"""
Span Attention CUDA Operator Setup

Optimized span attention implementation for super-resolution networks.
Supports both highly-optimized inference (bs=1) and flexible general use.
"""

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# CUDA source files (最终版本)
CUDA_SOURCES = [
    'csrc/span_attention.cpp',
    'csrc/span_attention_kernel_optimized.cu',    # 原始优化版本（16/28/32/48/52通道）
    'csrc/span_attention_kernel_general.cu',       # 通用版本（任意配置）
    'csrc/span_attention_kernel_templated.cu',     # 模板版本（16/24/56/64通道）
    'csrc/span_attention_kernel_opt2.cu',          # 高度优化版本（32/48通道，最终采用）
]

# CUDA compile flags
CUDA_FLAGS = [
    '-O3',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '--compiler-options', '-fPIC',
    '-use_fast_math',
    '-ftz=true',
    '-prec-div=false',
    '-prec-sqrt=false',
    '--std=c++17',
    # GPU architecture support
    '-gencode', 'arch=compute_70,code=sm_70',  # V100
    '-gencode', 'arch=compute_80,code=sm_80',  # A100
    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 30xx
    '-gencode', 'arch=compute_90,code=sm_90',  # H100
    '--maxrregcount=128',
]

setup(
    name='span_attention',
    version='0.4.0',
    description='Optimized span attention CUDA operator for super-resolution',
    ext_modules=[
        CUDAExtension(
            name='span_attention',
            sources=CUDA_SOURCES,
            extra_compile_args={
                'cxx': ['-O3', '--std=c++17'],
                'nvcc': CUDA_FLAGS,
            },
            include_dirs=[torch.utils.cpp_extension.include_paths()],
            libraries=['cudart'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    },
)
