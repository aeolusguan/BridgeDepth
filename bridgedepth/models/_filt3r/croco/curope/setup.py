# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# BridgeDepth V2: private extension name and standard TORCH_CUDA_ARCH_LIST support.

from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
setup(
    name="bridgedepth-filt3r-curope",
    ext_modules=[
        CUDAExtension(
            name="_bridgedepth_filt3r_curope",
            sources=[str(HERE / "curope.cpp"), str(HERE / "kernels.cu")],
            extra_compile_args={"nvcc": ["-O3", "--use_fast_math"], "cxx": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
