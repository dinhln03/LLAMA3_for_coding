from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
	name='ultmul', 
	ext_modules=[
	CUDAExtension(
		'ultMul_cuda',
[
			'ultMul_cuda.cpp',
			'ultMul_cuda_kernel.cu', #.cpp and .cu file must have different name
		])],
	cmdclass = {
		'build_ext': BuildExtension
	}
)
