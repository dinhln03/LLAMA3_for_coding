import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Distutils import build_ext
import numpy as np
from os.path import join as pjoin
from setup_cuda import cuda_setup

mpi_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
mpi_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

nvcc = find_in_path('nvcc', os.environ['PATH'])
if isinstance(nvcc, str):
    print('CUDA')
    # setup(name='PackageName',
    #       author='Nina Herrmann',
    #       version='1.0',
    #       description='This is a package for Muesli',
    #       ext_modules=cythonize(cuda_setup.get_module()),
    #       cmdclass={'build_ext': cuda_setup.custom_build_ext()}
    #       )
else:
    module = Extension('_da', sources=['da.cxx', 'da_wrap.cxx'],
                       include_dirs=[np.get_include(), 'src'],
                       library_dirs=['/usr/include/boost/'],
                       language="c++",
                       swig_opts=['-c++'],
                       libraries=['/usr/include/boost/chrono'],
                       extra_compile_args=(["-fopenmp"] + mpi_compile_args),
                       extra_link_args=(["-fopenmp"] + mpi_link_args)
                       )

    setup(name='da',
          author='Nina Herrmann',
          version='1.0',
          description='This is a package for Muesli',
          ext_modules=[module],
          py_modules=["da"]
          )
