from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Fibonacci',
    package_dir={'Fibonacci/functions_folder': ''},
    ext_modules=cythonize("fib_module.pyx"),
)
