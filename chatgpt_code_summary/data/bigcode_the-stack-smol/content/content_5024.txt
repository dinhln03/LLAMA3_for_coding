import os
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md") as f:
    readme = f.read()

classifiers = [
    'Intended Audience :: Developers',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    ]

setup(
    name = "tinypandas",
    version = "0.0.1",
    description = "A small pure python library with Pandas like API",
    long_description = readme,
    packages = ['tinypandas', 'tinypandas.tests'],
    package_dir = { 'tinypandas' : 'src', 'tinypandas.tests' : 'tests' },
    install_requires = [ ],
    author = "@lexual, Dilawar Singh <dilawars@ncbs.res.in>",
    maintainer = "Dilawar Singh",
    maintainer_email = "dilawars@ncbs.res.in",
    url = "http://github.com/dilawar/",
    license='GPL?',
    classifiers=classifiers,
)
