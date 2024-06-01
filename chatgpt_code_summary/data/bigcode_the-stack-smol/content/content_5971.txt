#!/usr/bin/env python
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pystadel',

    version='1.0.0',
    description='Class for sending SMSes using Stadel SMS gateway',
    long_description=long_description,
    url='https://github.com/luttermann/pystadel',

    author='Lasse Luttermann Poulsen',
    author_email='lasse@poulsen.dk',

    license='BSD-2-Clause',

    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',

        # It might work in other versions, but these are not testet.
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='sms stadel',

    py_modules=["stadel"],
)
