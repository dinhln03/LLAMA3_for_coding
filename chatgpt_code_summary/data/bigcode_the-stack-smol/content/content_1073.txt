#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Public Title.
 Doc str for module users

 .. moduleauthor:: Max Wu <http://maxwu.me>
 
 .. References::
    https://packaging.python.org/en/latest/distributing.html
    https://github.com/pypa/sampleproject
    
 .. Test Samples in doctest format
>>> None
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open(path.join(here, 'src', 'cistat', 'version.py')) as f:
    exec(f.read())
    VERSION = get_version()

setup(
    name='cistat',
    version=VERSION,
    description='A sample Python project',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/maxwu/cistat',

    # Author details
    author='Max Wu',
    author_email='maxwunj@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Testers, Developers',
        'Topic :: Software Test :: Statistic Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='CI Stat CircleCI',

    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=required,
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #package_data={
    #    'sample': ['package_data.dat'],
    #},

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'cistat-cli=cistat:cli_app',
        ],
    },
)
