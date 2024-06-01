#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.dont_write_bytecode=True
from distutils.core import setup

from pyxsltp import __version__

setup(
    name = "pyxsltp",
    version = __version__,
    py_modules = ['pyxsltp'],
    scripts = ['pyxsltp'],
)
