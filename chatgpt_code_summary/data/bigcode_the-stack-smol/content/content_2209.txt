"""
This file imports `__all__` from the solvers directory, thus populating the solver registry.
"""

from pysperf.solvers import *
from .config import solvers

__all__ = ['solvers']
