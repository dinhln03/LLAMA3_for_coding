"""
# =============================================================================
# Creates the stiffness matrix as requested, using the material properties 
# provided in the TPD file (for v2020 files).
#
# Author: William Hunter, Tarcísio L. de Oliveira
# Copyright (C) 2008, 2015, William Hunter.
# Copyright (C) 2020, 2021, Tarcísio L. de Oliveira
# =============================================================================
"""
from __future__ import division

import os

from sympy import symbols, Matrix, diff, integrate, zeros
from numpy import abs, array

from ..utils import get_logger

logger = get_logger(__name__)

def create_K(_L, _E, _nu, _k, _t):
    # Initialize variables
    _a, _b, _c = _L, _L, _L  # element dimensions (half-lengths)
    _G = _E / (2 * (1 + _nu))  # modulus of rigidity
    _g = _E /  ((1 + _nu) * (1 - 2 * _nu))

    # SymPy symbols:
    x, y, z = symbols('x y z')
    N1, N2, N3, N4 = symbols('N1 N2 N3 N4')
    N5, N6, N7, N8 = symbols('N5 N6 N7 N8')
    xlist = [x, x, x, x, x, x, x, x]
    ylist = [y, y, y, y, y, y, y, y]
    zlist = [z, z, z, z, z, z, z, z]

    # Shape functions:
    N1 = (_a - x) * (_b - y) * (_c - z) / (8 * _a * _b * _c)
    N2 = (_a + x) * (_b - y) * (_c - z) / (8 * _a * _b * _c)
    N3 = (_a + x) * (_b + y) * (_c - z) / (8 * _a * _b * _c)
    N4 = (_a - x) * (_b + y) * (_c - z) / (8 * _a * _b * _c)
    N5 = (_a - x) * (_b - y) * (_c + z) / (8 * _a * _b * _c)
    N6 = (_a + x) * (_b - y) * (_c + z) / (8 * _a * _b * _c)
    N7 = (_a + x) * (_b + y) * (_c + z) / (8 * _a * _b * _c)
    N8 = (_a - x) * (_b + y) * (_c + z) / (8 * _a * _b * _c)

    # Create strain-displacement matrix B:
    B0 = tuple(map(diff, [N1, N2, N3, N4, N5, N6, N7, N8], xlist))
    B1 = tuple(map(diff, [N1, N2, N3, N4, N5, N6, N7, N8], ylist))
    B2 = tuple(map(diff, [N1, N2, N3, N4, N5, N6, N7, N8], zlist))
    B = Matrix([B0, B1, B2])

    # Create conductivity matrix:
    C = Matrix([[_k, 0, 0],
                [0, _k, 0],
                [0, 0, _k]])

    dK = B.T * C * B

    # Integration:
    logger.info('SymPy is integrating: K for H8T...')
    K = dK.integrate((x, -_a, _a),(y, -_b, _b),(z, -_c, _c))

    # Convert SymPy Matrix to NumPy array:
    K = array(K, dtype='double')
    C = array(C, dtype='double')

    # Set small (<< 0) values equal to zero:
    K[abs(K) < 1e-6] = 0

    # Return result:
    logger.info('Created stiffness matrix.')
    return K, B, C

# EOF H8T_K.py
