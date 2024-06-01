"""Tests for solver module
"""

# from mathgrid import solver
from mathgrid import solver


def test_calculator_01():
    assert solver.calculator('=((1+3)*2)/(6-4)') == 4
    assert solver.calculator('((1+3)*2)/(6-4)') == '((1+3)*2)/(6-4)'
    assert solver.calculator('=hola') == 'hola'
