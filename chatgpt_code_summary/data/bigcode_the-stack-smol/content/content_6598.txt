'''
References:
 - An Outline of Set Theory, Henle
'''

from . import fol


class ElementSymbol(fol.ImproperSymbol):
    def __init__(self):
        fol.PrimitiveSymbol.__init__('∈')

    def symbol_type(self) -> str:
        return 'element of'

    @staticmethod
    def new() -> "ElementSymbol":
        return ElementSymbol()
