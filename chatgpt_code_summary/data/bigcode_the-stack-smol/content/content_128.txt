from TASSELpy.java.lang.Number import Number, metaNumber
from TASSELpy.java.lang.Comparable import Comparable
from TASSELpy.utils.DocInherit import DocInherit
from TASSELpy.utils.Overloading import javaOverload,javaConstructorOverload
from TASSELpy.javaObj import javaObj
from TASSELpy.utils.helper import make_sig
from abc import ABCMeta
import numpy as np

java_imports = {'Long':'java/lang/Long',
                'String':'java/lang/String'}

class metaLong:
    __metaclass__ = ABCMeta
    @classmethod
    def __subclasshook__(cls, C):
        if C == np.int64:
            return True
        elif C == np.uint64:
            return True
        elif issubclass(C,Long):
            return True
        elif issubclass(C,long):
            return True
        else:
            return False

## Wrapper class for java.lang.Long
class Long(Comparable, Number):
    """
    Wrapper class for java.lang.Long
    """
    _java_name = java_imports['Long']
    @javaConstructorOverload(java_imports['Long'],
        (make_sig(['long'],'void'),(metaLong,)),
        (make_sig([java_imports['String']],'void'),(str,)))
    def __init__(self, *args, **kwargs):
        """
        Instantiates a new Long

        Signatures:

        Long(long value)
        Long(String s)

        Arguments:

        Long(long value)
           value -- The long to wrap in the object
        Long (String s)
           s -- The string representing the long
        """
        super(Long, self).__init__(*args, generic=(Long,), **kwargs)
    @DocInherit
    @javaOverload("compareTo",
                  (make_sig([java_imports['Long']],'int'),(metaLong,),None))
    def compareTo(self, *args):
        pass
    ###################################
    ## Numeric magic methods
    ###################################
    def __pos__(self):
        return Long(+self.toPrimative())
    def __neg__(self):
        return Long(-self.toPrimative())
    def __abs__(self):
        return Long(abs(self.toPrimativelongValue()))
    def __invert__(self):
        return Long(~self.toPrimative())
    def __floor__(self):
        return Long(np.floor(self.toPrimative()))
    def __ceil__(self):
        return Long(np.ceil(self.toPrimative()))
    ###################################
    ## Arithmetic magic methods
    ###################################
    def __add__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() + other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() + other))
    def __radd__(self, other):
        return self.__add__(other)
    def __iadd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() - other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() - other))
    def __rsub__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(other.toPrimative()-self.toPrimative()))
            else:
                return Long(np.int64(other-self.toPrimative()))
    def __isub__(self, other):
        return self.__sub__(other)
    def __mul__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() * other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() * other))
    def __rmul__(self, other):
        return self.__mul__(other)
    def __imul__(self, other):
        return self.__mul__(other)
    def __floordiv__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() // other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() // other))
    def __rfloordiv__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(other.toPrimative() // self.toPrimative()))
            else:
                return Long(np.int64(other // self.toPrimative()))
    def __ifloordiv__(self, other):
        return self.__floordiv__(other)
    def __div__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() / other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() / other))
    def __rdiv__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(other.toPrimative() / self.toPrimative()))
            else:
                return Long(np.int64(other / self.toPrimative()))
    def __idiv__(self, other):
        return self.__div__(other)
    def __mod__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() % other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() % other))
    def __rmod__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(other.toPrimative() % self.toPrimative()))
            else:
                return Long(np.int64(other % self.toPrimative()))
    def __imod__(self, other):
        return self.__mod__(other)
    def __pow__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() ** other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() ** other))
    def __rpow__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(other.toPrimative() ** self.toPrimative()))
            else:
                return Long(np.int64(other ** self.toPrimative()))
    def __ipow__(self, other):
        return self.__pow__(other)
    def __lshift__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() << other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() << other))
    def __rlshift__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(other.toPrimative() << self.toPrimative()))
            else:
                return Long(np.int64(other << self.toPrimative()))
    def __ilshift__(self, other):
        return self.__lshift__(other)
    def __rshift__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() >> other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() >> other))
    def __rrlshift__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(other.toPrimative() >> self.toPrimative()))
            else:
                return Long(np.int64(other >> self.toPrimative()))
    def __irshift__(self, other):
        return self.__rshift__(other)
    def __and__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() & other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() & other))
    def __rand__(self, other):
        return self.__and__(other)
    def __iand__(self, other):
        return self.__and__(other)
    def __or__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() | other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() | other))
    def __ror__(self, other):
        return self.__or__(other)
    def __ior__(self, other):
        return self.__or__(other)
    def __xor__(self, other):
        if isinstance(other, metaNumber):
            if isinstance(other, Number):
                return Long(np.int64(self.toPrimative() ^ other.toPrimative()))
            else:
                return Long(np.int64(self.toPrimative() ^ other))
    def __rxor__(self, other):
        return self.__xor__(other)
    def __ixor__(self, other):
        return self.__xor__(other)
    def __repr__(self):
        return "Long(%d)" % self.longValue()
    @DocInherit
    def toPrimative(self):
        return self.longValue()

