from __future__ import division

import sys


class Inf(float):
    __name__ = __name__
    __file__ = __file__

    @staticmethod
    def div(p, q):
        """
        ``p / q`` returning the correct infinity instead of
        raising ZeroDivisionError.
        """

        from math import copysign

        if q != 0.0:
            # Normal case, no infinities.
            return p / q
        elif p == 0.0:
            return p / q  # Doesn't return, raises an Exception.
        elif copysign(1, q) > 0:
            # q is +0.0, return inf with same sign as p.
            return copysign(inf, p)
        else:
            # q is -0.0, return inf with flipped sign.
            return copysign(inf, -p)


sys.modules[__name__] = inf = Inf("+inf")
