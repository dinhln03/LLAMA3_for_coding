# -*- coding: utf-8 -*-
"""
WOLFEYES'S FRAMEWORK
Python 3 / OpenCV 3

This file describes some TypeCheking decorators.
Might be useless, but allows for almost very precise type checking,
 especially on keyworded args, which might help.
"""

# 'kargs' get the arguments and passes the decorator
def args(*types, **ktypes):
    """Allow testing of input types:
    argkey=(types) or argkey=type"""

    # The decorator modifies the function
    def decorator(func):

        def modified(*args, **kargs):
            # The modified fonction takes some args and kargs,
            # Which we need to check before passing it to func.
            # Works much like a proxy/firewall.

            # Check args:
            position = 1
            for arg, T in zip(args, types):
                if not isinstance(arg, T):
                    raise TypeError("Positional arg (%d) should be of type(s) %s, got %s" % (position, T, type(arg)))
                position += 1

            # Check kargs:
            for key, arg in kargs.items():
                if key in ktypes:
                    T = ktypes[key]
                    if not isinstance(arg, T):
                        raise TypeError("Keyworded arg '%s' should be of type(s) %s, got %s" % (key, T, type(arg)))

            # Actual result after check
            return func(*args, **kargs)

        # The decorator has decorated 'func' in 'modified'
        return modified

    # We pass the actual decorator right after getting the kargs
    return decorator

# 'ret' gets the possible output types
def ret(*types, **kargs):

    # Garde-fou
    if len(types) is 1 and not isinstance(types[0], type) and callable(types[0]):
        raise ValueError("You should not pass a function to TypeError.ret, maybe you did not write as '@TypeError.ret()...'")

    # This decorator will modify the function 'func'
    def decorator(func):

        def modified(*args, **kargs):
            # This is the modified function, 'modified' Works
            # like a proxy, just passes the arguments and
            # checks the type of the return's value

            ret = func(*args, **kargs)
            if not isinstance(ret, types):
                raise TypeError("The function %s is returning an abnormal value, expected %s but got %s" % (func, types, type(ret)))

            # func's return
            return ret

        # Modified function
        return modified

    # The actual decorator
    return decorator
