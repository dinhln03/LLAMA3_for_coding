# example of except/from exception usage

try:
    1 / 0
except Exception as E:
    raise NameError('bad') from E

"""
Traceback (most recent call last):
  File "except_from.py", line 4, in <module>
    1 / 0
ZeroDivisionError: division by zero

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "except_from.py", line 6, in <module>
    raise NameError('bad') from E
NameError: bad
"""

# *implicit related exception
# try:
#     1 / 0
# except:
#     wrongname # NameError

# raise <exceptionnsme> from None complitly stops relation of exception
