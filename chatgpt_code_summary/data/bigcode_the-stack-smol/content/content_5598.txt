import math

t = int(raw_input())
for i in range(t) :
	n = int(raw_input())
	print math.factorial(n)

'''Why using math.factorial() is faster?
   beacuse many of the Python libraries are in C or C++ and not it Python.
   Hence the speed improves.'''
