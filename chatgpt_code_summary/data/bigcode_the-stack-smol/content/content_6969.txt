import numpy as np
import matplotlib.pyplot as plt


def subf(n):
	if n <= 1:
		return 0
	elif n == 2:
		return 1
	
	return (n - 1) * (subf(n - 1) + subf(n - 2))


x = np.arange(1, 5, 1)

y = np.vectorize(subf)(x)

plt.plot(x, y)

plt.show()

