
import numpy as np
import matplotlib.pyplot as plt

ts = np.load("ts.npy")
ts -= ts[0]
xs = np.load("xs.npy")

print("ts shape: {0}".format(np.shape(ts)))
print("xs shape: {0}".format(np.shape(xs)))

plt.figure()
plt.scatter(ts, xs)
plt.show()
