'''
Test script for GrFNN, plotting the entrainment for a sin wave of changing frequency.

@author T. Kaplan
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

from gfnn import FrequencyType, FrequencyDist, ZParams, GrFNN
from plot import spectro_plot

# Construct our model by instantiating the class defined above
dim_in = 300
freq_dist = FrequencyDist(0.25, 6.0, dim_in, FrequencyType.LINEAR)
zparams = ZParams()
model = GrFNN(freq_dist, zparams, fs=160)

# Stimulus - 50 seconds of FHz sin, at a changing frequency (4->2)
F = 4
t1 = np.arange(0, 25, model.dt)
x1 = np.sin(2 * np.pi * F * t1) * 0.25
t2 = np.arange(25, 50, model.dt)
x2 = np.sin(2 * np.pi * F/2 * t2) * 0.25

# Prepare an initial plot
t = np.concatenate([t1, t2])
x = np.concatenate([x1, x2])
px = freq_dist.dist
py = np.zeros(px.shape)
plt.plot(px, py)

zs = np.empty((len(t), dim_in), dtype=np.complex64)
t0 = time.time()

for i in range(len(t)):
    out = model(x[i])
    zs[i] = out
    # Update plot:
    if i % 10 == 0:
        py = np.abs(out)
        plt.gca().lines[0].set_ydata(py)
        plt.gca().relim()
        plt.gca().autoscale_view()
        plt.pause(0.01)

t1 = time.time()
print('Took', round(t1-t0, 2))
plt.show()
