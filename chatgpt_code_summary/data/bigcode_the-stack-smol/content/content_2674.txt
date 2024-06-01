import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

lines = open("for_james.csv").read().splitlines()
data = [[float(x) for x in lines[i].split(", ")] for i in range(len(lines))]

# each item in data is a list of floats that can be passed to plt.hist

for i in range(9):
    plt.hist(data[i], bins=np.logspace(1, 3, 20))
    plt.title(f'Precipitating Energy Distribution at t = {i+0.5} sec')
    plt.xscale("log"); plt.yscale("log"); plt.xlabel('Energy (KeV)'); plt.ylabel('Number of Particles')
    plt.ylim(10,600); plt.xlim(10,1000) 
    plt.savefig(f'results/plots/preciphist{i}.png')
    plt.clf()