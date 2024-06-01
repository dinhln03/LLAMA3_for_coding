#!/usr/bin/env python3

import sys
import psutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

if (len(sys.argv) < 2):
    print("usage: python3 driver.py <runs>")
    sys.exit(1)

input_file = 'fib_time'
output_file = "time.png"
runs = int(sys.argv[1])

def outlier_filter(data, threshold=2):
    data = np.array(data)
    z = np.abs((data - data.mean()) / data.std())
    return data[z < threshold]

def data_processing(data, n):
    catgories = data[0].shape[0]
    samples = data[0].shape[1]
    final = np.zeros((catgories, samples))

    for c in range(catgories):
        for s in range(samples):
            final[c][s] = \
                outlier_filter([data[i][c][s] for i in range(n)]).mean()
    return final


if __name__ == '__main__':
    Ys = []

    for i in range(runs):
        # bind process on cpu0
        subprocess.run('sudo taskset 0x1 ./client 2>&1 > /dev/null', shell=True)
        output = np.loadtxt(input_file, dtype='float').T
        Ys.append(np.delete(output, 0, 0))

    X = output[0]
    Y = data_processing(Ys, runs)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('perf', fontsize=16)
    ax.set_xlabel(r'$n_{th} fibonacci$', fontsize=16)
    ax.set_ylabel('time (ns)', fontsize=16)

    ax.plot(X, Y[0], marker='*', markersize=3, label='user')            # user
    ax.plot(X, Y[1], marker='+', markersize=3, label='kernel')          # kernel
    ax.plot(X, Y[2], marker='^', markersize=3, label='kernel to user')  # kernel to user

    ax.legend(loc = 'upper left')

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
