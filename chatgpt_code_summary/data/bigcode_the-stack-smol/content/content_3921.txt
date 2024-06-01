import os
import time
import sys

total = 0.0
N = 10001 # number of cases (seeds) to test

for seed in range(1,N):


    #vis_command = "java TerrainCrossingVis -exec \"/home/dawid/TopCoder/TerrainCrossing/./TerrainCrossing\" -novis -seed "
    vis_command = "java TerrainCrossingVis -exec \"$PWD/./TerrainCrossing\" -novis -seed "
    vis_command = vis_command + str(seed)

    start_time = time.time()
    output = os.popen(vis_command).readlines()
    finish_time = time.time()
    time_elapsed = finish_time - start_time

    if(time_elapsed > 10.0):
        print("Exiting...")
        sys.exit()

    print("Case " + str(seed-1) + " time: " + str(time_elapsed) + " score: " + str(float(output[0][:-1])), end="\n")
    
    total = total + float(output[0][:-1])

    #total = total + float(output[-1])

mean = total/(N-1)
print("Mean score: " + str(mean))
