import pickle
import numpy as np

# pickle_file = 'experiment_pickle_12_0.15_5_0.075.p'
pickle_file = 'experiment_pickle_12_0.1_5_0.075.p'

content = pickle.load(open(pickle_file))

familys = content.keys()

for family in familys:
    collected = []
    measurements = content[family]
    for measurement in measurements:
        collected.append(np.mean(measurement[1]))
    print family, ':', round(np.median(collected), 3), '+-', round(np.percentile(collected, 75) - np.percentile(collected, 25), 3)
