#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:18:43 2020

@author: admangli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv').values

#%%

slot_machines = 10

#%% Random ad selection reward
import random

random_reward = 0

for i in range(len(dataset)):
    random_reward += dataset[i, random.randint(0, slot_machines - 1)]
    
#%%
number_of_ad_selections = [0]*slot_machines
reward_sums = [0]*slot_machines
ad_selection_sequence = []
UCB_range = np.zeros((slot_machines, 2)) # To get an idea of underlying distributino

# Generate initial seed, selecting each machine at least once randomly
for round in range(0, slot_machines):
    target_ad = random.randint(0, slot_machines - 1)
    while (number_of_ad_selections[target_ad] == 1):
        target_ad = random.randint(0, slot_machines - 1)
    number_of_ad_selections[target_ad] += 1
    reward_sums[target_ad] += dataset[round][target_ad]
    ad_selection_sequence.append(target_ad)
    
for round in range(slot_machines, len(dataset)):
    # Calculate Ri and Delta for each ad for the current round
    Ri = [0]*slot_machines
    Deltai = [0]*slot_machines
    max_UCB = 0
    target_ad = -1
    for ad in range(0, slot_machines):
        Ri[ad] = reward_sums[ad] / number_of_ad_selections[ad]
        Deltai[ad] = math.sqrt(1.5 * math.log(round + 1)/number_of_ad_selections[ad])
        UCB_range[ad, 0] = Ri[ad] + Deltai[ad]
        UCB_range[ad, 1] = Ri[ad] - Deltai[ad]
        if UCB_range[ad, 0] > max_UCB:     # Pick the ad with maximum UCB = Ri + Delta for current round
            max_UCB = UCB_range[ad, 0]
            target_ad = ad
    
    # Increment selected ad's reward and number of selections
    if target_ad != -1:
        number_of_ad_selections[target_ad] += 1
        reward_sums[target_ad] += dataset[round][target_ad]
        ad_selection_sequence.append(target_ad)
        
#%% Visualize results

# Plot a histogram showing how many times each ad was selected
plt.hist(ad_selection_sequence)
plt.xlabel('Ad Number')
plt.ylabel('Number of selections')
plt.title('Ad selection comparision')
plt.show()