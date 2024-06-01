# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:02:16 2018

@author: ning
"""

import os
working_dir = ''
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from utils import (cv_counts)
saving_dir = '../results/cv_counts'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

# Exp 1
for participant in ['AC', 'CL', 'FW', 'HB', 'KK', 'LM', 'MC', 'MP1', 'MP2', 'NN', 'RP','SD', 'TJ', 'TS', 'WT']:

    experiment = 'pos'
    df         = pd.read_csv(os.path.join(working_dir,'../data/PoSdata.csv'))
    df         = df[df.columns[1:]]
    df.columns = ['participant',
                  'blocks',
                  'trials',
                  'firstgabor',
                  'success',
                  'tilted',
                  'correct',
                  'RT_correct',
                  'awareness',
                  'RT_awareness',
                  'confidence',
                  'RT_confidence']
    df_sub = df[df['participant'] == participant]
    # make sure all the attributes are either 0 or 1
    df_sub.loc[:,'success'   ] = df_sub.loc[:,'success'   ].values - 1
    df_sub.loc[:,'awareness' ] = df_sub.loc[:,'awareness' ].values - 1
    df_sub.loc[:,'confidence'] = df_sub.loc[:,'confidence'].values - 1
    
    ##################################################################
    np.random.seed(12345)
    # use all 6 possible features
    feature_names = [
                     'correct',
                     'awareness',
                     'confidence',
                     'RT_correct',
                     'RT_awareness',
                     'RT_confidence']
    target_name = 'success'
    results = dict(sub              = [],
                   window           = [],
                   fold             = [],
                   )
    for name in feature_names:
        results['{}_high_cond_{}_low'.format(target_name,name)] = []
        results['{}_high_cond_{}_high'.format(target_name,name)] = []
    
    for n_back in np.arange(1,5): # loop through the number of trials looking back
    # this is the part that is redundent and the code is long
        results          = cv_counts(
                                          df_sub,
                                          feature_names,
                                          target_name,
                                          results,
                                          participant,
                                          experiment,
                                          window=n_back,
                                          
                                          ) 
    temp = pd.DataFrame(results)
    temp.to_csv(os.path.join(saving_dir,'Pos_6_features (cv_count)_{}.csv'.format(participant)),index=False) # save as a csv
    ################################################################################
    # use success, awareness, and confidence as features
    np.random.seed(12345)
    # use judgement features
    feature_names = [
                     'correct',
                     'awareness',
                     'confidence',]
    target_name = 'success'
    results = dict(sub              = [],
                   window           = [],
                   fold             = [],
                   )
    for name in feature_names:
        results['{}_high_cond_{}_low'.format(target_name,name)] = []
        results['{}_high_cond_{}_high'.format(target_name,name)] = []
    
    for n_back in np.arange(1,5): # loop through the number of trials looking back
    # this is the part that is redundent and the code is long
        results          = cv_counts(
                                          df_sub,
                                          feature_names,
                                          target_name,
                                          results,
                                          participant,
                                          experiment,
                                          window=n_back,
                                          
                                          ) 
    temp = pd.DataFrame(results)
    temp.to_csv(os.path.join(saving_dir,'Pos_3_1_features (cv_count)_{}.csv'.format(participant)),index=False) # save as a csv
    ###############################################################################
    # use reactimes as features
    np.random.seed(12345)
    # use all 6 possible features
    feature_names = [
                     'RT_correct',
                     'RT_awareness',
                     'RT_confidence']
    target_name = 'success'
    results = dict(sub              = [],
                   window           = [],
                   fold             = [],
                   )
    for name in feature_names:
        results['{}_high_cond_{}_low'.format(target_name,name)] = []
        results['{}_high_cond_{}_high'.format(target_name,name)] = []
    
    for n_back in np.arange(1,5): # loop through the number of trials looking back
    # this is the part that is redundent and the code is long
        results          = cv_counts(
                                          df_sub,
                                          feature_names,
                                          target_name,
                                          results,
                                          participant,
                                          experiment,
                                          window=n_back,
                                          
                                          ) 
    temp = pd.DataFrame(results)
    temp.to_csv(os.path.join(saving_dir,'Pos_RT_features (cv_count)_{}.csv'.format(participant)),index=False) # save as a csv


































