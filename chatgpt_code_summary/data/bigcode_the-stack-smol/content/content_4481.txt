# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:25:47 2018

@author: Steven
"""
import sys
import argparse

from radioxenon_ml.read_in import ml_matrix_composition as mlmc
from radioxenon_ml.solve import iterate
import numpy as np
"""
import radioxenon_ml.read_in.ml_matrix_composition
import radioxenon_ml.solve.iterate
import radioxenon_ml.solve.variance
"""
"""the master file for the radioxenon_ml package"""  
parser = argparse.ArgumentParser(description='This is the master file for running the maximum likelihood package.')
parser.add_argument('-o', '--offset', 
                    type=int,
                    default=84,
                    help='where to start the file selection from list of test files'
                    )
args = parser.parse_args(sys.argv[1:])

spectrum_file_location = 'radioxenon_ml/test_files/test'
offset = args.offset
err = 0.01                              #acceptable error in normalized activity
scale_array = np.array([1,1,1,1])       #Should have elements equal to the number of isotopes
#scale_array = np.array([0.561,0.584,0.9,0.372,0.489,0.489,1])   #scaling factor for each simulation file
                                                              #currently taken from (Czyz, 2017)
n = np.shape(scale_array)[0]            #number of simulated spectra

simulation, experiment, totcount = mlmc.form_matrix(spectrum_file_location,scale_array,n,offset);    #known issue: requires UTF-8 encoding
#simulation, experiment = mlmc.scale_matrix(simulation_unscaled,experiment_unscaled,)

A,J,K,q=iterate.iterate(simulation, experiment, err)
print("\n_____________________________________\nTotal activity percents = " + str(A*100))