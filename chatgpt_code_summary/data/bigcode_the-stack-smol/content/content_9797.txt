#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 09:10:56 2018

@author: gtucker
"""

import numpy as np
import datetime
from grainhill import GrainFacetSimulator
from grainhill import SlopeMeasurer
import landlab
from landlab.io.native_landlab import save_grid
import os


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory ' + directory)


params = {
    'grid_size' : (111, 81),
    'report_interval' : 5.0, 
    'output_interval' : 1.0e99, 
    'disturbance_rate' : 1.0e-4,
    'weathering_rate' : 0.0,
    'dissolution_rate': 0.0,
    'friction_coef' : 1.0,
    'fault_x' : -0.01, 
    'cell_width' : 0.5, 
    'grav_accel' : 9.8,
    }


# Open a file to record output:
d = datetime.datetime.today()
today_str = str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
results_file = open('results_v_vs_w' + today_str + '.csv', 'w')
results_file.write('Landlab version,' + landlab.__version__ + ',\n')


# Print header in file
results_file.write('Uplift interval (yr),Weathering rate '
                   + 'parameter (1/yr),Gradient (m/m),'
                   + 'Slope angle (deg)\n')


# Sweep through a range of dissolution rate parameters
for uplift_interval_exp in np.arange(2, 5.2, 0.2):
    for weath_exp in np.arange(-5, -1.8, 0.2):

        weath_rate = 10.0**weath_exp
        uplift_interval = 10.0**uplift_interval_exp
        params['uplift_interval'] = uplift_interval
        params['weathering_rate'] = weath_rate

        # Set run duration long enough for uplift of 150 rows
        params['run_duration'] = 100 * uplift_interval
        params['plot_interval'] = 10 * uplift_interval

        print('Uplift interval: ' + str(params['uplift_interval']) + ' 1/y')
        print('Weathering rate: ' + str(params['weathering_rate']) + ' 1/y')

        opname = ('tau' + str(int(round(10 * uplift_interval_exp))) + 'w' + str(int(round(10 * weath_exp))))
        create_folder(opname)
        params['plot_file_name'] = opname + '/' + opname

        gfs = GrainFacetSimulator(**params)
        gfs.run()

        sm = SlopeMeasurer(gfs)
        sm.pick_rock_surface()
        (m, b) = sm.fit_straight_line_to_surface()
        angle = np.degrees(np.arctan(m))

        results_file.write(str(uplift_interval) + ','  + str(weath_rate) + ','
                           + str(m) + ',' + str(angle) + '\n')
        results_file.flush()

        save_grid(gfs.grid, opname + '/' + opname + '.grid', clobber=True)

results_file.close()
