import numpy as np

from sigman.analyzer import InvalidArgumentError

procedure_type = 'points'
description = (
"""Procedure calculate time of B point from equation: 
RB = 1.233RZ-0.0032RZ^2-31.59
where RZ - time between R and dz/dt max [ms]
          RB - time between R and B
Equation was proposed by D.L. Lozano in paper "Where to B in dZ/dt" (2007)
""")
author = 'mzylinski'
arguments = {
    }
default_arguments = {
    }
output_type = 'B'
required_waves = ['Signal']
required_points = [ 'R','dzdtmax']


def procedure(waves, points, begin_time, end_time, settings):
    wave = waves['Signal']
    R = points['R']
    dzdtmax = points['dzdtmax']

    r_x = []
    r_y = []

    for i in range(0,len(R)-1):
        data = wave.data_slice(R.data_x[i], R.data_x[i+1])

        
        RZ = (dzdtmax.data_x[i] - R.data_x[i])/wave.sample_length
        RB = 1.233*RZ -0.0032*(RZ*RZ)-31.59
        
        t = int(round(RB))
        if (t<0):
            t = 0
        r_y.append(data[t])
        r_x.append(R.data_x[i] + t*wave.sample_length)


    return r_x, r_y

def interpret_arguments(waves, points, arguments):
    output_arguments = {}
    for key, item in arguments.items():
        try:
            output_arguments[key] = float(item)
        except:
            raise InvalidArgumentError("{} is invalid.".format(arguments[key]))
    return output_arguments

def execute(waves, points, begin_time, end_time, arguments):
    arguments = interpret_arguments(waves, points, arguments)
    return procedure(waves, points, begin_time, end_time, arguments)