#! /usr/bin/env python
import os

os.mkdir('_testing')
os.chdir('_testing')
os.environ['MPLBACKEND'] = 'Agg'

from pymt.components import FrostNumberGeoModel as Model

model = Model()

for default in model.defaults:
    print('{name}: {val} {units}'.format(
        name=default[0], val=default[1][0], units=default[1][1]))
