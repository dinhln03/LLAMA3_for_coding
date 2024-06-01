
import numpy as np
from math import *
import pymultinest
import sys
sys.path.insert(0, '/home/kochenma/pysb')
from pysb.integrate import Solver
import csv
import datetime
import time as tm
from model_550 import model
from pysb.pathfinder import set_path
set_path('bng', '/home/kochenma/BioNetGen')


data_object = []
with open('earm_data.csv') as data_file:
	reader = csv.reader(data_file)
	line = list(reader)
	for each in line:
		data_object.append(each)
for i, each in enumerate(data_object):
	if i > 0:
		for j, item in enumerate(each):
			data_object[i][j] = float(data_object[i][j])
data_object = data_object[1:]

time = []
for each in data_object:
	time.append(float(each[0]))

model_solver = Solver(model, time, integrator='vode', integrator_options={'atol': 1e-12, 'rtol': 1e-12})


def prior(cube, ndim, nparams):

	for k, every in enumerate(model.parameters):
		if every.name[-3:] == '1kf':
			cube[k] = cube[k]*4 - 4
		if every.name[-3:] == '2kf':
			cube[k] = cube[k]*4 - 8
		if every.name[-3:] == '1kr':
			cube[k] = cube[k]*4 - 4
		if every.name[-3:] == '1kc':
			cube[k] = cube[k]*4 - 1


postfixes = ['1kf', '2kf', '1kr', '1kc']


def loglike(cube, ndim, nparams):

	point = []
	cube_index = 0
	for k, every in enumerate(model.parameters):
		if every.name[-3:] in postfixes:
			point.append(10**cube[cube_index])
			cube_index += 1
		else:
			point.append(model.parameters[k].value)
	model_solver.run(point)
	failed = False
	for every in model_solver.yobs:
		for thing in every:
			if thing <= -0.00000001 or np.isnan(thing):
				failed = True
	if failed:
		return ['fail', -10000.0]
	else:
		parpc = model_solver.yobs[-1][6]/(model_solver.yobs[-1][1] + model_solver.yobs[-1][6])
		if (parpc > 0.0) and (parpc < 1.00000001):
			print log(parpc), point
			return ['sim', log(parpc)]


		else:
			return ['fail', -10000.0]
n_params = 0
for m, lotsa in enumerate(model.parameters):
	if lotsa.name[-3:] == '1kf':
		n_params += 1
	if lotsa.name[-3:] == '2kf':
		n_params += 1
	if lotsa.name[-3:] == '1kr':
		n_params += 1
	if lotsa.name[-3:] == '1kc':
		n_params += 1

start_time = tm.clock()
counts = [0, 0]
pymultinest.run(loglike, prior, n_params, evidence_tolerance=0.0001, n_live_points=16000, log_zero=-1e3, sampling_efficiency=0.3, outputfiles_basename='/scratch/kochenma/log_casp_act/550/', resume = False, verbose = False, counts=counts)

print counts
print 'start time', start_time
print 'end time', tm.clock()