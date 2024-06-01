import time
from array import array
from itertools import product
from time import clock

import sys

from java.lang import Math

sys.path.append("./ABAGAIL.jar")

import java.util.Random as Random

from shared import ConvergenceTrainer
from opt.example import FourPeaksEvaluationFunction
from opt.ga import DiscreteChangeOneMutation, SingleCrossOver
import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC


# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/tsp.py

random = Random()
maxIters = [2, int(2e4+1)]
numTrials = 5
# Problem Sizes
N_list = [50, 100, 150, 200]


OUTPUT_DIRECTORY = "output"
outfile = OUTPUT_DIRECTORY + '/PEAKS4/{}/PEAKS4_{}_{}_LOG.csv'

# MIMIC
sample_list = [50, 100, 150, 200]
keepRate_list = [0.2, 0.3, 0.4, 0.5]
for t in range(numTrials):
    for samples, keepRate, m, N in product([100], [0.2], [0.5], N_list):
        fname = outfile.format('MIMIC', 'MIMIC_{}_{}'.format("problemSizes", N), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')

        T = N / 5
        fill = [2] * N
        ranges = array('i', fill)

        keep = int(samples*keepRate)

        ef = FourPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        df = DiscreteDependencyTree(m, ranges)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = ConvergenceTrainer(mimic)

        times = [0]
        for i in range(0, maxIters[0]):

            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(mimic.getOptimal())
            fevals = ef.fEvals
            ef.fEvals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

# RHC
restart_list = [20, 40, 60, 80]
for t in range(numTrials):
    for restart, N in product([80], N_list):
        fname = outfile.format('RHC', 'RHC_{}_{}'.format("problemSize", N), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')

        T = N / 5
        fill = [2] * N
        ranges = array('i', fill)

        ef = FourPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        rhc = RandomizedHillClimbing(hcp, restart)
        fit = ConvergenceTrainer(rhc)

        times = [0]
        for i in range(0, maxIters[0]):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(rhc.getOptimal())
            fevals = ef.fEvals
            ef.fEvals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

# SA
temperature_list = [1E1, 1E3, 1E5, 1E7, 1E9, 1E11]
CE_list = [0.35, 0.55, 0.75, 0.95]
for t in range(numTrials):
    for temperature, CE, N in product([1E11], [0.35], N_list):
        fname = outfile.format('SA', 'SA_{}_{}'.format("problemSizes", N), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')

        T = N / 5
        fill = [2] * N
        ranges = array('i', fill)

        ef = FourPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(temperature, CE, hcp)
        fit = ConvergenceTrainer(sa)

        times = [0]
        for i in range(0, maxIters[0]):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(sa.getOptimal())
            fevals = ef.fEvals
            ef.fEvals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

# GA
mateRate_list = [0.2, 0.4, 0.6, 0.8]
mutateRate_list = [0.2, 0.4, 0.6, 0.8]
for t in range(numTrials):
    for pop, mateRate, mutateRate, N in product([700], [0.2], [0.6], N_list):
        fname = outfile.format('GA', 'GA_{}_{}'.format("problemSizes", N), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        mate = int(pop*mateRate)
        mutate = int(pop*mutateRate)

        T = N / 5
        fill = [2] * N
        ranges = array('i', fill)

        ef = FourPeaksEvaluationFunction(T)
        odd = DiscreteUniformDistribution(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = ConvergenceTrainer(ga)

        times = [0]
        for i in range(0, maxIters[0]):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(ga.getOptimal())
            fevals = ef.fEvals
            ef.fEvals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)
