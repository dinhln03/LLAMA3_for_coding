"""
Evolutionary optimization of something
"""

import random
import multiprocessing

import numpy as np
import numpy.random as npr

import matplotlib.pylab as plt

from tqdm import tqdm

from automata import SnowDrift


class EvolutionaryOptimizer(object):
    """ Optimize!
    """
    def __init__(self):
        """ Set some parameters
        """
        self.mutation_probability = 0.02

    def init(self, size):
        """ Generate initial population
        """
        raise NotImplementedError

    def get_fitness(self, obj):
        """ Compute fitness of individual of population
        """
        raise NotImplementedError

    def mutate(self, obj):
        """ Mutate single individual
        """
        raise NotImplementedError

    def crossover(self, mom, dad):
        """ Generate offspring from parents
        """
        raise NotImplementedError

    def run(self, size, max_iter=100):
        """ Let life begin
        """
        population = self.init(size)

        res = []
        for _ in tqdm(range(max_iter)):
            pop_fitness = [self.get_fitness(o) for o in population]

            # crossover best individuals and replace worst with child
            best_indiv = np.argpartition(pop_fitness, -2)[-2:]
            mom, dad = population[best_indiv]
            child = self.crossover(mom, dad)

            worst_indiv = np.argmin(pop_fitness)
            population[worst_indiv] = child

            # apply mutations
            mut = lambda o: \
                self.mutate(o) if random.random() < self.mutation_probability \
                else o
            population = np.array([mut(o) for o in population])

            res.append(
                (np.mean(population, axis=0), np.var(population, axis=0)))
        return res

class SnowdriftOptimizer(EvolutionaryOptimizer):
    """ Optimize snowdrift game by assuming each individual to be the pair of
        benefit and cost floats
    """
    def init(self, size):
        pop = []
        for _ in range(size):
            pop.append((random.uniform(0, 1), random.uniform(0, 1)))
        return np.array(pop)

    def crossover(self, mom, dad):
        return np.mean([mom, dad], axis=0)

    def mutate(self, obj):
        sigma = 0.05
        return (obj[0] * random.gauss(1, sigma), obj[1] * random.gauss(1, sigma))

    def get_fitness(self, obj):
        # setup system
        lattice = npr.random_integers(0, 1, size=(2, 1))
        model = SnowDrift(lattice)

        # generate dynamics
        iter_num = 100

        benefit, cost = obj
        res = list(model.iterate(iter_num, benefit=benefit, cost=cost))

        # cut off transient
        ss = res[-int(iter_num/10):]

        # compute fitness
        fit = -np.sum(ss)
        return fit

def plot_runs(runs):
    """ Plot population evolutions
    """
    ts = range(len(runs[0]))
    cmap = plt.get_cmap('viridis')
    for i, r in enumerate(runs):
        mean, var = zip(*r)
        bm, cm = zip(*mean)
        bv, cv = zip(*var)

        color = cmap(float(i)/len(runs))

        plt.errorbar(ts, bm, fmt='-', yerr=bv, c=color)
        plt.errorbar(ts, cm, fmt='--', yerr=cv, c=color)

    plt.title('population evolution overview')
    plt.xlabel('time')
    plt.ylabel('value')

    plt.ylim((0, 1))

    plt.plot(0, 0, '-', c='black', label='benefit value')
    plt.plot(0, 0, '--', c='black', label='cost value')
    plt.legend(loc='best')

    plt.savefig('result.pdf')
    plt.show()


def work(i):
    """ Handle one optimization case
    """
    opti = SnowdriftOptimizer()
    return opti.run(20)

def main():
    """ Setup environment
    """
    core_num = int(multiprocessing.cpu_count() * 4/5)
    print('Using %d cores' % core_num)

    with multiprocessing.Pool(core_num) as p:
        runs = [i for i in p.imap_unordered(work, range(10))]

    plot_runs(runs)

if __name__ == '__main__':
    main()
