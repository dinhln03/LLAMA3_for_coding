#
# This file is part of SEQGIBBS
# (https://github.com/I-Bouros/seqgibbs.git) which is released
# under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import scipy.stats
import numpy as np
import numpy.testing as npt

import seqgibbs as gibbs


def fun(x):
    """
    Function returning the parameters of the normal sampler.
        mean = product of elements of x
        variance = exp(|x|)/(1+exp(|x|)).
    """
    return np.prod(x), np.exp(np.sum(x))/(np.exp(np.sum(x))+1)


def another_fun(x):
    """
    Function returning the parameters of the normal sampler.
        mean = sum of elements of x
        variance = exp(|x|)/(1+exp(|x|)).
    """
    return np.sum(x), np.exp(np.sum(x))/(np.exp(np.sum(x))+1)


class TestSysGibbsAlgoClass(unittest.TestCase):
    """
    Test the 'SysGibbsAlgo' class.
    """
    def test__init__(self):
        sampler = gibbs.SysGibbsAlgo(num_dim=2)

        self.assertEqual(sampler.num_dim, 2)
        self.assertEqual(len(sampler.one_d_samplers), 0)
        self.assertEqual(len(sampler.chain_states), 1)
        npt.assert_array_equal(sampler.initial_state, np.zeros(2))
        npt.assert_array_equal(sampler.current_state, np.zeros(2))

        with self.assertRaises(TypeError):
            gibbs.SysGibbsAlgo('0', np.ones(2))

        with self.assertRaises(ValueError):
            gibbs.SysGibbsAlgo(0, np.ones(2))

        with self.assertRaises(ValueError):
            gibbs.SysGibbsAlgo(3, np.ones(2))

        with self.assertRaises(ValueError):
            gibbs.SysGibbsAlgo(2, [[1], [2]])

    def test_change_initial_state(self):
        sampler = gibbs.SysGibbsAlgo(num_dim=2)
        sampler.change_initial_state(new_state=np.array([2, 0]))

        npt.assert_array_equal(sampler.initial_state, np.array([2, 0]))

        with self.assertRaises(ValueError):
            sampler.change_initial_state(new_state=np.array([[1], [2]]))

        with self.assertRaises(ValueError):
            sampler.change_initial_state(new_state=np.array([1, 2, 0]))

    def test_add_1_d_sampler(self):
        sampler = gibbs.SysGibbsAlgo(num_dim=2, initial_state=np.array([2, 3]))
        new_1_d_sampler = gibbs.OneDimSampler(scipy.stats.norm.rvs, fun)

        sampler.add_1_d_sampler(new_1_d_sampler)
        self.assertEqual(len(sampler.one_d_samplers), 1)

        with self.assertRaises(TypeError):
            sampler.add_1_d_sampler(0)

    def test_run(self):
        sampler = gibbs.SysGibbsAlgo(
            num_dim=2, initial_state=np.array([2, 3]))

        # Feed in the two partial conditional samplers
        first_1_d_sampler = gibbs.OneDimSampler(scipy.stats.norm.rvs, fun)
        second_1_d_sampler = gibbs.OneDimSampler(
            scipy.stats.norm.rvs, another_fun)

        sampler.add_1_d_sampler(first_1_d_sampler)
        sampler.add_1_d_sampler(second_1_d_sampler)

        # Run 3 complete scan cycles of the algorithm
        sampler.run(num_cycles=3)
        last_state = sampler.chain_states[-1]

        self.assertEqual(len(sampler.chain_states), 4)
        self.assertEqual(len(last_state), len(sampler.initial_state))
        npt.assert_array_equal(last_state, sampler.current_state)

        # Run 3 more complete scan cycles of the algorithm
        sampler.run(num_cycles=3, mode='continue')
        self.assertEqual(len(sampler.chain_states), 7)

        # Rerun for 3 complete scan cycles of the algorithm
        sampler.run(num_cycles=3, mode='restart')
        self.assertEqual(len(sampler.chain_states), 4)

        with self.assertRaises(ValueError):
            sampler.run(num_cycles=3, mode='0')

        with self.assertRaises(TypeError):
            sampler.run(num_cycles=3.5)

        with self.assertRaises(ValueError):
            sampler.run(num_cycles=0, mode='restart')


class TestRandGibbsAlgoClass(unittest.TestCase):
    """
    Test the 'RandGibbsAlgo' class.
    """
    def test__init__(self):
        sampler = gibbs.RandGibbsAlgo(num_dim=2)

        self.assertEqual(sampler.num_dim, 2)
        self.assertEqual(len(sampler.one_d_samplers), 0)
        self.assertEqual(len(sampler.chain_states), 1)
        npt.assert_array_equal(sampler.initial_state, np.zeros(2))
        npt.assert_array_equal(sampler.current_state, np.zeros(2))

        with self.assertRaises(ValueError):
            gibbs.RandGibbsAlgo(3, dimen_prob=np.ones(2))

        with self.assertRaises(ValueError):
            gibbs.RandGibbsAlgo(2, dimen_prob=[[1], [2]])

    def test_change_dimen_prob(self):
        sampler = gibbs.RandGibbsAlgo(num_dim=3)
        sampler.change_dimen_prob(new_probs=np.array([2, 0, 1]))

        npt.assert_array_equal(
            sampler.dimen_prob,
            np.array([2, 0, 1])/np.sum(np.array([2, 0, 1])))

        with self.assertRaises(ValueError):
            sampler.change_dimen_prob(new_probs=np.array([[2], [0], [1]]))

        with self.assertRaises(ValueError):
            sampler.change_dimen_prob(new_probs=np.array([2, 1]))

    def test_run(self):
        sampler = gibbs.RandGibbsAlgo(
            num_dim=2,
            initial_state=np.array([2, 3]),
            dimen_prob=np.array([2, 5]))

        # Feed in the two partial conditional samplers
        first_1_d_sampler = gibbs.OneDimSampler(scipy.stats.norm.rvs, fun)
        second_1_d_sampler = gibbs.OneDimSampler(
            scipy.stats.norm.rvs, another_fun)

        sampler.add_1_d_sampler(first_1_d_sampler)
        sampler.add_1_d_sampler(second_1_d_sampler)

        # Run 3 complete scan cycles of the algorithm
        sampler.run(num_cycles=3)
        last_state = sampler.chain_states[-1]

        self.assertEqual(len(sampler.chain_states), 4)
        self.assertEqual(len(last_state), len(sampler.initial_state))
        npt.assert_array_equal(last_state, sampler.current_state)

        # Run 3 more complete scan cycles of the algorithm
        sampler.run(num_cycles=3, mode='continue')
        self.assertEqual(len(sampler.chain_states), 7)

        # Rerun for 3 complete scan cycles of the algorithm
        sampler.run(num_cycles=3, mode='restart')
        self.assertEqual(len(sampler.chain_states), 4)
