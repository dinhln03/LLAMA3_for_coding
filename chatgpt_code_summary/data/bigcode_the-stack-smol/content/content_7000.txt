from unittest import TestCase

from block import source, conjunction, negation, operator_of, nor
from components import rs_flip_flop
from simulation import Simulation


def sr_simulation(initial_s, initial_q):
    nor1, nor2, source_r, source_s, _, _ = rs_flip_flop(initial_q, initial_s)
    simulation = Simulation([source_s, source_r, nor1, nor2])
    return source_s, source_r, nor2.outputs[0], nor1.outputs[0], simulation


class SimulationTest(TestCase):
    def test_step_pushes(self):
        block = source()
        simulation = Simulation([block])

        simulation.run()
        self.assertEqual(1, block.outputs[0].value)

    def test_step_three_blocks(self):
        block1 = source()
        block2 = source()
        block3 = operator_of(conjunction, block1, block2)
        simulation = Simulation([block3, block2, block1])

        simulation.run()
        self.assertEqual(1, block3.outputs[0].value)

    def test_step_four_blocks(self):
        source1 = source()
        source2 = source()
        conj = operator_of(conjunction, source1, source2)
        neg = operator_of(negation, conj)
        simulation = Simulation([neg, conj, source2, source1])

        simulation.run()
        self.assertEqual(0, neg.outputs[0].value)

    def test_disconnected(self):
        block = nor()
        simulation = Simulation([block])

        simulation.run()
        self.assertEqual(None, block.outputs[0].value)

    def test_flip_flop_initial(self):
        self.assert_sr_flip_flop(1, 0, 1)  # set
        self.assert_sr_flip_flop(0, 1, 0)  # reset

    def test_flip_flop_multiple_steps(self):
        source_s, _, q, not_q, simulation = sr_simulation(1, 0)
        simulation.run()

        source_s.switch(1)
        simulation.run()

        self.assertEqual(1, q.value)
        self.assertEqual(0, not_q.value)

    def assert_sr_flip_flop(self, signal_s, signal_r, expected_q):
        _, _, q, not_q, simulation = sr_simulation(signal_s, signal_r)

        simulation.run()
        self.assertEqual(expected_q, q.value)
        self.assertEqual(not expected_q, not_q.value)
