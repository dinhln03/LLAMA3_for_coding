"""The tests for day17."""
from days import day17
from ddt import ddt, data, unpack
import unittest
import helpers


@ddt
class MyTestCase(unittest.TestCase): # noqa D101
    @data(
        [[
            'x=495, y=2..7',
            'y=7, x=495..501',
            'x=501, y=3..7',
            'x=498, y=2..4',
            'x=506, y=1..2',
            'x=498, y=10..13',
            'x=504, y=10..13',
            'y=13, x=498..504'], '57'])
    @unpack
    def test_example_a(self, test_input, expected): # noqa D102
        result = day17.part_a(test_input)
        self.assertEqual(result, expected)

    def test_answer_part_a(self): # noqa D102
        result = day17.part_a(helpers.get_file_contents('day17.txt'))
        self.assertEqual(result, '38021')

    @data(
        [[
            'x=495, y=2..7',
            'y=7, x=495..501',
            'x=501, y=3..7',
            'x=498, y=2..4',
            'x=506, y=1..2',
            'x=498, y=10..13',
            'x=504, y=10..13',
            'y=13, x=498..504'], '29'])
    @unpack
    def test_example_b(self, test_input, expected): # noqa D102
        result = day17.part_b(test_input)
        self.assertEqual(result, expected)

    def test_answer_part_b(self): # noqa D102
        result = day17.part_b(helpers.get_file_contents('day17.txt'))
        self.assertEqual(result, '32069')
