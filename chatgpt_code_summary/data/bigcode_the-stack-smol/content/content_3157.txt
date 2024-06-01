import unittest
from collections import namedtuple

import m

sample = """\
ab
ac

b
b\
"""

TestCase = namedtuple("TestCase", ["text", "output"])


class TestDec6(unittest.TestCase):

    def test_get_groups(self):
        cases = [
            TestCase(sample, [['ab', 'ac'], ['b', 'b']]),
        ]
        for c in cases:
            result = m.read_groups(c.text)
            self.assertEqual(result, c.output, c)

    def test_count_answers(self):
        cases = [
            TestCase(sample, [3, 1]),
        ]
        for c in cases:
            groups = m.read_groups(c.text)
            nrs = []
            for group in groups:
                result = m.union_answers(group)
                nrs.append(result)
            self.assertEqual(nrs, c.output, c)

    def test_count_intersection_answers(self):
        cases = [
            TestCase(sample, [1, 1]),
        ]
        for c in cases:
            groups = m.read_groups(c.text)
            nrs = []
            for group in groups:
                result = m.intersection_answers(group)
                nrs.append(result)
            self.assertEqual(nrs, c.output, c)


if __name__ == '__main__':
    unittest.main()
