#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import unittest
import code

class TestDay01(unittest.TestCase):
    # Part 01
    def test_example01(self):
        expense_report = [1721, 299]
        expected = 514579

        result = code.part01(expense_report)
        self.assertEqual(result, expected)

    # Don't count a 2020/2 value twice
    def test_duplicate(self):
        expense_report = [1010, 1721, 299]
        expected = 514579

        result = code.part01(expense_report)
        self.assertEqual(result, expected)

    # Part 02
    def test_example02(self):
        expense_report = [979, 366, 675]
        expected = 241861950

        result = code.part02(expense_report)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
