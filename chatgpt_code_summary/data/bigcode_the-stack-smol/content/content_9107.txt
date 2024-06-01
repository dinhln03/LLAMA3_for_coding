from unittest import TestCase
from estrutura_dados.ordering_algorithms import bubble_sort, quick_sort

numbers = [82, 9, 6, 16, 5, 70, 63, 64, 59, 72, 30, 10, 26, 77, 64, 11, 10, 7, 66, 59, 55, 76, 13, 38, 19, 68, 60, 42, 7, 51]

_sorted = [5, 6, 7, 7, 9, 10, 10, 11, 13, 16, 19, 26, 30, 38, 42, 51, 55, 59, 59, 60, 63, 64, 64, 66, 68, 70, 72, 76, 77, 82]

class BubbleSort(TestCase):
    def test_order(self):
        self.assertEqual(
            _sorted,
            bubble_sort(numbers=numbers)
        )
    
    def test_quick_sort(self):
        self.assertEqual(
            _sorted,
            quick_sort(numbers)
        )
