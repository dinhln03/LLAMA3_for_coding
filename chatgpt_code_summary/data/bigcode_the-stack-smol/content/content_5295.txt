import unittest
import progresspie


class TestCalculationMethods(unittest.TestCase):

    def test_is_black(self):
        self.assertEqual(progresspie.is_black(99, 99, 99), False)

        self.assertEqual(progresspie.is_black(0, 55, 55), False)
        self.assertEqual(progresspie.is_black(12, 55, 55), False)

        self.assertEqual(progresspie.is_black(13, 55, 55), True)
        self.assertEqual(progresspie.is_black(13, 45, 45), False)
        self.assertEqual(progresspie.is_black(87, 20, 40), True)


if __name__ == '__main__':
    unittest.main()
