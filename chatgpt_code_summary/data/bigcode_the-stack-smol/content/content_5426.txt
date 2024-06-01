import unittest
import example_module

class MainTest(unittest.TestCase):
    def test_add(self):
        self.assertEqual(example_module.add(1, 1), 2)

    def test_subtract(self):
        self.assertEqual(example_module.subtract(1, 1), 0)

if __name__ == '__main__':
    unittest.main()