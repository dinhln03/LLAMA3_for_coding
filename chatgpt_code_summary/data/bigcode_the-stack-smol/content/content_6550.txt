import unittest
from CsvReader import CsvReader
from Calculator import MyCalculator


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.calculator = MyCalculator()

    def test_instantiate_calculator(self):
        self.assertIsInstance(self.calculator, MyCalculator)

    def test_addition(self):
        test_data = CsvReader('src/csv/TestAddition.csv').data
        for row in test_data:
            result = float(row['Result'])
            self.assertEqual(self.calculator.add(row['Value 1'], row['Value 2']), result)
            self.assertEqual(self.calculator.result, int(row['Result']))

    def test_subtraction(self):
        test_data = CsvReader('src/csv/TestSubtraction.csv').data
        for row in test_data:
            result = float(row['Result'])
            self.assertEqual(self.calculator.subtract(row['Value 1'], row['Value 2']), result)
            self.assertEqual(self.calculator.result, int(row['Result']))

    def test_multiplication(self):
        test_data = CsvReader('src/csv/TestMultiplication.csv').data
        for row in test_data:
            result = float(row['Result'])
            self.assertEqual(self.calculator.multiply(row['Value 1'], row['Value 2']), result)
            self.assertEqual(self.calculator.result, int(row['Result']))

    def test_division(self):
        test_data = CsvReader('src/csv/TestDivision.csv').data
        for row in test_data:
            result = float(row['Result'])
            self.assertAlmostEqual(self.calculator.divide(row['Value 1'], row['Value 2']), result)
            self.assertAlmostEqual(self.calculator.result, float(row['Result']))

    def test_square(self):
        test_data = CsvReader('src/csv/TestSquare.csv').data
        for row in test_data:
            result = float(row['Result'])
            self.assertEqual(self.calculator.square(row['Value 1']), result)
            self.assertEqual(self.calculator.result, int(row['Result']))

    def test_square_root(self):
        test_data = CsvReader('src/csv/TestSquareRoot.csv').data
        for row in test_data:
            result = float(row['Result'])
            self.assertAlmostEqual(self.calculator.square_root(row['Value 1']), result)
            self.assertAlmostEqual(self.calculator.result, float(row['Result']))


if __name__ == '__main__':
    unittest.main()