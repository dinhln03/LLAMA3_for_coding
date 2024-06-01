from sujson.Csv2json import Csv2json
import unittest
import filecmp

class ConvertCsvToJson(unittest.TestCase):
    def setUp(self):
        self.csv_to_json = Csv2json()

    def test_conversion(self):
        self.csv_to_json.load("files/Netflix.csv", delimiter=";")
        self.csv_to_json.convert("files/Netflix_jtest.json")
        self.assertTrue(filecmp.cmp("files/Netflix_jtest.json", "files/Netflix.json"))


if __name__ == '__main__':
    unittest.main()
