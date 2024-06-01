# Author
# Angelica ACOSTA ARTETA

import unittest
from balance import summing, stringarray, need, weighting

class TestBalance(unittest.TestCase):

    def test_summing(self):
        self.assertEqual(summing([]), 0)
        self.assertEqual(summing([3]), 3)
        self.assertEqual(summing([1,1,1,1,1]), 5)
        self.assertEqual(summing([1,-1]), 0)
        self.assertEqual(summing([-2,-2]), -4)
    
    def test_stringarray(self):
        self.assertEqual(stringarray([]), "")
        self.assertEqual(stringarray([5]), "5")
        self.assertEqual(stringarray([1,1,1,1,1]), "1, 1, 1, 1, 1")
    
    def test_need(self):
        self.assertEqual(need(1), (1,0))
        self.assertEqual(need(0), (None,None))
        self.assertEqual(need(-4), (None,None))
        self.assertEqual(need(27), (27,3))

    def test_weighting(self):
        first_test=weighting(1)
        self.assertEqual(summing(first_test[0]), summing(first_test[1]))
        second_test=weighting(532)
        self.assertEqual(summing(second_test[0]), summing(second_test[1]))
        thrird_test=weighting(1000)
        self.assertEqual(summing(thrird_test[0]), summing(thrird_test[1]))
    

if __name__ == '__main__':
    unittest.main()