from unittest import TestCase


class TestNumbers2Words(TestCase):
  from numbers2words import numbers2words

  # 1-10
  def test_one(self):
    self.assertEqual(numbers2words(1), "one")

  def test_two(self):
    self.assertEqual(numbers2words(2), "two")

  def test_three(self):
    self.assertEqual(numbers2words(3), "three")

  def test_four(self):
    self.assertEqual(numbers2words(4), "four")

  def test_five(self):
    self.assertEqual(numbers2words(5), "five")

  def test_six(self):
    self.assertEqual(numbers2words(6), "six")

  def test_seven(self):
    self.assertEqual(numbers2words(7), "seven")

  def test_eight(self):
    self.assertEqual(numbers2words(8), "eight")

  def test_nine(self):
    self.assertEqual(numbers2words(9), "nine")

  def test_ten(self):
    self.assertEqual(numbers2words(10), "ten")

  # 11-20
  def test_eleven(self):
    self.assertEqual(numbers2words(11), "eleven")

  def test_twelve(self):
    self.assertEqual(numbers2words(12), "twelve")

  def test_thirteen(self):
    self.assertEqual(numbers2words(13), "thirteen")

  def test_fourteen(self):
    self.assertEqual(numbers2words(14), "fourteen")

  def test_fifteen(self):
    self.assertEqual(numbers2words(15), "fifteen")

  def test_sixteen(self):
    self.assertEqual(numbers2words(16), "sixteen")

  def test_seventeen(self):
    self.assertEqual(numbers2words(17), "seventeen")

  def test_eighteen(self):
    self.assertEqual(numbers2words(18), "eighteen")

  def test_nineteen(self):
    self.assertEqual(numbers2words(19), "nineteen")

  def test_twenty(self):
    self.assertEqual(numbers2words(20), "twenty")


