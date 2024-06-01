class Jumble(object):
  def __init__(self):
    self.dict = self.make_dict()

  def make_dict(self):
    dic = {}
    f = open('/usr/share/dict/words', 'r')
    for word in f:
      word = word.strip().lower()
      sort = ''.join(sorted(word))
      dic[sort] = word 
    return dic

  def unjumble(self, lst):
    for word in lst:
      word = word.strip().lower()
      sorted_word = "".join(sorted(word))
      if sorted_word in self.dict:
        self.dict[sorted_word]
      else:
        return None


if __name__ == "__main__":
  f_list = ['prouot', 'laurr', 'jobum', 'lethem']
  s_list = ['siconu', 'tefon', 'tarfd', 'laisa']
  t_list = ['sokik', 'niumem', 'tenjuk', 'doore']
  unjumble = Jumble()
  print(unjumble.unjumble(f_list))
  print(unjumble.unjumble(s_list))
  print(unjumble.unjumble(t_list))