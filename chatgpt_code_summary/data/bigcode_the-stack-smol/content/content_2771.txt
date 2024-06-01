
import math

def is_prime(num):
  if num < 2:
    return False
  for i in range(num):
    if i < 2:
      continue
    if num % i == 0:
      return False
  return True

def get_nth_prime(n):
  cnt = 0
  i = 0
  while cnt < n:
    i += 1
    if is_prime(i):
      cnt += 1
  return i  

if __name__ == '__main__':
  #print get_nth_prime(6)
  print get_nth_prime(10001)

