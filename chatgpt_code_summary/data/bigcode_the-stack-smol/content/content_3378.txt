# Fix the code so that there's no error!

def count_evens(start, end):
  """Returns the number of even numbers between start and end."""
  counter = start
  num_evens = 0
  while counter <= end:
    if counter % 2 == 0:
      num_evens += 1
    counter += 1
  return num_evens

def count_multiples(start, end, divisor):
  """Returns the number of multiples of divisor between start and end."""
  counter = start
  num_multiples = 0
  while counter <= end:
    if counter % divisor == 0:
      num_multiples += 1
    counter += 1
  return num_multiples

count_both = count_evens(10, 20) + count_multiples(10, 20, 3)
