"""
Time complexity: O(n^2)
This sorting algorithm puts the smallest element in the first
place after the first iteration. Similarly, after the second 
iteration, the second smallest value becomes the second value
of the list. The process continues and eventually the list
becomes sorted.
"""


for i in range(n):
  for j in range(i+1, n):
    if num[i] > num[j]:
      num[i], num[j] = num[j], num[i]
