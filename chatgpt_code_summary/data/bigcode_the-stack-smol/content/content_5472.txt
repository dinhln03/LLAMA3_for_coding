#
# LeetCode
#
# Problem - 581
# URL - https://leetcode.com/problems/shortest-unsorted-continuous-subarray/
#

class Solution:
  def findUnsortedSubarray(self, arr: List[int]) -> int:
    if (not arr):
      0

    index1 = -1
    index2 = -1

    for i in range(1, len(arr)):
      if (arr[i] < arr[i-1]):
        index1 = i-1
        break

    for i in range(len(arr)-2, -1, -1):
      if (arr[i] > arr[i+1]):
        index2 = i+1
        break

    if (index1 == -1):
      return 0
    else:
      maxSubArr = max(arr[index1:index2+1])
      minSubArr = min(arr[index1:index2+1])

      for i in range(0, index1):
        if (arr[i] > minSubArr):
          index1 = i
          break

      for i in range(len(arr)-1, index2, -1):
        if (arr[i] < maxSubArr):
          index2 = i
          break

      return index2 - index1 + 1
