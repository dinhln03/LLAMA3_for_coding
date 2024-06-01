"""
Write a function that takes in an array of integers and returns a sorted version of that array. Use the QuickSort algorithm to sort the array.

"""

def quick_sort(array):
    if len(array) <= 1:
        return array

    _rec_helper(array, 0, len(array) - 1)
    return array

def _rec_helper(array, start, end):
    # base case
    if start >= end:
        return

    pivot = start
    left = pivot + 1
    right = end

    while left <= right:
        if array[left] > array[pivot] and array[right] < array[pivot]:
            _swap(array, left, right)
        if array[pivot] >= array[left]:
            left += 1
        if array[pivot] <= array[right]:
            right -= 1
    _swap(array, pivot, right)
    if right - start > end - right:
        _rec_helper(array, start, right - 1)
        _rec_helper(array, right + 1, end)
    else:
        _rec_helper(array, right + 1, end)
        _rec_helper(array, start, right - 1)

def _swap(array, left, right):
    array[left], array[right] = array[right], array[left]

#test 
array = [3, 4, 7, 1, 1, 2, 5, 1, 3, 8, 4]
assert quick_sort(array) == sorted(array)
print('OK')
