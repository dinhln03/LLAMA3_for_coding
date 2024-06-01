#!/usr/bin/env python3

def selection_sort(lst):

    length = len(lst)
    for i in range(length - 1):
        least = i
        for k in range(i + 1, length):
            if lst[k] < lst[least]:
                least = k
        lst[least], lst[i] = (lst[i], lst[least])
    return lst

print(selection_sort([5, 2, 4, 6, 1, 3]))