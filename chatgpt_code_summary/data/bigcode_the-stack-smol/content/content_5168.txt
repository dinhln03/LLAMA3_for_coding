from typing import List


def bubblesort(nums: List[int]):
    """ sort list """
    for i in range(0, len(nums)):
        for j in range(0, len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                tmp = nums[j]
                nums[j] = nums[j + 1]
                nums[j + 1] = tmp
    return nums
