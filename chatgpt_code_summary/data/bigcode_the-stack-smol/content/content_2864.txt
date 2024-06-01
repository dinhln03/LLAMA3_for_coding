class NumArray:
    # O(n) time | O(n) space - where n is the length of the input list
    def __init__(self, nums: List[int]):
        self.nums = []
        currentSum = 0
        for num in nums:
            currentSum += num
            self.nums.append(currentSum)
    # O(1) time to look up the nums list
    def sumRange(self, left: int, right: int) -> int:
        if left > 0:
            return self.nums[right] - self.nums[left - 1]
        else:
            return self.nums[right]