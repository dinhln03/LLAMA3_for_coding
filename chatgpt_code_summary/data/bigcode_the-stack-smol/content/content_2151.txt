class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        lookup = dict(((v, i) for i, v in enumerate(nums)))
        return next(( (i+1, lookup.get(target-v)+1) 
                for i, v in enumerate(nums) 
                    if lookup.get(target-v, i) != i), None)

a  = Solution()
print(a.twoSum([2, 11, 7, 15],9))
# 越简单的问题越要小心