class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        a = set()
        for i in nums:
            if i in a:
                a.remove(i)
            else:
                a.add(i)
                
        for i in a:
            return i