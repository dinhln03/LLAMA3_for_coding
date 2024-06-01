class Solution:
    def maxNumber(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[int]
        """
    
        def prep(nums, k):
            dr = len(nums) - k  # 要删除的数目
            stay = []  # 保留的list
            for num in nums:
                # 删除的空间 dr
                # 删除的必要 stay[-1] < num 即 堆栈法：上升就替换，下降就保留。 
                while dr and stay and stay[-1] < num:
                    stay.pop()
                    dr -= 1
                stay.append(num)
            return stay[:k]

        def merge(x, y):
            return [max(x, y).pop(0) for _ in x + y]

        l1 = len(nums1)
        l2 = len(nums2)
        #dr = l1 + l2 -k

        r = [0]
        for i in range(k + 1):
            # 遍历所有可能并比较大小
            if i <= l1 and  k-i <= l2:
                r = max(merge(prep(nums1, i), prep(nums2, k - i)), r)
        return r


if __name__ == "__main__":
    n1 = [3, 4, 6, 5]
    n2 = [9, 1, 2, 5, 8, 3]
    k = 5
    so = Solution()
    res = so.maxNumber(n1, n2, k)
    print(res)
