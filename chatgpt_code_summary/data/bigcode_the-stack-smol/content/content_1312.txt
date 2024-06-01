"""
724. Minimum Partition
https://www.lintcode.com/problem/minimum-partition/description

01背包
算法班2020 C27 01背包变形

第1种dp定义
dp[i][j]: considering previous i items to fill <=j, what the maximum value

dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - nums[i - 1]] + nums[i - 1])

dp[0][0] = 0
dp[i][0] = 0

answer
max(dp[n])

2d array

time limit exceeded
"""
class Solution:
    """
    @param nums: the given array
    @return: the minimum difference between their sums 
    """
    def findMin(self, nums):
        # write your code here
        if not nums:
            return 0

        n = len(nums)
        total_sum = sum(nums)
        target = total_sum // 2

        dp = [[0] * (target + 1) for _ in range(2)]

        now, old = 0, 0
        for i in range(1, n + 1):
            old = now
            now = 1 - now
            for j in range(0, target + 1):
                dp[now][j] = dp[old][j]
                if j >= nums[i - 1]:
                    dp[now][j] = max(dp[now][j], dp[old][j - nums[i - 1]] + nums[i - 1])

        return total_sum - 2 * max(dp[now])

s = Solution()
nums = [987,523,979,847,734,706,452,903,702,332,713,181,991,843,879,505,718,694,18,303,795,521,696,388,866,908,350,528,445,780,864,295,257,337,704,648,495,949,39,33,606,553,618,191,854,405,715,413,472,185,216,489,212,199,162,462,929,191,429,726,902,9,579,403,370,435,871,160,197,884,619,716,182,7,906,974,679,531,852,158,861,174,445,701,871,557,942,798,921,389,450,485,901,179,515,401,117,451,731,828,685,20,50,673,891,232,30,385,511,338,375,118,81,392,296,546,903,59,580,620,268,422,597,876,333,766,158,295,443,204,434,357,632,592,543,341,434,58,525,683,338,165,332,51,152,191,378,63,10,475,951,469,622,811,296,415,282,547,994,358,134,195,888,75,195,805,908,673,867,346,935,318,603,507,45,209,54,641,515,867,881,880,290,781,452,808,775,998,731,908,451,592,608,87,1000,812,30,673,393,380,241,135,421,144,954,64,747,502,633]
print(s.findMin(nums))