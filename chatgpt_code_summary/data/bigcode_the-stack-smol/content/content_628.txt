from typing import List


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)

        if m == 0:
            return 0
        if m == 1:
            return sum(grid[0])

        n = len(grid[0])

        dp = [[0] * n for _ in range(m)]

        for x in range(n):
            for y in range(m):
                if x == y == 0:
                    dp[0][0] = grid[0][0]
                elif x == 0:
                    dp[y][x] = dp[y - 1][x] + grid[y][x]
                elif y == 0:
                    dp[y][x] = dp[y][x - 1] + grid[y][x]
                else:
                    dp[y][x] = min(dp[y][x - 1], dp[y - 1][x]) + grid[y][x]

        return dp[-1][-1]


s = Solution()
print(s.minPathSum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]))
print(s.minPathSum([[1, 3, 1]]))
print(s.minPathSum([]))
print(s.minPathSum([[1, 2, 5], [3, 2, 1]]))
