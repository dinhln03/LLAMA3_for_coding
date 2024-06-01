from typing import List

'''
1. subproblems: dp(amount) the minimum number of coins needed to make changes for amount of S using the given coin denomination
2. guessing: all the available denomination c_i
3. relate subproblems: dp(amount) = min(dp(amount - c_i) + 1) for all possible c_i

Time complexity: O(#subproblems * #coins)
'''
class Solution:

    # top down solution
    def coinChange(self, coins: List[int], amount: int) -> int:

        # for amount less than 1, return 0
        if amount < 1:
            return 0
        
        memo = {}
        def helper(coins, amount):
            
            # for subproblems that we have alreay solve and memorized
            if amount in memo:
                return memo[amount]
            
            # base case, we reach out the bottom of the tree.
            if amount == 0:
                return 0

            # go through all possible coin denomination(breaches in tree)
            dp = float('inf')
            for coin in coins:

                if coin > amount:
                    continue
                
                # relate subproblems
                dp = min(helper(coins, amount - coin) + 1, dp)
                
            memo[amount] = dp
            return dp

        helper(coins, amount)
        return -1 if memo[amount] == float('inf') else memo[amount]


    # bottom-up solution, DAG
    def coinChange_2(self, coins: List[int], amount: int) -> int:

        memo = [float('inf') for i in range(amount + 1)]

        # dp[i] = min{dp[i - c_i] + 1} for all c_i
        memo[0] = 0
        for i in range(amount + 1):
            
            # check all the states that are reachable by coins to state i
            for coin in coins:
                if i < coin: 
                    continue

                memo[i] = min(memo[i], memo[i - coin] + 1)
        
        print(memo)
        return -1 if memo[amount] == float('inf') else memo[amount]
        

    



x = Solution()
# rs = x.coinChange([1, 2, 5], 2)
print(x.coinChange_2([1,2,5], 11))