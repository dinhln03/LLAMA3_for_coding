class Solution:
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
#         max value taken as amount+1 because in worst case, it can be amount - when denoms of only 1
        res = [amount+1]*(amount+1)
        res[0] = 0
        for i in range(1, amount+1):
            for j in coins:
                if j <= i:
                    res[i] = min(res[i], res[i-j] + 1)

        if res[amount] > amount:
            return -1
        else:
            return res[amount]
        
