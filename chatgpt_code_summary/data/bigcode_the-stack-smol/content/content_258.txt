#!/usr/bin/python
# https://practice.geeksforgeeks.org/problems/knapsack-with-duplicate-items/0

def sol(n, w, wt, v):
    """
    We do not need to create a 2d array here because all numbers are available
    always
    Try all items for weight ranging from 1 to w and check if weight
    can be picked. Take the max of the result
    """
    dp = [0 for i in range(w+1)]
    
    for i in range(n):
        for j in range(w+1):
            if wt[i] <= j:
                dp[j] = max(dp[j], v[i]+dp[j-wt[i]])
    return dp[w]