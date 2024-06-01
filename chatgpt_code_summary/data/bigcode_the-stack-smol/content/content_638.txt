class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        '''
        #最长连续公共子串
        l1=len(text1)
        l2=len(text2)

        if l1==0 or l2==0:
            return 0
        dp = [[0 for i in range(l2)] for i in range(l1)]
        res = 0
        if text1[0]==text2[0]:
            dp[0][0]=1
            res=1
        for i in range(1,l2):
            if text2[i]==text1[0]:
                dp[0][i]=1
                res=1
        for i in range(1,l1):
            if text1[i]==text2[0]:
                dp[i][0]=1
                res=1


        for i in range(1,l1):
            for j in range(1,l2):
                if text1[i]==text2[j]:
                    dp[i][j]=dp[i-1][j-1]+1
                    res=max(res,dp[i][j])

        return res
        '''

        '''
        #最长子串（可不连续）：其实就是在问text1[:i+1]和text2[:j+1]有多少个相同的字母
        l1 = len(text1)
        l2 = len(text2)

        if l1 == 0 or l2 == 0:
            return 0
        dp = [[0 for i in range(l2)] for i in range(l1)]
        if text1[0] == text2[0]:
            dp[0][0] = 1
        for i in range(1, l2):
            if text2[i] == text1[0] or dp[0][0]==1 or dp[0][i-1]==1:
                dp[0][i] = 1
        for i in range(1, l1):
            if text1[i] == text2[0] or dp[0][0]==1 or dp[i-1][0]==1:
                dp[i][0] = 1

        for i in range(1, l1):
            for j in range(1, l2):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j]=max(dp[i][j-1],dp[i-1][j])

        return dp[-1][-1]
        '''

        #recursion

        #exit case
        if len(text1)==0 or len(text2)==0:
            return 0

        if text1[-1]==text2[-1]:
            return 1+self.longestCommonSubsequence(text1[:-1],text2[:-1])
        else:
            return max(self.longestCommonSubsequence(text1[:-1],text2),self.longestCommonSubsequence(text1,text2[:-1]))


if __name__ == '__main__':
    sol=Solution()
    text1 ="ylqpejqbalahwr"
    text2 ="yrkzavgdmdgtqpg"
     # "hofubmnylkra"
     # "pqhgxgdofcvmr"
    print(sol.longestCommonSubsequence(text1,text2))
