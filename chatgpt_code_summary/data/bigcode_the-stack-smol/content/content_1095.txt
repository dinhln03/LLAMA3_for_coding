class Solution:
    def removeDuplicates(self, S: str) -> str:
        i = 1
        while i < len(S):
            if i <= 0:
                i += 1
                continue
            if S[i] == S[i - 1]:
                S = S[:i - 1] + S[i + 1:]
                i = i - 1
            else:
                i = i + 1
        return S


slu = Solution()
print(slu.removeDuplicates("abbaca"))
