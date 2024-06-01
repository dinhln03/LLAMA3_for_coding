class Solution:
    def maxChunksToSorted(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        stacks = []
        for num in arr:
            if not stacks:
                stacks.append([num])
            elif num >= stacks[-1][0]:
                stacks.append([num])
            else:
                # num < stacks[-1][0]
                stacks[-1].append(num)
                while len(stacks) >= 2:
                    if num < stacks[-2][0]:
                        stacks[-2][0] = max(stacks[-2][0], stacks[-1][0])
                        stacks[-2].extend(stacks.pop())
                    else:
                        break
        # print(stacks)
        return len(stacks)


sol = Solution().maxChunksToSorted
print(sol([5, 4, 3, 2, 1]))
print(sol([2, 1, 3, 4, 4]))
print(sol([5, 3, 1, 2, 4]))
print(sol([1, 0, 1, 3, 2]))
print(sol([5, 1, 1, 8, 1, 6, 5, 9, 7, 8]))
