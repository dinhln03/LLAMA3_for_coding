# Time:  O(n * n!/(c_a!*...*c_z!), n is the length of A, B,
#                                  c_a...c_z is the count of each alphabet,
#                                  n = sum(c_a...c_z)
# Space: O(n * n!/(c_a!*...*c_z!)

# 854
# Strings A and B are K-similar (for some non-negative integer K)
# if we can swap the positions of two letters
# in A exactly K times so that the resulting string equals B.
#
# Given two anagrams A and B, return the smallest K for which A and B are
# K-similar.
#
# Example 1:
#
# Input: A = "ab", B = "ba"
# Output: 1
# Example 2:
#
# Input: A = "abc", B = "bca"
# Output: 2
# Example 3:
#
# Input: A = "abac", B = "baca"
# Output: 2
# Example 4:
#
# Input: A = "aabc", B = "abca"
# Output: 2
# Note:
# - 1 <= A.length == B.length <= 20
# - A and B contain only lowercase letters from
# the set {'a', 'b', 'c', 'd', 'e', 'f'}

# Solution Framework:
# The underlying graph of the problem is a graph with 6 nodes 'a', 'b', ..., 'f' and the edges A[i] -> B[i].
# Our goal is for this graph to have only self-edges (edges of the form a -> a.)

# If A = 'ca...' and B = 'ab...', then the first two edges of the underlying graph are c -> a and a -> b;
# and a swap between A[1] and A[0] changes these two edges to the single edge c -> b. Let's call this type
# of operation 'cutting corners'. Intuitively, our optimal swap schedule always increases the # of matches
# (A[i] == B[i]s) for each swap, so cutting corners is the only type of operation we need to consider.
# (This is essentially the happy swap assumption, proved in 765 - Couples Holding Hands)
#
# Now consider 'cycle decomposition' of the underlying graph. [This decomposition (or the # of cycles),
# is not necessarily unique.] Through operations of cutting corners, we'll delete all the (non-self) edges.
# Each cycle of length k requires k-1 operations to delete. Thus, the answer is just the minimum possible
# value of sum(C_k - 1), where C_1,... C_k are the lengths of the cycles in some cycle decomposition of
# the underlying graph. This can be re-written as (# of non-self edges) - (# of cycles).
# Hence, we want to maximize the # of cycles in a cycle decomposition of the underlying graph.

import collections

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class Solution(object):
    def kSimilarity(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: int
        """
        # Perform a regular breadth-first search: the neighbors to each node string S are all the strings
        # reachable with 1 swap to get the first unmatched character in S matched.
        # we can prove that an optimal solution swaps the left-most unmatched character A[i] with an
        # appropriate match A[j] which equals to B[i] (j > i), as this increases # of self-edges.

        # Time complexity: This reduces the # of "neighbors" of a node (string state) from O(N^2) to O(N):
        # O(N^2): swap any pair of chars in the string,
        # O(N): only swap the first unmatched char.
        def neighbors(s):
            for i, c in enumerate(s):
                if c != B[i]:
                    break
            t = list(s)
            for j in xrange(i+1, len(s)):
                if t[j] == B[i]:
                    t[i], t[j] = t[j], t[i]
                    yield "".join(t)
                    t[j], t[i] = t[i], t[j]

        q = collections.deque([A])
        steps = {A:0} # we need a set to remove repeatedness anyway, so put 'steps' together
        while q:
            s = q.popleft()
            if s == B:
                return steps[s]
            for t in neighbors(s):
                if t not in steps:
                    steps[t] = steps[s] + 1
                    q.append(t)

print(Solution().kSimilarity('abac', 'baca'))