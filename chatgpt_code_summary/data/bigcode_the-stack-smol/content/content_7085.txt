class Solution:
  def numSpecialEquivGroups(self, A: List[str]) -> int:
    S = set()
    for s in A:
      S.add(''.join(sorted(s[::2]) + sorted(s[1::2])))
    return len(S)
