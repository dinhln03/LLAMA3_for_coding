# ABC082b
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10**6)

s = input()[:-1]
t = input()[:-1]
print('Yes' if sorted(s) < sorted(t, reverse=True) else 'No')
