A, B, C, D = map(int, input().split())

s1 = set(range(A, B + 1))
s2 = set(range(C, D + 1))

print(len(s1) * len(s2) - len(s1.intersection(s2)))
