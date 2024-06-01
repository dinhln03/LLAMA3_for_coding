from collections import defaultdict

dd = defaultdict(list)

n, m = map(int, input().split())

groupA = []
for i in range(n):
    a_element = input()
    dd[a_element].append(str(i+1))

for _ in range(m):
    b_element = input()
    if b_element not in dd:
        print(-1)
    else:
        print(" ".join(dd[b_element]))
