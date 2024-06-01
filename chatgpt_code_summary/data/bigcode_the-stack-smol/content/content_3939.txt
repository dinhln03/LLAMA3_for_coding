N = int(input())

line = []
for a in range(N):
    line.append(int(input()))

total = 0
curIter = 1

while min(line) < 999999:
    valleys = []
    for a in range(N):
        if line[a] < 999999:
            if (a == 0 or line[a] <= line[a - 1]) and (a == N - 1 or line[a] <= line[a + 1]):
                valleys.append(a)
    for a in valleys:
        line[a] = 999999
    total += (curIter * len(valleys))
    curIter += 1

print(total)