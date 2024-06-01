n = int(input())

arr = [[None for i in range(2*n+1)]for i in range(2*n+1)]

m = (2*n + 1) // 2
for i in range(n):
    arr[i][m] = i

arr[n][m] = n

for i in range(n+1,2*n+1):
    arr[i][m] = arr[i-1][m]-1

for y in range(1,m+1):
    for x in range(len(arr[0])):
        if x < m:
            arr[y][x] = arr[y-1][x+1]
        if x > m:
            arr[y][x] = arr[y-1][x-1]

for y in range(2*n-1,m,-1):
    for x in range(len(arr[0])):
        if x < m:
            arr[y][x] = arr[y+1][x+1]
        if x > m:
            arr[y][x] = arr[y+1][x-1]

for y in range(len(arr)):
    for x in range(len(arr[0])):
        if arr[y][x] is None:
            arr[y][x] = ' '
        else:
            arr[y][x] = str(arr[y][x])

out = [" ".join(xs).rstrip() for xs in arr]
print("\n".join(out))