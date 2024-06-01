def solve(n):
    a = []
    for _ in range(n):
        name, h = input().split()
        h = float(h)
        a.append((name, h))
    a.sort(key = lambda t: t[1], reverse=True)
    m = a[0][1]
    for n, h in a:
        if h != m: break
        print(n, end = " ")
    print()


while True:
    n = int(input())
    if n == 0: break
    solve(n)
    