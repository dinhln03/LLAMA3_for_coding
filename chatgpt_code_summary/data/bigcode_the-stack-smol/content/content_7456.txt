def valid(ps, a, n):
    ps = set(ps)
    for i in xrange(1, n):
        for j in xrange(1, n):
            for k in xrange(i):
                for l in xrange(j):
                    if a[i][j] != -1 or i*n+j in ps: continue
                    if a[k][l] != -1 or k*n+l in ps: continue
                    if a[i][l] != -1 or i*n+l in ps: continue
                    if a[k][j] != -1 or k*n+j in ps: continue
                    return False
    return True

def dfs(idx, cur, n, a, b, r, c, ans):
    if idx == n*n:
        if not valid(cur, a, n): return
        t = 0
        for x in cur:
            i, j = divmod(x, n)
            if a[i][j] != -1: continue
            t += b[i][j]
        ans[0] = min(ans[0], t)
        return
    dfs(idx+1, cur, n, a, b, r, c, ans)
    cur += idx,
    dfs(idx+1, cur, n, a, b, r, c, ans)
    cur.pop()
    
def solve(cid):
    n = int(raw_input())
    a, b = [], []
    for _ in xrange(n):
        a += map(int, raw_input().split()),
    for _ in xrange(n):
        b += map(int, raw_input().split()),
    r = map(int, raw_input().split())
    c = map(int, raw_input().split())
    ans = [float('inf')]
    dfs(0, [], n, a, b, r, c, ans)
    print 'Case #{}: {}'.format(cid, ans[0])

for cid in xrange(1, int(raw_input())+1):
    solve(cid)
