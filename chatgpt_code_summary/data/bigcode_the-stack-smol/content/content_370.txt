x = int(input())
m = int(input())
if x < 10:
    if x <= m:
        print(1)
    else:
        print(0)
else:
    xarr = []
    while x:
        xarr = [x % 10] + xarr
        x //= 10
    n = len(xarr)
    l = max(xarr) + 1
    def check(base, xarr):
        ans = xarr[0] * (base ** (n - 1))
        if ans > m:
                return False
        return True

    def check1(base, xarr):
        ans = 0
        for i in range(n):
            ans += xarr[i] * base ** (n - 1 - i)
            if ans > m:
                return False
        return True

    r = 1
    while check(2 * r, xarr):
        r *= 2
    r *= 2
    ll, rr = l, r
    while ll < rr:
        mid = ll + (rr - ll) // 2
        if check1(mid, xarr):
            ll = mid + 1
        else:
            rr = mid

    if ll - 1 < l:
        print(0)
    else:
        print(ll - l)
