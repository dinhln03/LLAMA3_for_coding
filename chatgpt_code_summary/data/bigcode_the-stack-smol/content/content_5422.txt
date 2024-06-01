n = int(input())
x = int(input())

# n = 5 : 101 => x ** 4 * x ** 1

ans = 1
while n > 0:
    if n & 1:
        ans *= x
        n >>= 1
        x *= x
        continue
    n >>= 1
    x *= x

print(ans)
