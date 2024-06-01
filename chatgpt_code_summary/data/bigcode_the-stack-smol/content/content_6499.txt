T = int(input())

if 1 <= T <= 100:
    for i in range(T):
        A = int(input())
        B = int(input())
        C = int(input())
        if (1 <= A <= 10 ** 16) and (1 <= B <= 10 ** 16) and (1 <= C <= 10 ** 16):
            for x in range(A, B + 1):
                if x % C == 0:
                    print(x)
