steps = 0

c = {}
m = 1

def collatz(n):
    global steps

    if n in c:
        steps += c[n]
        return

    if n == 1:
        return

    steps += 1

    if n % 2 == 0:
        collatz(n/2)
        return
        
    collatz(3 * n + 1)

def main(max):
    global steps
    global m

    for i in range(1, max):
        collatz(i)
        c[i] = steps

        if steps > c[m]:
            m = i

        steps = 0

main(1000000)
print(m)
