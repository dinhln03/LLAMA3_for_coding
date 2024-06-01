# Uses python3
import sys

def get_change(money, coins):
    t = [j+1 for j in range(money+1)]
    
    # boundary condition
    t[0] = 0
    for j in range(1, money+1):
        for c in coins:
            if c <= j:
                t[j] = min(t[j], 1+t[j-c])

    return t[money]

if __name__ == '__main__':
    coins = [1, 3, 4]
    money = int(input())

    print(get_change(money, coins))
