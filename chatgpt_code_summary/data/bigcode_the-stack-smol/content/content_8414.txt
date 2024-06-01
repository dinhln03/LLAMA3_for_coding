a, b, c = map(int, input().split())

if(a > c and a > b):
    if(b > c):
        print(b)
    else:
        print(c)
elif(b > a and b > c):
    if(a > c):
        print(a)
    else:
        print(c)
else:
    if(a > b):
        print(a)
    else:
        print(b)