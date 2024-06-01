import math
n=int(input())
c=list(map(int, input().split()))
print(sum([abs(i) for i in c]))
print(math.sqrt(sum([i*i for i in c])))
print(max([abs(i) for i in c]))