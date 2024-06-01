import itertools
m,n = input().split()
n = int(n)
l = [ ''.join(list(i)) for i in list(itertools.permutations(list(m), n))]
l.sort()
for i in l:
    print(i)