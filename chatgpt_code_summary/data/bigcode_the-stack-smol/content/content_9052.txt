arr = []
dict = {}

for i in range(10):
    arr.append(int(input())%42)

for i in arr:
    dict[i] = 0

print(len(dict))