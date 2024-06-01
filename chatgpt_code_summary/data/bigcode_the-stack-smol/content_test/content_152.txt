n = int(input())
k = int(input())

total = n
for i in range(k):
  total += int(str(n) + ('0' * (i+1)))
print(total)