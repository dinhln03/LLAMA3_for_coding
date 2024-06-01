import math
n=int(input("team:"))
S=math.factorial(n)//math.factorial(n-3)
D=math.factorial(n)
print("top places:" +str(S))
print("all places:" +str(D))
