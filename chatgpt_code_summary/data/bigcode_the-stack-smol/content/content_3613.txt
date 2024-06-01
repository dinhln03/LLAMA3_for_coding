n=(input("Enter a number"))
a=len(n)
s=int(n)
sum=0
p=s
while s>0:
    b=s%10
    sum=sum+b**a
    s=s//10
if sum==p:
    print("It is an Amstrong Number")
else:
    print("It is Not an Amstrong Number")