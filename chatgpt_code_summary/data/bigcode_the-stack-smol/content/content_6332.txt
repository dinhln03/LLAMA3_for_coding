option='y'
while option=='y':
    def fibo(n):
        a=0
        b=1
        for i in range(0,n):
            temp=a
            a=b
            b=temp+b
        return a

    print("Enter the limit of fibonacci series")
    num=int(input())

    for c in range(0,num):
        print (fibo(c))
    print("Do you want to continue?(y/n)")
    option=input()
print('Thank you for using this programme')
