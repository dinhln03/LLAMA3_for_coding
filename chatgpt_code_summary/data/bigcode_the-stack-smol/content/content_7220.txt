#This code aims to return an "n" result of the Fibonnachi Sequence.
#Below are two fucntions, each of which return the same results by following different algorithms.

def getFibNExt (n):
    fibAr = [0, 1, 0]
    for i in range(n-1):
        fibAr[2] = fibAr[0]
        fibAr[0] += fibAr[1]
        fibAr[1] = fibAr[2]
    return fibAr[0]

def getFibNSimp (n):
    if ( n == 2):
        return 1
    elif ( n <= 1):
        return 0
    else:
        #Since the fibonnachi numbers are a recursive sum of all the numbers of the set prior to them we can rely on recursion to get the value of the set.
        return getFibNSimp(n-1) + getFibNSimp(n-2)

def outputFibN():
    validSelection = False
    fibSelect = int(input("Enter a position in the Fibonnachi series to extract: "))
    while not validSelection:
        select = input("For the simple fibonnachi algorithm enter (s), for the extended algorithm press (e), if you want a list until... ")
        if (select == "e"):
            print("The",fibSelect,"° Fibonnachi number is: ",getFibNExt(fibSelect))
            validSelection = True
        elif (select == "s"):
            print("The", fibSelect, "° Fibonnachi number is: ", getFibNSimp(fibSelect))
            validSelection = True
        else:
            validSelection = False
            print("Invalid selection, please press (e) for the extended algorithm or (s) for the simple algorithm.")

outputFibN()