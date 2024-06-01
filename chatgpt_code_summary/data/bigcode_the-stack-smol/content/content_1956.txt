import sympy
from sympy import *

def check_weak_prime(n):
    if not isprime(n):
        return(False)
    digits=[int(i) for i in str(n)]
    # For each digit location - test all other values to see if
    # the result is prime.  If so - then this is not a weak prime
    for position in range(len(digits)):
        digits2=[i for i in digits]
        for j in range(10):
            if j != digits[position]:
                digits2[position]=j
                m=0
                for i in digits2:
                    m=10*m+i
                if isprime(m):
                    return(False)
    return(True)

def search_palindromic_weak_prime(nlow,nhigh):
    n=nlow
    if not isprime(n):
        n=nextprime(n)
    while(n<nhigh):
        if check_weak_prime(n):
            print("Weak prime = ",n)
            n2=int(str(n)[::-1])
            if check_weak_prime(n2):
                print("Solution found:")
                print("   n = ",n)
                print("   n2 = ",n2)
                return True
        n=nextprime(n)
    return False
