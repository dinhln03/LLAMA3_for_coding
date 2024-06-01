#!/usr/bin/env python3

# Enter your code here. Read input from STDIN. Print output to STDOUT


def string_manipulate(string):
    even_string=''
    odd_string=''
    for idx, val in enumerate(string):
        if idx % 2 == 0:
            even_string+=val
        else:
            odd_string+=val
    
    return even_string+" "+odd_string
            
    
if __name__ == '__main__':
    T = int(input().strip())
    for t in range(T):
        string = str(input().strip())    
        print(string_manipulate(string))
    

