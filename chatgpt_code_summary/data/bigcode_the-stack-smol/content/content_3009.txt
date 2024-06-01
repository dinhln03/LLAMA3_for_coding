#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the isBalanced function below.
def isBalanced(s):


    left_symbol = [ '{', '[', '(']
    right_symbol = [ '}', ']', ')']

    # fast checking of symbol counting equality
    for i in range(3):
        left_count = s.count( left_symbol[i] )
        right_count = s.count( right_symbol[i] )

        if left_count != right_count:
            return "NO"
    


    _stack = []



    for i in range( len(s) ):



        char = s[i]
        if char in { '{', '[', '(' } :
            # push into stack
            _stack.append( char )


        if char in { '}', ']', ')' } :
            # pop from stack and compare with left symbol

            index_of_right = right_symbol.index( char )

            index_of_left = left_symbol.index( _stack.pop(-1) )

            if index_of_left == index_of_right:
                # match of {}, [], or ()
                pass
            else:
                return "NO"


    if len(_stack) == 0:
        return "YES"
    else:
        return "NO"



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        s = input()

        result = isBalanced(s)

        fptr.write(result + '\n')

    fptr.close()
