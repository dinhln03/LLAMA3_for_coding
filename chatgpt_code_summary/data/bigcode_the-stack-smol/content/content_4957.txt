# Athena Health Preliminary Test - II

#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'moves' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY arr as parameter.
#

def moves(arr):
    ee=e=0
    for i in range(0,len(arr)):
        if(arr[i]%2==0): e+=1
    for i in range(0,len(arr)):
        if(arr[i]%2==0):
            if(i>=e): ee+=1
    return ee
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr_count = int(input().strip())

    arr = []

    for _ in range(arr_count):
        arr_item = int(input().strip())
        arr.append(arr_item)

    result = moves(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
