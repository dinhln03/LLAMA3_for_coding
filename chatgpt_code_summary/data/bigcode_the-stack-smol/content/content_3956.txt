#coding:utf-8

'''
    filename:get_numbers.py
        chap:8
    subject:2
    conditions:file [data],contains: numbers,annotations,empty line
    solution:function get_numbers
'''

import sys

def get_numbers(file):
    f = None
    numbers = []
    try:
        with  open(file,'rt') as f:
            for line in f:
                try:
                    numbers.append(int(line))
                except ValueError as e:
                    print('PASS:this line is not pure number:',e)
    except OSError as e:
        print('Opening file error:',e)
    except BaseException as e:
        print('Something is wrong :',e)
    return numbers


if __name__ == '__main__':
    numbers = get_numbers(sys.argv[1])
    print(numbers)


