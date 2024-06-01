# Timing functionality from Python's built-in module
from time import perf_counter
from functools import lru_cache


def timer(fn):
    def inner(*args):
        start = perf_counter()
        result = fn(*args)
        end = perf_counter()
        elapsed = end - start
        print(result)
        print('elapsed', elapsed)

    return inner


@timer
def calc_factorial(num):
    if num < 0:
        raise ValueError('Please use a number not smaller than 0')
    product = 1
    for i in range(num):
        product = product * (i+1)
    return product


# @timer
# @lru_cache()
# def fib(n):
#     if n < 2:
#         return n
#     return fib(n-1) + fib(n-2)

if __name__ == '__main__':
    calc_factorial(88)

    # fib(25)


