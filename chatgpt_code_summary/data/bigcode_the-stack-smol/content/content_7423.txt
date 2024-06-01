# 033
# Ask the user to enter two numbers. Use whole number division to divide
# the first number by the second and also work out the remainder and
# display the answer in a user-friendly way
# (e.g. if they enter 7 and 2 display “7 divided by 2 is 3
# with 1 remaining”).

from typing import List


def check_num_list(prompt: str, max_length: int = 0,
                   min_length: int = 0) -> List[float]:
    """Function to check if users input is a number, splitting number
    by spaces and checking that the correct amount of numbers are
    entered, returning them in a list"""

    while True:
        try:
            num = input(prompt)
            num = num.split(' ')
            if min_length:
                assert len(num) >= min_length, f'Please enter at least' \
                                               f' {min_length} numbers'
            if max_length:
                assert len(num) <= max_length, f'Please enter no more ' \
                                               f'than {max_length} ' \
                                               f'numbers'
            for index, value in enumerate(num):
                num[index] = float(value)

            return num

        except Exception as e:
            print(e)


def integer_devision_remainder(nums: List[float]) -> str:
    return f'{nums[0]} / {nums[1]} = {nums[0] // nums[1]} with' \
           f' a remainder of {nums[0] % nums[1]}'


if __name__ == '__main__':
    print('This program will tell you floor division and remainder '
          'between two numbers')

    request = 'Enter two numbers:'

    print(integer_devision_remainder(check_num_list(request, 2, 2)))
