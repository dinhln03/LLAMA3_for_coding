def get_min_max(ints):
    """
    Return a tuple(min, max) out of list of unsorted integers.

    Args:
       ints(list): list of integers containing one or more integers
    """

    # Handle non-list input
    if not isinstance(ints, list):
        return None, None

    # Define variables for min and max value and initialize to None
    min_value = None
    max_value = None

    for index, value in enumerate(ints):

        if index == 0:
            min_value = value
            max_value = value

        if value < min_value:
            min_value = value
        elif value > max_value:
            max_value = value

    return min_value, max_value


# Example Test Case of Ten Integers
import random

# Test case 1: random int array
l = [i for i in range(0, 10)]  # a list containing 0 - 9
print(f"Test case 1 - random list of int: {l}")
random.shuffle(l)

# Should print "Pass" as the result should be (0, 9)
print ("Pass" if ((0, 9) == get_min_max(l)) else "Fail")

# Test case 2: empty array
print(f"Test case 2 - empty array")

# Should print "Pass" as the result should be (None, None)
print ("Pass" if ((None, None) == get_min_max([])) else "Fail")

# Test case 3: array with single item
print(f"Test case 3 - array with single item")

# Should print "Pass" as the result should be (None, None)
print ("Pass" if ((1, 1) == get_min_max([1])) else "Fail")

# Test case 4: non array input
print(f"Test case 4 - non array input")

# Should print "Pass" as the result should be (None, None)
print ("Pass" if ((None, None) == get_min_max(10)) else "Fail")
