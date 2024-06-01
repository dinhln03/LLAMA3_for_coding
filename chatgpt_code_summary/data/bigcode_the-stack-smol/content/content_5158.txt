'''
Created on Sep 18, 2017

@author: jschm
'''
from cs115 import map
def powerset(lst):
    """returns the power set of the list - the set of all subsets of the list"""
    if lst == []:
        return [[]]
    #power set is a list of lists
    #this way is more efficent for getting the combinations of the characters in a list
    lose_it = powerset(lst[1:])
    use_it = map(lambda subset: [lst[0]] + subset, lose_it)
    return lose_it + use_it

print(powerset(['a', 'b', 'c']))

def subset(target, lst):
    """determines whether or not it is possible to create target sum using the
    values in the list. Values in teh list can be positive, negative, or zero."""
    if target == 0:
        return True
    #what if target is 0?
    if lst == []:
        return False
    #use_it = subset(target - lst[0], lst[1:])
    #lose_it = subset(target, lst[1:])
    """and and or are short-cut operators in python. THe second operand is not evaluated
    when the overall result can be deduced by evaluating the second operand"""
    #return use_it or lose_it
    return subset(target - lst[0], lst[1:]) or subset(target, lst[1:])

print(subset(5,[1,3,2,4,5]))

def subset_with_values(target, lst):
    """Determines whether or not it is possible to create the target sum using
    values in the list. Values in the list can be positive, negative, or zero.
    The function returns a tuple of exactly two items. The first is a boolean,
    that indicates true if the sum is possible and false if it is not. The second
    element in the tuple is a list of all values that add up to make the target sum."""
    if target == 0:
        return(True, [])
    if lst == []:
        return(False, [])
    use_it = subset_with_values(target - lst[0], lst[1:])
    if use_it[0]:
        return(True, [lst[0]] + use_it[1])
    return subset_with_values(target, lst[1:])

print(subset_with_values(8, [7,2,2,2,2]))
print(subset_with_values(12, [1,2,4,9]))

"""
def LCSWithValues2(S1,S2):
    if S1 == "" or S2 == "":
        return (0, "")
    if S1[0] == S2[0]:
        result = result + S1[0]
        return (1 + LCSWithValues2(S1[1:], S2[1:]), result)
    useS1 = LCSWithValues2(S1, S2[1:])
    useS2 = LCSWithValues2(S1[1:], S2)
    if useS1[0] > useS2[0]:
        return useS1
    return useS2

print(LCSWithValues2("sam", "spam"))

"""

def LCSWithValues(S1,S2):
    """returns the longest common string"""
    if S1 == "" or S2 == "":
        return (0, "")
    if S1[0] == S2[0]:
        result = LCSWithValues(S1[1:], S2[1:])
        return (1 + result[0], S1[0] + result[1])
    useS1 = LCSWithValues(S1, S2[1:])
    useS2 = LCSWithValues(S1[1:], S2)
    if useS1[0] > useS2[0]:
        return useS1
    return useS2

print(LCSWithValues("sam", "spam"))

#^^^the LCSWithValues2 does not work because the result variable needs to be defined, and if it is redefined it stays empty always.

def coin_row(lst):
    #one line:
    return 0 if lst == [] else max(lst[0] + coin_row(lst[2:]), coin_row(lst[1:]))
    """
    if(lst == []):
        return 0
    return max(lst[0] + coin_row(lst[2:]), coin_row(lst[1:]))
    """
    """
    if(lst == []):
        return 0
    use_it = lst[0] + coin_row(lst[2:])
    lose_it = coin_row(lst[1:])
    return max(use_it, lose_it)
    This is how you set up each function^^^
    and then you can make it nicer
    """
    """
    if(coin_row(lst[1:])>lst[0]):
    amount = coin_row(lst[1:])
    return max(coin_row(lst[2:]), coin_row(lst[2:]))
    """
def coin_row_with_values(lst):
    if lst == []:
        return [0, []]
    use_it = coin_row_with_values(lst[2:])
    new_sum = lst[0] + use_it[0]
    #that's the result^
    lose_it = coin_row_with_values(lst[1:])
    if new_sum > lose_it[0]:
        #only returns this once I think
        #nevermind!
        #print('hello')
        return [new_sum, [lst[0]] + use_it[1]]
    return lose_it

print(coin_row([10, 5, 5, 5, 10, 10, 1, 1]))
print(coin_row_with_values([10, 5, 5, 5, 10, 50, 1, 10, 1, 1, 25]))

#can use below as spell-checker
def distance(first, second):
    if first == '':
        return len(second)
    if second == '':
        return len(first)
    if first[0] == second[0]:
        return distance(first[1:], second[1:])
    substitution = 1 + distance(first[1:], second[1:])
    deletion = 1 + distance(first[1:], second)
    insertion = 1 + distance(first, second[1:])
    return min(substitution, deletion, insertion)

