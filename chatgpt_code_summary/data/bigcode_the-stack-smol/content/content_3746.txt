"""
This project demonstrates NESTED LOOPS (i.e., loops within loops)
in the context of SEQUENCES OF SUB-SEQUENCES.

Authors: David Mutchler, Vibha Alangar, Matt Boutell, Dave Fisher,
         Mark Hays, Amanda Stouder, Aaron Wilkin, their colleagues,
         and Lucas D'Alesio.
"""  # DONE: 1. PUT YOUR NAME IN THE ABOVE LINE.


def main():
    """ Calls the other functions to test them. """
    #run_test_largest_number()
    #run_test_largest_negative_number()
    run_test_first_is_elsewhere_too()


def run_test_largest_number():
    """ Tests the    largest_number    function. """
    # -------------------------------------------------------------------------
    # DONE: 2. Implement this TEST function.
    #   It TESTS the  largest_number  function defined below.
    #   Include at least ** 1 ** ADDITIONAL test beyond those we wrote.
    # -------------------------------------------------------------------------
    print()
    print('-------------------------------------')
    print('Testing the   LARGEST_NUMBER   function:')
    print('-------------------------------------')

    # Test 1:
    expected = 13
    answer = largest_number([(3, 1, 4),
                             (13, 10, 11, 7, 10),
                             [1, 2, 3, 4]])
    print('Expected and actual are:', expected, answer)

    # Test 2:
    expected = -1111111111111111
    answer = largest_number(([], [-1111111111111111], []))
    print('Expected and actual are:', expected, answer)

    # Test 3:
    expected = None
    answer = largest_number(([], [], []))
    print('Expected and actual are:', expected, answer)

    # DONE 2 (continued): Add your ADDITIONAL test(s) here:
    # Test 3:
    expected = 13
    answer = largest_number([(3, 1, 4),
                             (13, 10, 11, 7, 10),
                             [1, 2, 3, 4]])
    print('Expected and actual are:', expected, answer)

def largest_number(seq_seq):
    """
    Returns the largest number in the subsequences of the given
    sequence of sequences.  Returns None if there are NO numbers
    in the subsequences.

    For example, if the given argument is:
        [(3, 1, 4),
         (13, 10, 11, 7, 10),
         [1, 2, 3, 4]]
    then this function returns 13.

    As another example, if the given argument is:
      ([], [-1111111111111111], [])
    then this function returns -1111111111111111.

    As yet another example, if the given argument is:
      ([], [], [])
    then this function returns None.

    Preconditions:
      :type seq_seq: (list, tuple)
    and the given argument is a sequence of sequences,
    where each subsequence contains only numbers.
    """
    # -------------------------------------------------------------------------
    # DONE: 3. Implement and test this function.
    #   Note that you should write its TEST function first (above).
    # -------------------------------------------------------------------------

    x = None
    for j in range (len(seq_seq)):
        for k in range(len(seq_seq[j])):
            x = j
            y = k

    for l in range(len(seq_seq)):
        for o in range(len(seq_seq[l])):
            if seq_seq[l][o] > seq_seq[x][y]:
                x = l
                y = o

    if x == None:
        return None

    return seq_seq[x][y]

def run_test_largest_negative_number():
    """ Tests the    largest_negative_number    function. """
    # -------------------------------------------------------------------------
    # DONE: 4. Implement this TEST function.
    #   It TESTS the  largest_negative_number  function defined below.
    #
    #   Include enough tests to give you confidence that your solution
    #   to this challenging problem is indeed correct.
    # -------------------------------------------------------------------------
    print()
    print('-------------------------------------------------')
    print('Testing the   LARGEST_NEGATIVE_NUMBER   function:')
    print('-------------------------------------------------')

    # Test 1:
    expected = 11
    answer = largest_number([(3, 1, 4),
                             (-13, 10, 11, 7, 10),
                             [1, 2, 3, 4]])
    print('Expected and actual are:', expected, answer)

    # Test 2:
    expected = -2
    answer = largest_number(([-10], [-1111111111111111], [-2]))
    print('Expected and actual are:', expected, answer)

    # Test 3:
    expected = None
    answer = largest_number(([], [], []))
    print('Expected and actual are:', expected, answer)

def largest_negative_number(seq_seq):
    """
    Returns the largest NEGATIVE number in the given sequence of
    sequences of numbers.  Returns None if there are no negative numbers
    in the sequence of sequences.

    For example, if the given argument is:
        [(30, -5, 8, -20),
         (100, -2.6, 88, -40, -5),
         (400, 500)
        ]
    then this function returns -2.6.

    As another example, if the given argument is:
      [(200, 2, 20), (500, 400)]
    then this function returns None.

    Preconditions:
      :type seq_seq: (list, tuple)
    and the given argument is a sequence of sequences,
    where each subsequence contains only numbers.
    """
    # -------------------------------------------------------------------------
    # DONE: 5. Implement and test this function.
    #   Note that you should write its TEST function first (above).
    #
    # CHALLENGE: Try to solve this problem with no additional sequences
    #   being constructed (so the SPACE allowed is limited to the
    #   give sequence of sequences plus any non-list variables you want).
    # -------------------------------------------------------------------------

    s = []
    for k in range(len(seq_seq)):
        s2 = seq_seq[k]
        if s2 != []:
            s = s + [max(s2)]

    return max(s)

def run_test_first_is_elsewhere_too():
    """ Tests the    first_is_elsewhere_too    function. """
    # -------------------------------------------------------------------------
    # We have supplied tests for you. No additional tests are required,
    # although you are welcome to supply more tests if you choose.
    # -------------------------------------------------------------------------
    print()
    print('-------------------------------------')
    print('Testing the   FIRST_IS_ELSEWHERE_TOO   function:')
    print('-------------------------------------')

    # FYI: The notation below constructs what is called a DICTIONARY.
    #      It is like a list, but the indices can be any immutable
    #      objects (here, True or False), not just 0, 1, 2, ... as in lists.
    message = {True: 'Your code PASSED this test.\n',
               False: 'Your code FAILED this test.\n'}
    no_failures = True

    # Test 1:
    expected = True
    answer = first_is_elsewhere_too([(3, 1, 4),
                                     (13, 10, 11, 7, 10),
                                     [11, 12, 3, 10]])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 2:
    expected = False
    answer = first_is_elsewhere_too([(3, 1, 4),
                                     (13, 10, 11, 7, 10),
                                     [11, 2, 13, 14]])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 3:
    expected = False
    answer = first_is_elsewhere_too([[], [1, 2], [1, 2]])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 4:
    expected = True
    answer = first_is_elsewhere_too([('a', 9),
                                     (13, 10, 11, 7, 'a'),
                                     [11, 12, 3, 10]])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])  # Test 1:
    no_failures = no_failures and (answer == expected)

    # Test 5:
    expected = False
    answer = first_is_elsewhere_too([('a', 9),
                                     (13, 10, 11, 7, 'aa'),
                                     [11, 12, 3, 10]])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 6:
    expected = False
    answer = first_is_elsewhere_too([('a', 'a', 'b', 'b', 'a', 'b')])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 7:
    expected = False
    answer = first_is_elsewhere_too([()])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 8:
    expected = True
    answer = first_is_elsewhere_too([('a'), (), (), (), ('a')])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 9:
    expected = True
    answer = first_is_elsewhere_too([('a'), (), (), (), ('a'), ()])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 10:
    expected = False
    answer = first_is_elsewhere_too([('a'), (), (), (), ('b'), ()])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 11:
    expected = True
    answer = first_is_elsewhere_too(['hello', 'goodbye'])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 12:
    expected = False
    answer = first_is_elsewhere_too(['hello', 'xxxxxxxxxxx'])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 13:
    expected = False
    answer = first_is_elsewhere_too(['1234567890',
                                     'one two three',
                                     'i am free',
                                     'four five six',
                                     'get my sticks',
                                     'seven eight nine',
                                     'i am fine'])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 14:
    expected = True
    answer = first_is_elsewhere_too([(1000 * 'a') + 'b' + (500 * 'a'),
                                     (800 * 'c') + 'd' + 1200 * 'c',
                                     'b'])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 15:
    expected = True
    answer = first_is_elsewhere_too([(1000 * 'a') + 'b' + (500 * 'a'),
                                     (800 * 'c') + 'd' + 1200 * 'c',
                                     (700 * 'eee') + 'b' + (90 * 'd'),
                                     (800 * 'c') + 'd' + 1200 * 'c'])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 16:
    expected = True
    answer = first_is_elsewhere_too([(1000 * 'b') + 'acd' + (500 * 'f'),
                                     (800 * '1') + '234a',
                                     'eeee'])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 17:
    expected = True
    answer = first_is_elsewhere_too([(1000 * 'b') + 'acd' + (500 * 'f'),
                                     'a' + (800 * '1') + '234',
                                     '123'])
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 18:
    test1 = [(1000 * 'b') + 'acd' + (500 * 'f'),
             (800 * '1') + '234',
             '123']
    for k in range(95):
        test1.append(k * chr(k))
    test2 = []
    for k in range(30):
        test2.append(k * chr(k))

    expected = True
    answer = first_is_elsewhere_too(test1 + ['a'] + test2)
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 19 (continues test 18):
    expected = False
    answer = first_is_elsewhere_too(test1 + test2)
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    # Test 20 (continues test 18):
    expected = True
    a_inside = (100 * 'b') + 'a' + (100 * 'b')
    answer = first_is_elsewhere_too(test1 + [a_inside] + test2)
    print('Expected and actual are:', expected, answer)
    print(message[answer == expected])
    no_failures = no_failures and (answer == expected)

    if no_failures:
        print('*** Your code PASSED all')
    else:
        print('!!! Your code FAILED some')
    print('    of the tests for first_is_elsewhere_too')


def first_is_elsewhere_too(seq_seq):
    """
    Given a sequence of subsequences:
      -- Returns True if any element of the first (initial) subsequence
           appears in any of the other subsequences.
      -- Returns False otherwise.

    For example, if the given argument is:
        [(3, 1, 4),
         (13, 10, 11, 7, 10),
         [11, 12, 3, 10]]
    then this function returns True because 3 appears
    in the first subsequence and also in the third subsequence.

    As another example, if the given argument is:
        [(3, 1, 4),
         (13, 10, 11, 7, 10),
         [11, 2, 13, 14]]
    then this function returns False because 3 does not appear in
    any subsequence except the first, 1 does not appear in any
    subsequence except the first, and 4 does not appear in any
    subsequence except the first.

    As yet another example, if the given argument is:
      ([], [1, 2], [1, 2])
    then this function returns False since no element of the first
    subsequence appears elsewhere.

    Preconditions:
      :type seq_seq: (list, tuple)
    and the given argument is a sequence of sequences.
    """
    # -------------------------------------------------------------------------
    # DONE: 6. Implement and test this function.
    #          Some tests are already written for you (above).
    #
    # IMPLEMENTATION RESTRICTION:
    #   ** You may NOT use anything but comparison (==) in judging
    #      membership.  In particular, you may NOT use:
    #        -- the IN operator
    #              (example:  7 in [9, 6, 7, 9] returns True)
    #        -- the COUNT method
    #              (example:  [9, 6, 7, 9].count(9) returns 2)
    #        -- the INDEX method
    #              (example:  [9, 6, 7, 9, 6, 1].index(6) returns 1)
    #   in this problem, as doing so would defeat the goal of providing
    #   practice at loops within loops (within loops within ...)
    # -------------------------------------------------------------------------

    for j in range(len(seq_seq[0])):
        for k in range(1, len(seq_seq)):
            for i in range(len(seq_seq[k])):
                if seq_seq[k][i] == seq_seq[0][j]:
                    return True
    return False


# -----------------------------------------------------------------------------
# Calls  main  to start the ball rolling.
# -----------------------------------------------------------------------------
main()
