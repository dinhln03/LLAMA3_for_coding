'''
Given a string, write a function that uses recursion to output a
list of all the possible permutations of that string.

For example, given s='abc' the function should return ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

Note: If a character is repeated, treat each occurence as distinct,
for example an input of 'xxx' would return a list with 6 "versions" of 'xxx'
'''

from nose.tools import assert_equal


def permute(s):
    out = []

    # Base case
    if (len(s) == 1):
        out = [s]

    else:
        # For every letter in string
        for i, let in enumerate(s):

            # For every permutation
            for perm in permute(s[:i] + s[i + 1:]):

                # Add it to the output
                out += [let + perm]

    return out


class TestPerm(object):

    def test(self, solution):

        assert_equal(sorted(solution('abc')), sorted(
            ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']))
        assert_equal(sorted(solution('dog')), sorted(
            ['dog', 'dgo', 'odg', 'ogd', 'gdo', 'god']))

        print('All test cases passed.')


# Run Tests
t = TestPerm()
t.test(permute)
