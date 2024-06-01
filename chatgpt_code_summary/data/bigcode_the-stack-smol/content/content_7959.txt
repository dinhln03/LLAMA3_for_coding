"""
    Quick Sort
    ----------
    Uses partitioning to recursively divide and sort the list

    Time Complexity: O(n**2) worst case

    Space Complexity: O(n**2) this version

    Stable: No

    Psuedo Code: CLRS. Introduction to Algorithms. 3rd ed.

"""

count = 0
def sort(seq):
    """
    Takes a list of integers and sorts them in ascending order. This sorted
    list is then returned.

    :param seq: A list of integers
    :rtype: A list of sorted integers
    """
    global count
    if len(seq) <= 1:
        return seq
    else:
        pivot = seq[0]
        left, right = [], []
        for x in seq[1:]:
            count += 1
            if x < pivot:
                left.append(x)
            else:
                right.append(x)
        return sort(left) + [pivot] + sort(right)

if __name__ == '__main__':
    # print sort([9,8,7,6,5,4,3,2,1,0])
    print sort([1,2,3,4,5,6,7,8,9,10])
    print count
