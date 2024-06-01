def get_the_ith_largest(s1: list, s2: list, i: int):
    m = len(s1)
    n = len(s2)

    if i > m + n:
        raise IndexError('list index out of range')

    i -= 1
    l1 = 0
    r1 = i if m - 1 >= i else m - 1
    while l1 <= r1:
        c1 = (l1 + r1) // 2
        c1_f = i - c1 - 1
        c1_b = i - c1
        if c1_f >= 0 and (c1_f >= n or s2[c1_f] > s1[c1]):
            l1 = c1 + 1
        elif 0 <= c1_b < n and s2[c1_b] < s1[c1]:
            r1 = c1 - 1
        else:
            return s1[c1]

    return get_the_ith_largest(s2, s1, i + 1)


if __name__ == '__main__':
    s1_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    s2_test = [2, 3, 4, 6, 10, 20, 100]
    print(get_the_ith_largest(s2_test, s1_test, 8))
