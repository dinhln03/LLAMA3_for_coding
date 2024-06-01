def combination(n, r):
    """
    :param n: the count of different items
    :param r: the number of select
    :return: combination
    n! / (r! * (n - r)!)
    """
    r = min(n - r, r)
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    for i in range(1, r + 1):
        result //= i
    return result


def comb2():
    # from scipy.misc import comb
    pass
