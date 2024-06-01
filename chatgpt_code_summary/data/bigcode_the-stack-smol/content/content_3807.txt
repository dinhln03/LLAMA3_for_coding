# -*- coding: utf-8 -*-


from functools import cache

INPUT = 33100000


def sigma_pentagonal_numbers(limit):
    """
    >>> list(sigma_pentagonal_numbers(16))
    [1, 2, 5, 7, 12, 15]
    """

    n = 1
    p = 1

    while p <= limit:
        yield p

        if n > 0:
            n = -n
        else:
            n = -n + 1

        p = (3 * n * n - n) // 2


def sigma_sign_generator():
    while True:
        yield 1
        yield 1
        yield -1
        yield -1


@cache
def presents_for_house(house):
    """
    https://math.stackexchange.com/a/22744

    >>> presents_for_house(1)
    10
    >>> presents_for_house(2)
    30
    >>> presents_for_house(3)
    40
    >>> presents_for_house(8)
    150
    >>> presents_for_house(9)
    130
    """

    if house == 1:
        return 10

    presents = 0
    sign = sigma_sign_generator()

    for p in sigma_pentagonal_numbers(house):
        n = house - p

        if n == 0:
            presents += house * next(sign) * 10
        else:
            presents += presents_for_house(n) * next(sign)

    return presents


def part1(data):
    """
    # Takes too long so commented out
    # >>> part1(INPUT)
    # 776160
    """

    house = 0
    presents = 0

    max = 0

    while presents < data:
        house += 1
        presents = presents_for_house(house)

        if presents > max:
            max = presents
            print(max)

    return house


def part2(data):
    """
    >>> part2(INPUT)
    786240
    """

    upper_limit = INPUT
    house = [0] * (upper_limit + 1)
    elf = 1

    while elf <= upper_limit:
        elf_end = min(elf * 50, upper_limit)

        for number in range(elf, elf_end + 1, elf):
            index = number - 1
            house[index] += 11 * elf

            if house[index] >= data:
                upper_limit = min(number, upper_limit)

        elf += 1

    for i, value in enumerate(house):
        if value >= data:
            return i + 1

    raise ValueError()


def main():
    print(part1(INPUT))
    print(part2(INPUT))


if __name__ == "__main__":
    main()
