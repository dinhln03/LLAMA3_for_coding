from typing import List, Generator


def n_gons(partial: List[int], size: int, sums: int=None) -> \
        Generator[List[int], None, None]:
    length = len(partial)

    if length == size * 2:
        yield partial

    for i in range(1, size * 2 + 1):
        if i in partial:
            continue

        partial.append(i)

        if length == 2:
            sums = sum(partial[0: 3])
        elif (length > 2 and length % 2 == 0 and
              sums != sum(partial[-1: -4: -1]))\
                or \
             (length == size * 2 - 1 and sums != partial[1] + partial[-1] +
              partial[-2]):
            partial.pop()
            continue

        yield from n_gons(list(partial), size, sums)

        partial.pop()


def n_gon_to_representation(n_gon: List[int]) -> int:
    n_gon_str = [str(n) for n in n_gon]
    size = len(n_gon_str) // 2

    result = ''

    minimal = min(n_gon[0], *n_gon[3::2])
    index = n_gon.index(minimal)
    start = n_gon.index(minimal) // 2 if index >= 3 else 0

    for i in range(start, start + size):
        current = i % size

        if current == 0:
            result += ''.join(n_gon_str[0:3])
        elif current == size - 1:
            result += ''.join([n_gon_str[-1], n_gon_str[-2], n_gon_str[1]])
        else:
            result += ''.join([n_gon_str[current * 2 + 1],
                               n_gon_str[current * 2],
                               n_gon_str[current * 2 + 2]])

    return int(result)


def solve() -> int:
    return max([n_gon_to_representation(n_gon)
                for n_gon in n_gons([], 5)
                if n_gon_to_representation(n_gon) < 10 ** 16])
