#!/usr/bin/env python3


def main():
    pattern = input().upper()
    genome = input().upper()
    mismatches = int(input())

    occurrences = approximate_occurrences(genome, pattern, mismatches)
    for o in occurrences:
        print(o, end=' ')
    print()


LIST_A = ['C', 'T', 'G']
LIST_C = ['A', 'T', 'G']
LIST_T = ['C', 'A', 'G']
LIST_G = ['C', 'T', 'A']


def _generate_immediate_neighbours(pattern: str) -> list:
    """
    Generate immediate (different by one mismatch) neighbours of the given genome pattern
    :param pattern: a pattern to examine
    :return: neighbourhood, NOT including the given pattern
    """
    generated = []
    for i in range(len(pattern)):
        if pattern[i] == 'A':
            generated.extend([pattern[:i] + c + pattern[i + 1:] for c in LIST_A])
        elif pattern[i] == 'C':
            generated.extend([pattern[:i] + c + pattern[i + 1:] for c in LIST_C])
        elif pattern[i] == 'T':
            generated.extend([pattern[:i] + c + pattern[i + 1:] for c in LIST_T])
        elif pattern[i] == 'G':
            generated.extend([pattern[:i] + c + pattern[i + 1:] for c in LIST_G])

    return generated


def generate_neighbours(pattern: str, mismatches: int) -> set:
    """
    Generate neighbours for the given pattern (genome string)
    :param pattern: genome pattern
    :param mismatches: number of mismatches to generate neighbours
    :return: a set of patterns in the neighbourhood, including the 'pattern' itself
    """
    neighbourhood = set()
    neighbourhood.add(pattern)

    curr_patterns = [pattern]
    next_patterns = []

    for curr_mismatches in range(mismatches):
        for curr_pattern in curr_patterns:
            for neighbour in _generate_immediate_neighbours(curr_pattern):
                if neighbour not in neighbourhood:
                    neighbourhood.add(neighbour)
                    next_patterns.append(neighbour)

        curr_patterns = next_patterns
        next_patterns = []

    return neighbourhood


def approximate_occurrences(genome: str, pattern: str, mismatches: int) -> list:
    neighbours = generate_neighbours(pattern, mismatches)
    occurrences = set()

    for neighbour in neighbours:
        search_start = 0
        while search_start <= len(genome) - len(pattern):
            index_found = genome.find(neighbour, search_start)
            if index_found == -1:
                break
            occurrences.add(index_found)
            search_start = index_found + 1

    return sorted(list(occurrences))


if __name__ == '__main__':
    main()
