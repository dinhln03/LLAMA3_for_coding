from collections import Counter


def part1(lines):
    gamma = ''
    epsilon = ''
    num_bits = len(lines[0])

    for i in range(num_bits):
        most_common = Counter(map(lambda x: x[i], lines)).most_common(1)[0][0]
        gamma += most_common
        epsilon += '0' if most_common == '1' else '1'

    return int(gamma, base=2) * int(epsilon, base=2)

def get_value(data, default):
    for i in range(len(data[0])):
        cntr = Counter(map(lambda x: x[i], data))

        most_common = cntr.most_common(1)[0][0] if default == '1' else cntr.most_common()[-1][0][0]

        if cntr.most_common(1)[0][1] == cntr.most_common()[-1][1] and len(cntr.most_common()) > 1:
            most_common = default

        data = list(filter(lambda x: x[i] == most_common, data))

        if len(data) < 2:
            break

    print(data[0])
    return int(data[0], base=2)

def part2(lines):
    return get_value(lines, '1') * get_value(lines, '0')


def main():
    with open('input.txt', 'r') as f:
        lines = f.read().splitlines()

    print(f'Part 1: {part1(lines)}')
    print(f'Part 2: {part2(lines)}')


if __name__ == '__main__':
    main()
