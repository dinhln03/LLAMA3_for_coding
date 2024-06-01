from os.path import realpath


def main():
    inpString = open(f'{realpath(__file__)[:-2]}txt').read()

    inpString += '0' * (3 - len(inpString) % 3)         # Padding to make it divisible by 3
    inp = list(inpString)

    for i in range(len(inp)):
        if inp[i] not in '0123456789abcdef':
            inp[i] = '0'

    inp = ''.join(inp)

    v = len(inp)//3

    for i in range(0, len(inp), v):
        print(inp[i : i + 2], end='')                   # Print first 2 char of every 1/3rd part of input


if __name__ == '__main__':
    main()