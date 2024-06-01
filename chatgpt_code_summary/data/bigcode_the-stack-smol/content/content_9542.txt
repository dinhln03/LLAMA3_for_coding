def main():
    with open('inputs/01.in') as f:
        data = [int(line) for line in f]

    print(sum(data))

    result = 0
    seen = {0}
    while True:
        for item in data:
            result += item
            if result in seen:
                print(result)
                return
            seen.add(result)


if __name__ == '__main__':
    main()
