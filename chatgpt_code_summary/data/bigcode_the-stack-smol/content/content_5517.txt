import itertools

preamble = 25
numbers = []

with open('input.txt') as file:
    for line in file:
        numbers.append(int(line))

print(numbers)

for i in range(preamble, len(numbers)):
    pairs = itertools.combinations(numbers[i-preamble: i], 2)
    match = False
    for pair in pairs:
        if sum(pair) == numbers[i]:
            match = True
            break
    
    if not match:
        invalid_num = numbers[i]
        print(invalid_num)
        break

for i in range(len(numbers)):
    sum_len = 2
    while sum(numbers[i:i+sum_len]) < invalid_num:
        sum_len += 1
    else:
        if sum(numbers[i:i+sum_len]) == invalid_num:
            print(numbers[i:i+sum_len])
            print(min(numbers[i:i+sum_len]) + max(numbers[i:i+sum_len]))
            exit(0)
    

