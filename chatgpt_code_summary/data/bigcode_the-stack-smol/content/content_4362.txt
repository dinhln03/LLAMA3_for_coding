import ast
import math
import copy

with open('input.txt', 'r') as f:
    lines = f.readlines()
lines = [line[:-1] for line in lines]
lines = [ast.literal_eval(line) for line in lines]


def reduce(num):
    explodeCriteriaMet = True
    splitCriteriaMet = True

    while explodeCriteriaMet or splitCriteriaMet:
        explodeCriteriaMet = False
        splitCriteriaMet = False

        # check for pair nested inside four pairs
        for idx1, val1 in enumerate(num):
            if explodeCriteriaMet:
                break

            if type(val1) == list:
                for idx2, val2 in enumerate(val1):
                    if explodeCriteriaMet:
                        break

                    if type(val2) == list:
                        for idx3, val3 in enumerate(val2):
                            if explodeCriteriaMet:
                                break

                            if type(val3) == list:
                                for idx4, val4 in enumerate(val3):
                                    if type(val4) == list:
                                        explodeCriteriaMet = True
                                        num = explode(num, val4, [idx1, idx2, idx3, idx4])
                                        break

        if explodeCriteriaMet:
            continue

        # regular number is 10 or greater
        for idx1, val1 in enumerate(num):
            if splitCriteriaMet:
                break

            if type(val1) == list:
                for idx2, val2 in enumerate(val1):
                    if splitCriteriaMet:
                        break

                    if type(val2) == list:
                        for idx3, val3 in enumerate(val2):
                            if splitCriteriaMet:
                                break

                            if type(val3) == list:
                                for idx4, val4 in enumerate(val3):
                                    if val4 >= 10:
                                        splitCriteriaMet = True
                                        num = split(num, val4, [idx1, idx2, idx3, idx4])
                                        break

                            elif val3 >= 10:
                                splitCriteriaMet = True
                                num = split(num, val3, [idx1, idx2, idx3])
                                break

                    elif val2 >= 10:
                        splitCriteriaMet = True
                        num = split(num, val2, [idx1, idx2])
                        break

            elif val1 >= 10:
                splitCriteriaMet = True
                num = split(num, val1, [idx1])
                break

    return num


def split(num, value, idx):
    # print("SPLIT")
    # print("IDX:", idx)
    # print("VAL:", value)

    pair = [math.floor(value/2), math.ceil(value/2)]

    if len(idx) == 4:
        num[idx[0]][idx[1]][idx[2]][idx[3]] = pair
    elif len(idx) == 3:
        num[idx[0]][idx[1]][idx[2]] = pair
    elif len(idx) == 2:
        num[idx[0]][idx[1]] = pair
    elif len(idx) == 1:
        num[idx[0]] = pair

    return num


def getValueAtIndex(num, idx):
    for i in idx:
        num = num[i]
    return num


def changeValueAtIndex(num, idx, value):
    if len(idx) == 5:
        num[idx[0]][idx[1]][idx[2]][idx[3]][idx[4]] += value
    elif len(idx) == 4:
        num[idx[0]][idx[1]][idx[2]][idx[3]] += value
    elif len(idx) == 3:
        num[idx[0]][idx[1]][idx[2]] += value
    elif len(idx) == 2:
        num[idx[0]][idx[1]] += value
    elif len(idx) == 1:
        num[idx[0]] += value
    return num


def explode(num, item, idx):
    # store values to add:
    left = item[0]
    right = item[1]
    # the exploding pair is replaced with the regular number 0
    num[idx[0]][idx[1]][idx[2]][idx[3]] = 0

    # adding the values to their neighbors
    # left:
    for index, indexValue in enumerate(idx[:: -1]):
        if indexValue != 0:  # there is a neighbour to the left when the indexValue is 1
            idxLeft = idx[: 4-index]
            idxLeft[-1] = 0

            while type(getValueAtIndex(num, idxLeft)) == list:
                idxLeft.append(1)
            num = changeValueAtIndex(num, idxLeft, left)
            break

    # right
    for index, indexValue in enumerate(idx[:: -1]):
        if indexValue != 1:  # there is a neighbour to the right when the indexValue is 0
            idxRight = idx[: 4-index]
            idxRight[-1] = 1

            while type(getValueAtIndex(num, idxRight)) == list:
                idxRight.append(0)
            num = changeValueAtIndex(num, idxRight, right)
            break

    return num


def calculateSum(num1, num2):
    sum = [copy.deepcopy(num1), copy.deepcopy(num2)]
    return reduce(sum)


def calculateMagnitude(num):
    left = num[0]
    right = num[1]
    if type(left) == list:
        left = calculateMagnitude(left)
    if type(right) == list:
        right = calculateMagnitude(right)

    return (3*left + 2*right)


# part 1
sum = lines[0]
for line in lines[1:]:
    sum = calculateSum(sum, line)

print("What is the magnitude of the final sum?", calculateMagnitude(sum))

# part 2
largest = 0

for x in lines:
    for y in lines:
        if x != y:
            magnitude = calculateMagnitude(calculateSum(x, y))
            if magnitude > largest:
                largest = magnitude

print("What is the largest magnitude of any sum of two different snailfish numbers from the homework assignment?", largest)
