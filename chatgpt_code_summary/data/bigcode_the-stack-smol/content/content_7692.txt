# =================================================================
# IMPORT REQUIRED LIBRARIES
# =================================================================
import os

# =================================================================
# READ DATA
# =================================================================

data_location = os.path.join(os.path.abspath(""), '2021/day-06-lanternfish/')

# with open(os.path.join(data_location, 'input_small.txt'), 'r') as f:
with open(os.path.join(data_location, 'input.txt'), 'r') as f:
    data = f.read().split(",")

data = [int(fish) for fish in data] 

# print(data)

# =================================================================
# LOGIC - PART ONE
# =================================================================
def part_one():
    numDays = 18
    numFishes = len(data)

    # print("Initial state: ", data)

    # Each day
    for day in range(numDays):
        for i in range(numFishes):
            fish = data[i]
            if fish == 0:
                # a 0 becomes a 6
                data[i] = 6
                # and adds a new 8 to the end of the list
                data.append(8)
            else:
                data[i] = fish-1
        numFishes = len(data)
        # print("After ", str(day), " day: ", data)

    return(numFishes)

# =================================================================
# LOGIC - PART TWO
# =================================================================
def part_two():
    numDays = 256
    fishesDict = {}

    for i in range(9):
        fishesDict[i] = 0
        fishesDict[i] = data.count(i)

    # print(fishesDict)
    # print("Initial state: ", fishesDict)

    # Each day
    for day in range(numDays):
        newFishesDict = {}
        for i in range(9):
            newFishesDict[i] = 0

        holder = 0
        for i in fishesDict:
            if i == 0:
                holder += fishesDict[0]
            else:
                newFishesDict[i-1] = fishesDict[i]
        # A 0 becomes a 6
        newFishesDict[6] += holder
        # and adds a new 8 to the end of the list
        newFishesDict[8] += holder
        fishesDict = newFishesDict
        # print("After ", str(day+1), " day: ", fishesDict)
    
    numFishes = 0
    for i in range(9):
        numFishes += fishesDict[i]

    return(numFishes)

# =================================================================
# MAIN
# =================================================================
if __name__ == '__main__':
    # print("Part one result is: " , str(part_one()))
    print("Part two result is: " , str(part_two()))