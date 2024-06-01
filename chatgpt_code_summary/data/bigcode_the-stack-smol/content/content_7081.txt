#!/usr/bin/env python3
def sum_recursin(numList):
    if len(numList) == 1:
        return numList[0]
    else:
        return numList[0] + sum_recursin(numList[1:])


if __name__ == "__main__":
    print(sum_recursin(list(range(1, 101))))
