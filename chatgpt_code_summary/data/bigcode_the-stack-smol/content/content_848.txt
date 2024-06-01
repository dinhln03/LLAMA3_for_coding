#!/usr/bin/env python3

import math
import csv
import itertools
from pprint import pprint

import func


INPUTFILE = './task04.input'


def main():
    accept = 0
    with open(INPUTFILE, mode='r') as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")

        lines = list(reader)
        for line in lines:
            reject_line = False
            for word in line:
                if line.count(word) > 1:
                    reject_line = True
                    break
            if not reject_line:
                accept = accept + 1

        print("file {} has {} lines".format(INPUTFILE, len(lines)))
        print("we accept {} of them".format(accept))
                    


if __name__ == '__main__':
    main()
