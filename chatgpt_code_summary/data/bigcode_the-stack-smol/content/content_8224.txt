#!/bin/python3

# Copyright (C) 2020 Matheus Fernandes Bigolin <mfrdrbigolin@disroot.org>
# SPDX-License-Identifier: MIT

"""Day Thirteen, Shuttle Search."""


from sys import argv
from re import findall

from utils import open_file, arrange, usage_and_exit, product


def solve1(buses, est):
    """Get the earliest bus from the <buses> according to the <est>imate
    time.  """

    arrival = [bus - est%bus for bus in buses]
    earliest = min(arrival)

    return min(arrival)*buses[arrival.index(earliest)]


def solve2(buses, depart):
    """Find the smallest timestamp, such that all the <buses> follow their
    bus ID, which is indexically paired with <depart>.

    Here I used the Chinese Remainder Theorem, someone well acquainted to
    anyone who does competitive or discrete mathematics.  """

    # Desired residue class for each bus.
    mods = [(b - d) % b for b, d in zip(buses, depart)]

    # Cross multiplication of the elements in the sequence.
    cross_mul = [product(buses)//b for b in buses]

    return sum([c*pow(c, -1, b)*m for b, c, m
               in zip(buses, cross_mul, mods)]) % product(buses)


if __name__ == "__main__":
    usage_and_exit(len(argv) != 2)

    input_file = arrange(open_file(argv[1]))

    bus_data = [int(b) for b in findall(r"\d+", input_file[1])]
    estimate = int(input_file[0])
    depart_data = [i for i,d in enumerate(findall(r"\w+", input_file[1]))
                   if d != "x"]

    print(solve1(bus_data, estimate))
    print(solve2(bus_data, depart_data))
