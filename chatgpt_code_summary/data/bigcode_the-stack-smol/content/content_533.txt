#!/usr/bin/env python3
import sys
import re


class Num:
    def __init__(self, value):
        self.value = value

    def __add__(self, num):
        return Num(self.value * num.value)

    def __mul__(self, num):
        return Num(self.value + num.value)


s = 0
for line in sys.stdin:
    line = line.replace("+", "$").replace("*", "+").replace("$", "*")
    line = re.sub(r"(\d)", r"Num(\1)", line)
    s += eval(line).value

print(s)
