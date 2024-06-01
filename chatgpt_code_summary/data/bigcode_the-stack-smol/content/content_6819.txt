#!/usr/bin/python3
import time
from calcul import *
import sys
import os

max_exec = 10
red = "\033[31m"
white = "\033[39m"
cyan = "\033[36m"
green = "\033[32m"

save = sys.stdout
so = open("file.log", 'w')
sys.stdout = so

def rectangle_time(n):
    time_rect = []
    i = 0
    while i < max_exec:
        start_time = time.time()
        calcul_rectangles(n)
        time_rect.append(time.time() - start_time)
        i += 1
    return time_rect

def trapeze_time(n):
    time_trap = []
    i = 0
    while i < max_exec:
        start_time = time.time()
        calcul_trapezoïds(n)
        time_trap.append(time.time() - start_time)
        i += 1
    return time_trap

def simpson_time(n):
    time_simp = []
    i = 0
    while i < max_exec:
        start_time = time.time()
        calcul_simpson(n)
        time_simp.append(time.time() - start_time)
        i += 1
    return time_simp

def calc_dict(tab, name):
    i = 0
    result = 0
    dic = {}
    while i < max_exec:
        result += tab[i]
        i += 1
    result = result / max_exec
    dic["Name"] = name
    dic["Value"] = result
    return dic

def get_min_time(dict1, dict2, dict3):
    if dict1.get("Value") < dict2.get("Value") and dict1.get("Value") < dict3.get("Value"):
        return 1
    if dict2.get("Value") < dict1.get("Value") and dict2.get("Value") < dict3.get("Value"):
        return 2
    if dict3.get("Value") < dict2.get("Value") and dict3.get("Value") < dict1.get("Value"):
        return 3

def get_min_precision(prec1, prec2, prec3):
    prec1 = abs(prec1)
    prec2 = abs(prec2)
    prec3 = abs(prec3)
    if prec1 < prec2 and prec1 < prec3:
        return 1
    if prec2 < prec1 and prec2 < prec3:
        return 2
    if prec3 < prec2 and prec3 < prec1:
        return 3

def main():
    n = int(sys.argv[1])
    time_rect = rectangle_time(n)
    time_trap = trapeze_time(n)
    time_simp = simpson_time(n)
    dict_rect = calc_dict(time_rect, "Rectangles")
    dict_trap = calc_dict(time_trap, "Trapezoids")
    dict_simp = calc_dict(time_simp, "Simpson")
    preci_rect = calcul_rectangles(n) - (pi / 2)
    preci_trap = calcul_trapezoïds(n) - (pi / 2)
    preci_simp = calcul_simpson(n) - (pi / 2)
    sys.stdout = save
    print("{}Compute time:\n{}".format(cyan, white))
    print("Method : {}\t: {}{:.6f}{} sec".format(dict_rect.get("Name"), red, dict_rect.get("Value"), white))
    print("Method : {}\t: {}{:.6f}{} sec".format(dict_trap.get("Name"), red, dict_trap.get("Value"), white))
    print("Method : {}\t: {}{:.6f}{} sec".format(dict_simp.get("Name"), red, dict_simp.get("Value"), white))
    min_time = get_min_time(dict_rect, dict_trap, dict_simp)
    print("The fastest Method is:", end='')
    print(green, end='')
    if min_time == 1:
        print("\tRectangles Method")
    elif min_time == 2:
        print("\tTrapezoids Method")
    else:
        print("\tSimpson Method")
    print(white, end='')
    print("\n{}Relative precision:\n{}".format(cyan, white))
    print("Method : {}\t: {}{}{} a.u.".format(dict_rect.get("Name"), red, preci_rect, white))
    print("Method : {}\t: {}{}{} a.u.".format(dict_trap.get("Name"), red, preci_trap, white))
    print("Method : {}\t: {}{}{} a.u.".format(dict_simp.get("Name"), red, preci_simp, white))
    preci = get_min_precision(preci_rect, preci_trap, preci_simp)
    print("The most accurate:", end='')
    print(green, end='')
    if preci == 1:
        print("\tRectangles Method")
    elif preci == 2:
        print("\tTrapezoids Method")
    else:
        print("\tSimpson Method")
    print(white, end='')

main()
