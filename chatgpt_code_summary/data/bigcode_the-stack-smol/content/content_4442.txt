# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:46:26 2022

@author: Pedro
"""

def search(lista, target) -> int:
    for i in range(len(lista)):
        if lista [i] == target:
            return i
    return -1

def search2(lista, target) -> int:
    for i, element in enumerate(lista):
        if element == target:
            return i
    return -1