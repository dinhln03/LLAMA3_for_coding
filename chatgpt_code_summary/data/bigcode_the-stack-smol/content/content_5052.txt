# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 11:56:36 2017

Problemset1 - Problem 1
Note:
    's' is given by system like s = 'azcbobobegghakl'
@author: coskun
"""
s = 'azcbobobegghakl'
# Paste your code into this box 
nvl=0
for c in s:
    if c=='a' or c=='e' or c=='i' or c=='o' or c=='u':
        nvl += 1
print("Number of vowels: " + str(nvl))