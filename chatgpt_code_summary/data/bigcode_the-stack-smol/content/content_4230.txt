#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:23:08 2017
Author: Zachary W. Mikus

"""
#These are testing variables
d1 = {1:30, 2:20, 3:30, 5:80}
d2 = {1:40, 2:50, 3:60, 4:70}


def f(x, y):
    k = x + y
    return k

def commonKeys(longerList, shorterList):
    commonKeyList = []
    #Variables
    #intersectDictionary = The final returned intersect dictionary
    #commonKeyList = The list of keys that appear in both dictionaries
    for i in range(len(longerList)):
        if longerList[i] in shorterList:
            commonKeyList.append(longerList[i])
    return commonKeyList


def differentKeys(longerList, shorterList):
    #This function uses similar logic to the commonKeys function
    #Except it will see if the index is NOT in the other list and remove it
    #This runs the loop twice once through each loop to find the missing numbers
    #in each list
    
    differentKeyList = []
    for i in range(len(longerList)):
        if longerList[i] not in shorterList:
            differentKeyList.append(longerList[i])
    for i in range(len(shorterList)):
        if shorterList[i] not in longerList:
            differentKeyList.append(shorterList[i])
    return differentKeyList


def intersect(commonList, d1, d2):
    intersectDict = {}
    #This function takes the common list of keys, grabs the common values in 
    #both dictionaries and performs the f(x, y) function on them
    for i in range(len(commonList)):
       #currentIndex is the index in the dictionary, it will move 
       currentIndex = commonList[i]
       x = d1[currentIndex]
       y = d2[currentIndex]
       functionValue = f(x, y)
       intersectDict[currentIndex] = functionValue
    return intersectDict

def difference(differentKeyList, d1, d2):
    differenceDict = {}
    #This function takes the different list of keys, grabs the relevant values and
    #creates a dictionary
    #searches d
    for i in range(len(differentKeyList)):
        currentIndex = differentKeyList[i]
        if currentIndex in d1:
            differenceDict[currentIndex] = d1[currentIndex]
        if currentIndex in d2:
            differenceDict[currentIndex] = d2[currentIndex]
    return differenceDict


def diff_dictionary(d1, d2):
    differentKeyList = []
    #Turns key values in lists and finds the longest
    #keyListD1 = list of keys in d1
    #keyListD2 = list of keys in d2
    keyListD1 = list(d1.keys())
    keyListD2 = list(d2.keys())
    
    
    #determines which of the two lists is the longest and assigned it values
    #for the common list function
    if len(keyListD1) > len(keyListD2):
        longerList = keyListD1
        shorterList = keyListD2
        
    else:
        longerList = keyListD2
        shorterList = keyListD1
        
    #Finds the common keys
    commonList = commonKeys(longerList, shorterList)
    #Makes the intersect dictionary
    intersectDict = intersect(commonList, d1, d2)
    #Finds the different keys
    differentKeyList = differentKeys(longerList, shorterList)
    #Makes the different key dictionary
    differenceDict = difference(differentKeyList, d1, d2)
    #This now creates a list of the dictionaries put together
    return (intersectDict, differenceDict)

'''
#This is for calculating the difference dictionary. 
#The difference dictionary consists of every
#KEY VALUE# in the dictionaries that does not exist 
#in the other dictionary.
'''
#Variables
#differenceDictionary = The final returned difference dictionary
    


print(diff_dictionary(d1, d2))