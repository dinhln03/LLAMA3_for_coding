#!/usr/bin/env python3

# Macaw
#
# Testing file open and string concatenation.

import random
import pkgutil

def main():

    # This dictionary of words is for testing only and should *not* be considered secure.
    # Courtesy of https://gist.github.com/deekayen/4148741
    #f = open('dictionary.txt')
    f = pkgutil.get_data("macaw","dictionary.txt").decode("utf8")
    wordList = f.split()

    password = generatePassword(wordList)
    speakPassword(password)

def speakPassword(str):
    print(r"""
                   ,,,___
                 ,'   _  \__           ___________________________________________
                /    { O /  `\        /                                           \
               ,\     } /---./     .-'      """+str+"""
             /\  `-.__- `--'       `-.                                             |
            /  `._  :   |             \___________________________________________/
           /\_;  -' :   ;
           /  \_;  /   /
           /| \ \_/..-'
   ________|_\___/_\\\_\\\________
   ----------------;;-;;--------
           \/ `-'/
           |\_|_/|
            \/ \/
             \_/
    """)

def generatePassword(wordList):
    tempPass = ''
    for i in range(0, 5):
        word = wordList[random.randint(0,999)] # grab a random word from the dictionary file.
        tempPass = tempPass + word #concat that word to the end of the password.
    return tempPass
