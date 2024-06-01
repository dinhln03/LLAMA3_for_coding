# https://repl.it/@thakopian/day-4-2-exercise#main.py

# write a program which will select a random name from a list of names
# name selected will pay for everyone's bill

# cannot use choice() function

# inputs for the names - Angela, Ben, Jenny, Michael, Chloe

# import modules
import random

# set varialbles for input and another to modify the input to divide strings by comma

names_string = input("Give me everybody's names, separated by a comma. ")
names = names_string.split(", ")

# get name at index of list (example)
print(names[0])

# you can also print len of the names to get their range
print(len(names))
# set random module for the index values
# > this is standard format > random.randint(0, x)

# using the len as a substitute for x in the randint example  with a variable set to len(names)
num_items = len(names)
# num_items - 1 in place of x to get the offset of the len length to match a starting 0 position on the index values
# set the function to a variable
choice = random.randint(0, num_items - 1)

# assign the mutable name variable with an index of the choice variable to another variable for storing the index value of the name based on the index vaule
person_who_pays = names[choice]
# print that stored named variable out with a message
print(person_who_pays + " is going to buy the meal today")


#######

# This exercise isn't a practical application of random choice since it doesn't use the .choice() function
# the idea is to replace variables, learn by retention and problem solve
# create your own random choice function to understand how the code can facilitate that withouth the .choice() function
# that way you learn how to go through problem challenges and how to create your own workaround in case the out of the box content isn't everything you need for a given problem
