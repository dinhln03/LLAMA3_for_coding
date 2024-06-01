##########################################################################
# Author: Samuca
#
# brief: returns the int part of number
#
# this is a list exercise available on youtube:
#   https://www.youtube.com/playlist?list=PLHz_AreHm4dm6wYOIW20Nyg12TAjmMGT-
##########################################################################

number = float(input("Enter with any number: "))
print("the int part of {} is {}".format(number, int(number)))

#we can also do it with the method trunc, from math
from math import trunc
n = float(input("Enter with other number: "))
print("The int part of {} is {}".format(n, trunc(n)))
