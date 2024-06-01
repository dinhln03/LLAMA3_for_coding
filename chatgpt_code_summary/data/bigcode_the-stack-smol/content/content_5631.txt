# Solution to Problem 8
# Program outputs today's date and time in the format "Monday, January 10th 2019 at 1:15pm"
# To start we import the Python datetime module as dt.
from datetime import datetime as dt

#now equals the date and time now.
now = dt.now()

# Copied verbatim initially from stacoverflow Reference 1 below but amended to fit my referenceing of time as now.
# Suffix equals 'st' if the date now is 1,21 or 23 else it is 'nd' if the date noe is 2 or 22 else it is 'rd' if date now is 3 or23 for eveything else it is 'th.
suffix = 'st' if now in [1,21,31] else 'nd' if now in [2, 22] else 'rd' if now in [3, 23] else 'th'

# Display to the user the Heading "Todays Date and Time:"
print("Todays Date and time:") 
# Below displays to the user a the date and time in a string in inverted commas todays date and time in the format Day, Month Date year at Current Time am/pm.
# Used Reference 3 below to remove the leading 0 when desplaying the time. 
print(now.strftime('%A, %B %d%%s %Y at %#I:%M %p',) % suffix,)

# Reference 1: https://stackoverflow.com/a/11645978
# Reference 2: https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
# Reference 3: https://stackoverflow.com/questions/904928/python-strftime-date-without-leading-0One problem is that '{dt.hour}' uses a 24 hour clock :(. Using the second option still brings you back to using '{%#I}' on Windows and '{%-I}' on Unix. – ubomb May 24 '16 at 22:47 
# Used lecture from week 6 as a base for the problem also looked at the Python tutorial.
# Laura Brogan 19/03/2019