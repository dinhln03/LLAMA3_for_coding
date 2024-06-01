"""DATA STRUCTURES"""
# Algorithms are set of rules used to solve a problem
# Data structures are a way of organizing data in a computer
# colors = ['red', 'yellow', [5, 6], 'blue']
friends = ['Josh', 'Renee', 'Agnes']
# print(colors)
# print(colors[1])
# colors[2] = 'green'  # mutability of lists
# print(colors)
# print(len(friends))
# print(len(colors))  # gives you the number of items in the list variable
# print(range(len(friends)))

# for i in range(len(friends)):  # loops through list when you know position of items
#    friend = friends[i]
#    print('Happy new year,', friend)

# for friend in friends:  # better for looping since you get to write less code
# print('Happy New Year, %s!' % friend)

numbers = [2, 4, 6, 8, 10]
for i in range(len(numbers)):  # range can also be used as such to update elements using indices
    numbers[i] = numbers[i] * 2
print(numbers)
