names = ["John", "Bob", "Dell", "python"];
print(names[0])
print(names[-1])
print(names[-2])

names[0] = "Amina"

print(names[0])

print(names[0:3])

# List methods
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.insert(0, -1)
numbers.remove(3)
is_there = 1 in numbers
numbers.count(3) # it will return count of 3
# numbers.sort() # Ascending order

numbers.reverse() # descending order

numbers = numbers.copy() # To clone original list

print(is_there)
print(numbers)
print(len(numbers))
numbers.clear()
print(numbers)
