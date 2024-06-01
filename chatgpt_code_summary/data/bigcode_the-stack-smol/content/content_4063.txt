a = []
# append element at the end.
a.append(2)
a.append(3)
print(a)
# insert at a specific location.
a.insert(0, 5)
a.insert(10, 5)
print(a)
# when specified a position not in list, it inserts at the end.
a.insert(100, 6)
print(a)

# Deleting elements from a list.
a.remove(5)  # removes the first occurence of value passed
print(a, len(a))
del a[0]
print(a, len(a))

# access the last element
print(a[-1])


# Printing a list
print(len(a))
for item in range(len(a)):  # the len is not inclusive
    print("(", item, ", ", a[item], ")")
print("-" * 30)

for item in range(0, len(a), 1):  # the len is not inclusive
    print("(", item, ", ", a[item], ")")
print("-" * 30)


# Reverse printing a list
for item in range(len(a) - 1, -1, -1):  # the len is not inclusive
    print("(", item, ", ", a[item], ")")
print("-" * 30)

# Jump a certain number of times.
for item in range(0, len(a), 2):  # the len is not inclusive
    print("(", item, ", ", a[item], ")")
print("-" * 30)
