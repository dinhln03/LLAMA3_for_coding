numbers = input().split(', ')
numbers_list = list(map(int, numbers))
even_list = []

for i in range (len(numbers_list)):
    if numbers_list[i] % 2 == 0:
        even_list.append(i)
print(even_list)

# found_indices = map(lambda x: x if numbers_list[x] % 2 == 0 else 'no', range(len(numbers_list)))
# even_indices = list(filter(lambda a: a != 'no', found_indices))
# print(even_indices)
