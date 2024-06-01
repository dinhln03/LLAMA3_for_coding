# Write a Python function to sum all the numbers in a list
# Sample List : [8, 2, 3, 0, 7]
# Expected Output : 20



def sum_list(list):
    sum = 0
    for i in list:
        sum += i
    return sum

list = [8, 2, 3, 0, 7]
print(sum_list(list))