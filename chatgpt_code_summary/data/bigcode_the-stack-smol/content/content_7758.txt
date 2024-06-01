'''
Задача 1

Вывести на экран циклом пять строк из нулей, причем каждая строка должна быть пронумерована.
'''

# for i in range(1, 6):
#     print(f'{i} --', '0'*i)
#######
# print('num 1')
# i = 0
# while i < 5:
#     i += 1
#     print('line', i, 'is 0')
#
# print('')
#######

'''
Задача 2

Пользователь в цикле вводит 10 цифр. Найти количество введеных пользователем цифр 5.
'''
# amount = 0
# for i in range(10):
#     n = input()
#     if '5' in n:
#         amount += 1
# print(amount)
#########
# print('num 2')
# sum = 0
# for i in range(10):
#     answer = int(input('Напишите цифру: '))
#     if answer == 5:
#         sum += 1
#
# print('Количество 5 =', sum)
# print('')

#########

'''
Задача 3

Найти сумму ряда чисел от 1 до 100. Полученный результат вывести на экран.
'''
# sum = 0
#
# for i in range(1,101):
#     sum+=i
# print(sum)
#############
# print('num 3')
# sum = 0
# for i in range(1,101):
#     sum+=i
# print(sum)
# print('')
################


'''
Задача 4

Найти произведение ряда чисел от 1 до 10. Полученный результат вывести на экран.
'''
# n = 1
# for i in range(2, 11):
#     n *= i
# print(n)
################
# print('num 4')
# comp = 1
# for i in range(1,11):
#     comp *= i
# print(comp)
# print('')
################
'''
Задача 5

Вывести цифры числа на каждой строчке.
'''

# integer_number = 2129
#
# #print(integer_number%10,integer_number//10)
#
# while integer_number>0:
#     print(integer_number%10)
#     integer_number = integer_number//10
################

# # print('num 5')
# integer_number = 2129
# while integer_number>0:
#     print(integer_number%10)
#     integer_number = integer_number//10
# print('')
#
################

'''
Задача 6

Найти сумму цифр числа.
'''
# inter_number = 12345
# n = 0
# for i in str(inter_number):
#     n += int(i)
# print(n)

################
# inter_number = 12345
# n = 0
# for i in str(inter_number):
#     n += int(i)
# print(n)
################



'''
Задача 7

Найти произведение цифр числа.
'''
# n = 1
# for i in str(inter_number):
#     n *= int(i)
# print(n)
################
# inter_number = 12345
# x = 1
# for i in str(inter_number):
#     x *= int(i)
# print(x)

################

'''
Задача 8

Дать ответ на вопрос: есть ли среди цифр числа 5?
'''
# integer_number = 213413
# while integer_number>0:
#     if integer_number%10 == 5:
#         print('Yes')
#         break
#     integer_number = integer_number//10
# else: print('No')

################
# print('num 8')
# integer_number = 2134135
# while integer_number>0:
#     if integer_number%10 == 5:
#         print('Yes')
#         break
#     integer_number = integer_number//10
# else: print('No')
# print('')
################

'''
Задача 9

Найти максимальную цифру в числе
'''
# inter_number = 59675
# maximum = 0
# # print(max(str(inter_number)))
# while inter_number != 0:
#     if inter_number % 10 > maximum:
#         maximum = inter_number % 10
#     inter_number //= 10
# print(maximum)

################
# print('num 9')
# integer_number = 59675
# tmp = 0
# while integer_number>0:
#     if integer_number%10 >= tmp:
#         tmp = integer_number%10
#     integer_number = integer_number//10
# print(tmp)
# print('')
################

'''
Задача 10

Найти количество цифр 5 в числе
'''
# inter_number = 595675
# count = 0
# while inter_number != 0:
#     if inter_number % 10 == 5:
#         count += 1
#     inter_number //= 10
# print(count)
################
# print('num 10')
# integer_number = 595675
# tmp = 0
# while integer_number>0:
#     if integer_number%10 == 5:
#         tmp += 1
#     integer_number = integer_number//10
# print(tmp)
# print('')
################