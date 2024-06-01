# 6. Больше числа п. В программе напишите функцию, которая принимает два
# аргумента: список и число п. Допустим, что список содержит числа. Функция
# должна показать все числа в списке, которые больше п.
import random


def main():
    list_num = [random.randint(0, 100) for i in range(20)]
    print(list_num)
    n = int(input('entered n: '))
    print("This is list " + str(check_n(list_num, n)) + " of numbers\nthat are "
            "greater than the number you provided ", n, ".", sep="")


def check_n(list_num, n):
    num_greater_n = []
    for i in range(len(list_num)):
        if list_num[i] > n:
            num_greater_n.append(list_num[i])
    return num_greater_n


main()
