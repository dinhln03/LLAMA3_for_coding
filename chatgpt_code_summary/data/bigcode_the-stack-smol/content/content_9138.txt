'''
@Date: 2019-08-22 20:40:54
@Author: ywyz
@LastModifiedBy: ywyz
@Github: https://github.com/ywyz
@LastEditors: ywyz
@LastEditTime: 2019-08-22 20:48:24
'''
years, months = eval(input("Enter years and months: "))
if (months == 1 or months == 3 or months == 5 or months == 7 or months == 8
        or months == 10 or months == 12):
    print(years, ".", months, " has 31 days. ")
elif (months == 4 or months == 6 or months == 9 or months == 11):
    print(years, ".", months, "has 30 days. ")
elif (months == 2):
    if (years % 4 == 0 and years % 100 != 0) or (years % 400 == 0):
        print(years, ".", months, "has 29 days. ")
    else:
        print(years, ".", months, "has 28 days. ")
else:
    print("Wrong Input!")
