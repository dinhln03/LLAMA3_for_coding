# Enter your code for "Degree Distribution" here.

import csv

degrees = []
students = []

for l in csv.DictReader(open("degrees.csv")):
    degrees.append(l)

for l in csv.DictReader(open("students.csv")):
    students.append(l)

students = sorted(students, key=lambda x: float(x["score"]))
students.reverse()

print(students)
