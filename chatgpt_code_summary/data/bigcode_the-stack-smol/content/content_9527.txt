# Script that cuts the greet string into chunks and prints it out

greet = "Hello World!"
print(greet)

print("Start: ", greet[0:3])
print("Middle: ", greet[3:6])
print("End: ", greet[-3:])

a = greet.find(",")

print("Portion before comma", greet[:a])
