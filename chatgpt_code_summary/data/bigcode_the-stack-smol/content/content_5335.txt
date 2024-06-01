repeat_age = 0
while repeat_age < 3:
    age = int(
        input(f"Check your movie ticket price by typing your age below. You may check {3 - repeat_age} more times\n"))
    if age < 3:
        print("Your ticket is free!")
    elif 3 <= age <= 12:
        print("Your ticket costs $10")
    elif age > 12:
        print("Your ticket costs $15")
    if repeat_age == 2:
        break
    check = input("Would you like to check another ticket price? Type 'quit' to exit this program\n")
    if check == 'quit':
        break
    repeat_age += 1
print(f"Thank you for using our service! You have checked {repeat_age + 1} times")
