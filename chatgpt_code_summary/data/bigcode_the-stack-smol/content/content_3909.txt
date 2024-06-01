import random
from time import sleep


def Guess():
    global attempts
    # If the user choose anything but a number between 0 and 10, they will get stuck in loop.
    while True:
        try:
            attempts += 1  # This will count every attempt made by the user
            user_number = int(input().replace(' ', ''))
        except:
            print("You should put a number between 0 and 10 <3")
        else:
            if user_number > 10 or user_number < 0:
                print("I told you a number between 0 and 10 <3")
            else:
                break
    return user_number


def NextGame():
    # If the user choose anything but "[S] or [N]", they will get stuck in loop.
    while True:
        choice = input(
            "Do you want to play again? [S]/[N] ").upper().replace(' ', '')
        if (choice in "[S]" or choice in "[N]") and choice not in "[]":
            break
        else:
            print("I didn't understand your choice.", end=' ')
    return choice


# Introduction
print("\033[1;36m=-"*20, "\033[m")
print(f'\033[1;36m {"Lets play Number Guesser!":^40}\033[m')
print("\033[1;36m=-"*20, "\033[m")
sleep(2)

# The user will choose a mode or will get stuck in a loop until they do so.
while True:
    mode = input(
        "\nFirst of all, choose a mode: \n[1] Normal mode \n[2] Hide the thimble\n").replace(' ', '')
    while True:
        if mode.isnumeric() == False or int(mode) != 1 and int(mode) != 2:
            mode = input("I said to you to choose 1 or 2.\n")
        else:
            break

    # If the user choose the "normal mode"
    if int(mode) == 1:
        while True:
            # It will reset the amount of attempts every time the player choose to play it.
            attempts = 0

            # The computer will choose a random number
            print("I chose a number between 0 and 10, try to guess it! ")
            while True:
                pc_number = random.randint(0, 10)

                # The user will type a number between 0 and 10 or will get stuck in a loop until they do so.
                user_number = Guess()
                if user_number != pc_number:
                    print(
                        "Oops! You are wrong, let me chose another number... Guess it!")

                # When the user win
                else:
                    break
            print(f"Yes! You are right! You made it with {attempts} attempts!")

            # The user choices if they want to play again or not.
            choice = NextGame()
            break
        if choice not in "[S]":
            break

    elif int(mode) == 2:  # If the user choose the "Hide the thimble mode"
        # It will reset the amount of attempts every time the player choose to play it.
        attempts = 0
        # The computer will choose a random number
        pc_number = random.randint(0, 10)
        print("I chose a number between 0 and 10, try to guess it!")

        # The user will choose a number between 0 and 10, otherwise they will get stuck in a loop.
        while True:
            user_number = Guess()
            if pc_number == user_number:  # If the user number is the same as the computer one, the user wins!
                break
            # If the user's choice is 2 numbers or less apart from the computer one, the user will know they are getting close.
            elif pc_number > user_number >= pc_number-2 or pc_number < user_number <= pc_number+2:
                print("Hot.")

            # Else, they know they aren't close to the computer's number.
            else:
                print("Cold.")

        # When the user win
        print(f"Yes! You are right! You made it with {attempts} attempts!")
        choice = NextGame()
        if choice not in "[S]":
            break

# Goodbye
print(f"\nBye, bye! I'll miss you <3")
print("\033[1;34;107mBy: Kaique Apolinário\033[m")
