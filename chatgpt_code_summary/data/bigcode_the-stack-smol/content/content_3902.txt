answer = input ("Would you like to play?")

if answer.lower().strip() == "yes":
    print ("Yay! Let's get started.")

    answer = input ("You have reached an apple tree, would you like to pick an apple?").lower ().strip()
    if answer == "yes":
        answer = input ("would you like to eat the apple?")

        if answer == "yes":
            print ("That was not a great idea!")
        else:
             print ("good choice, you made it out safely.")

             answer = input ("you encounter the apple tree owner and are accussed of stealing. would you like to? (run/apologize)")

             if answer == "run":
                print ("you have been arressted! Game Over!")
            else:
                print ("you have won! Congratulations!")

    elif answer == "no":
        print ("congratulations you have won!")
    else: 
        print ("Invalid choice, you lost!")
else: 
    print ("Aww that's so sad")
