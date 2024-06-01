'''In ​Repetition Based on User Input​, you saw a loop that prompted users until they typed quit. This code won’t work if users type Quit, or QUIT, or any other version that isn’t exactly quit. Modify that loop so that it terminates if a user types that word with any capitalization.'''


text = ""
while text.lower()!= "quit":
    text = (input("Please Enter the command quit to exit this program,with any case you'd like:"))
    
    if text.lower() == "quit":
        print("...exitting program")
