###############################################################################
#
# DONE:
#
#   1. READ the code below.
#   2. TRACE (by hand) the execution of the code,
#        predicting what will get printed.
#   3. Run the code and compare your prediction to what actually was printed.
#   4. Decide whether you are 100% clear on the CONCEPTS and the NOTATIONS for:
#        -- DEFINING a function that has PARAMETERS
#        -- CALLING a function with actual ARGUMENTS.
#
# *****************************************************************************
#      If you are NOT 100% clear on the above concepts,
#      ask your instructor or a student assistant about them during class.
# *****************************************************************************
#
# After you have completed the above, mark this _TODO_ as DONE.
#
###############################################################################


def main():
    hello("Snow White")
    goodbye("Bashful")
    hello("Grumpy")
    hello("Sleepy")
    hello_and_goodbye("Magic Mirror", "Cruel Queen")


def hello(friend):
    print("Hello,", friend, "- how are things?")


def goodbye(friend):
    print("Goodbye,", friend, '- see you later!')
    print('   Ciao!')
    print('   Bai bai!')


def hello_and_goodbye(person1, person2):
    hello(person1)
    goodbye(person2)


main()
