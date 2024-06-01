# First create a Shuffle list
my_shuffle_list = [1,2,3,4,5]

# Now Import shuffle
from random import shuffle
shuffle(my_shuffle_list)
print(my_shuffle_list)          # check wether shuffle is working or not

# Now let's create Guess Game. First create a list
mylist = ['','o','']

# Define function which will used further
def shuffle_list(mylist):
    shuffle(mylist)
    return mylist
print(mylist)                   # First check your mylist without shuffle
print(shuffle_list(mylist))     # Now check that function for shuffle worning or not

# Now create function for user to take input as guess number
def user_guess():
    guess = ''
    while guess not in ['0','1','2']:
        guess = input("Pick a number : 0, 1 or 2 : ")
        return int(guess)
print(user_guess())

def check_guess(mylist,guess):
    if mylist[guess] == 'o':
        print('Correct Guess')
    else:
        print('Wrong Better luck next Time')

# Initial list
mylist = ['','o','']
#shuffle list
mixedup_list = shuffle_list(mylist)
# Get user guess
guess = user_guess()
check_guess(mixedup_list,guess)
