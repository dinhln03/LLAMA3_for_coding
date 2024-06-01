import sys, os, json

version = (3,7)
assert sys.version_info >= version, "This script requires at least Python {0}.{1}".format(version[0],version[1])




# Game loop functions
def render(game,current):
    ''' Displays the current room '''

    print('You are in the ' + game['rooms'][current]['name'])
    print(game['rooms'][current]['desc'])

def getInput():
    ''' Asks the user for input and returns a stripped, uppercase version of what they typed '''

    response = input('What would you like to do? ').strip().upper()
    return response


def update(response,game,current):
    ''' Process the input and update the state of the world '''
    for e in game['rooms'][current]['exits']:
        if response == e['verb']:
            current = e['target']
    return current



def main():

    game = {}
    with open('house.json') as json_file:
        game = json.load(json_file)

    current = 'START'

    quit = False

    while not quit:

        render(game,current)
        response = getInput()
        current = update(response,game,current)

        if response == 'QUIT':
            quit = True




if __name__ == '__main__':
	main()