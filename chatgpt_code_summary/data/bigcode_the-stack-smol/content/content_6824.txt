import random
import time

def dead_state(width, height):
    board = []
    line = []
    for i in range(width):
        for j in range(height):
            line.append(0)
        board.append(line)
        line = []
    return board

def random_state(width, height):
    state = dead_state(width, height)

    for i in range(width):
        for j in range(height):
            state[i][j] = 1 if random.random() >= 0.5 else 0
    return state

def render(state):
    term_print = ''
    for i in range(len(state[:])):
        for j in range(len(state[i][:])):
            if state[i][j] == 1:
                term_print += '#'
            else:
                term_print += ' '
        term_print += "\n"
    print(term_print)

def next_state(state):
    # check the inputs for the dead state
    # how to get the length of the row and height from a list of lists
    width = len(state[:])
    height = len(state[:][:])
    test_state = dead_state(width, height)

    for i in range(len(state[:])):
        for j in range(len(state[i][:])):
            # Alive cell
            if state[i][j] == 1:
                test_state[i][j] = alive_cell(i,j,state)
            # Dead cell
            else:
                test_state[i][j] = dead_cell(i,j,state)
    return test_state

def alive_cell(i,j,state):
    alive = 0
    width = len(state[:])
    height = len(state[:][:])
    # break is not being utilized properly
    # when the break hits it ends the innermost loop not just an iteration
    for row in range(i-1,i+2):
        for column in range(j-1,j+2):
            # print('\t\talive',row,column)
            if row < 0 or row >= height:
                # too wide
                continue
            if column < 0 or column >= width:
                # too tall
                continue

            if state[row][column] == 1:
                alive += 1
                # print('\talive',row,column)
    alive -= 1
    # print('alive', alive)
    if alive == 2 or alive == 3:
        # current cell stays alive
        return 1
    else:
        # current cell dies
        return 0

def dead_cell(i,j,state):
    alive = 0
    width = len(state[:])
    height = len(state[:][:])
    for row in range(i-1,i+2):
        for column in range(j-1,j+2):
            # print('\t\tdead',row,column)
            if row < 0 or row >= height:
                # too wide
                continue
            if column < 0 or column >= width:
                # too tall
                continue

            if state[row][column] == 1:
                alive += 1
                # print('\tdead',row,column)
    # print('dead', alive)
    if alive == 3:
        # current cell revives
        return 1
    else:
        # current cell stays dead
        return 0

def load_board_state(location):
    board = []
    x = []
    with open(location, 'r') as f:
        for line in f:
            for ch in line:
                if ch == '\n':
                    continue
                x.append(int(ch))
            board.append(x)
            x = []
    return board


if __name__ == '__main__':
    loaded_board = load_board_state('./toad.txt')
    render(loaded_board)
    flag = False
    while(True):
        time.sleep(0.5)
        if flag == False:
            next_board = next_state(loaded_board)
            render(next_board)
            flag = True
        else:
            next_board = next_state(next_board)
            render(next_board)
    # init_state = random_state(25,25)
    # render(init_state)
    # count = 0
    # while(True):
    #     # Wait for 1 second
    #     time.sleep(.5)
    #     if count == 0:
    #         next_board = next_state(init_state)
    #         render(next_board)
    #         count = 1
    #     else:
    #         next_board = next_state(next_board)
    #         render(next_board)
