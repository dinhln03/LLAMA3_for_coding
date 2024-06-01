import numpy as np  # import numpy

with open("data/day4.txt") as f:
    drawing_numbers = f.readline()
    board_lst = []
    board_line = []
    counter = 0

    for line in f:
        if line != '\n':
            board_line.append(line.strip())  
        if len(board_line) == 5:
            board_lst.append(board_line)
            board_line = []
            
drawing_numbers = drawing_numbers.strip().split(',')


def create_board(board_lst):
    board_array = []
    for item in board_lst:
        board = [x for x in item.split(' ') if x.strip() != '']
        board_array.append(board)
    board_array = np.array(board_array)
    board_array = board_array.astype(float)
    return board_array

def check_winning(board_lst, number_lst):
    winning_condition = {
        'Answer': 0,
        'counter': 625
    }
    for item in board_lst:
        board = create_board(item)
        counter=0
        for number in number_lst:
            number = float(number) 
            counter += 1
            if number in board:
                result = np.where(board == number)
                board[int(result[0])][int(result[1])] = np.nan
            if np.all(np.isnan(board), axis=1).any() or np.all(np.isnan(board), axis=0).any():
                if counter < winning_condition['counter']:
                    winning_condition['counter'] = counter
                    winning_condition['Answer'] = number * np.nansum(board)
                    print('The Answer is:', winning_condition)
                    
                
            
check_winning(board_lst, drawing_numbers)