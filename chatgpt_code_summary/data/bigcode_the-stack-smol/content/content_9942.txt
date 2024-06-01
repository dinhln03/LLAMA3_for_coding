import numpy as np
import pytest

from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, pretty_print_board, initialize_game_state, \
    string_to_board, apply_player_action, connected_four, check_connect_topleft_bottomright


def test_initialize_game_state():

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

def test_output_pretty_print_board():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    ret = pretty_print_board(initialBoard)
    assert ret != ''

def test_empty_pretty_print_board():

    initialBoard = np.ndarray(shape=(7, 6), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    ret = pretty_print_board(initialBoard)
    assert ret == '\n|==============|\n' \
                    '|              |\n' \
                    '|              |\n' \
                    '|              |\n' \
                    '|              |\n' \
                    '|              |\n' \
                    '|              |\n' \
                    '|==============|\n' \
                    '|0 1 2 3 4 5 6 |'

def test_player1_pretty_print_board():

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(PLAYER1)

    ret = pretty_print_board(initialBoard)
    assert ret == '\n|==============|\n' \
                    '|X X X X X X X |\n' \
                    '|X X X X X X X |\n' \
                    '|X X X X X X X |\n' \
                    '|X X X X X X X |\n' \
                    '|X X X X X X X |\n' \
                    '|X X X X X X X |\n' \
                    '|==============|\n' \
                    '|0 1 2 3 4 5 6 |'

def test_player2_pretty_print_board():

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(PLAYER2)

    ret = pretty_print_board(initialBoard)
    assert ret == '\n|==============|\n' \
                    '|O O O O O O O |\n' \
                    '|O O O O O O O |\n' \
                    '|O O O O O O O |\n' \
                    '|O O O O O O O |\n' \
                    '|O O O O O O O |\n' \
                    '|O O O O O O O |\n' \
                    '|==============|\n' \
                    '|0 1 2 3 4 5 6 |'

def test_precision_pretty_print_board():

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[0,0] = PLAYER1

    ret = pretty_print_board(initialBoard)
    assert ret == '\n|==============|\n' \
                  '|              |\n' \
                  '|              |\n' \
                  '|              |\n' \
                  '|              |\n' \
                  '|              |\n' \
                  '|X             |\n' \
                  '|==============|\n' \
                  '|0 1 2 3 4 5 6 |'


def test_dimensions_pretty_print_board():

    initialBoard = np.ndarray(shape=(7, 6), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    with pytest.raises(ValueError):
        ret = pretty_print_board(initialBoard)

def test_invalid_piece_pretty_print_board():
    initialBoard = np.ndarray(shape=(7, 6), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)
    initialBoard[0, 0] = 60

    with pytest.raises(ValueError):
        ret = pretty_print_board(initialBoard)


def test_string_to_board():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    print = '\n|==============|\n' \
           '|              |\n' \
           '|              |\n' \
           '|              |\n' \
           '|              |\n' \
           '|              |\n' \
           '|              |\n' \
           '|==============|\n' \
           '|0 1 2 3 4 5 6 |'

    ret = string_to_board(print)
    assert ret.all() == initialBoard.all()


def test_drop_piece():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    ret = apply_player_action(initialBoard, 0, PLAYER1)
    drop_board = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    drop_board.fill(NO_PLAYER)
    drop_board[0,5] = 1

    print(ret)
    assert ret.all() == drop_board.all()

def test_connected_four_false():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    assert connected_four(initialBoard, PLAYER1, 5) == False

def test_connected_four_true():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(PLAYER1)

    assert connected_four(initialBoard, PLAYER1, 5) == True

def test_connected_four_row_true():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = 1
    initialBoard[5, 1] = 1
    initialBoard[5, 2] = 1
    initialBoard[5, 3] = 1

    print(initialBoard)
    assert connected_four(initialBoard, PLAYER1, 0) == True

def test_connected_four_row_false():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = 1
    initialBoard[5, 1] = 1

    initialBoard[5, 3] = 1

    print(initialBoard)

    with pytest.raises(AssertionError):
        assert connected_four(initialBoard, PLAYER1, 0) == True


def test_connected_four_BL_TR_true():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = 1
    initialBoard[4, 1] = 1
    initialBoard[3, 2] = 1
    initialBoard[2, 3] = 1

    print(initialBoard)
    assert connected_four(initialBoard, PLAYER1, 0) == True

def test_connected_four_BL_TR_false():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = 1
    initialBoard[4, 1] = 1
    initialBoard[3, 2] = 1


    print(initialBoard)

    with pytest.raises(AssertionError):
        assert connected_four(initialBoard, PLAYER1, 0) == True



def test_connected_four_BR_TL_true():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 5] = 1
    initialBoard[4, 4] = 1
    initialBoard[3, 3] = 1
    initialBoard[2, 2] = 1

    print(initialBoard)
    assert connected_four(initialBoard, PLAYER1, 5) == True

def test_connected_four_BR_TL_false():
    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 5] = 1
    initialBoard[4, 4] = 1
    initialBoard[2, 2] = 1


    assert connected_four(initialBoard, PLAYER1, 5) == False


def test_diagonal_check_BLTR_true():
    from agents.common import diagonal_check

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = 1
    initialBoard[4, 1] = 1
    initialBoard[3, 2] = 1
    initialBoard[2, 3] = 1

    print(initialBoard)

    assert diagonal_check(initialBoard, PLAYER1, 0, 5, 1, -1) == True

def test_diagonal_check_TLBR_YX_true():
    from agents.common import diagonal_check

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 4] = 1
    initialBoard[4, 3] = 1
    initialBoard[3, 2] = 1
    initialBoard[2, 1] = 1

    print(initialBoard)

    assert diagonal_check(initialBoard, PLAYER1, 4, 5, -1, -1) == True

def test_TLBR_YX_true():
    from agents.common import check_connect_topleft_bottomright

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 4] = 1
    initialBoard[4, 3] = 1
    initialBoard[3, 2] = 1
    initialBoard[2, 1] = 1

    print(initialBoard)

    assert check_connect_topleft_bottomright(initialBoard, PLAYER1, 4, 0) == True

def test_diagonal_check_TLBR_XY_true():
    from agents.common import diagonal_check

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 6] = 1
    initialBoard[4, 5] = 1
    initialBoard[3, 4] = 1
    initialBoard[2, 3] = 1

    print(initialBoard)

    assert diagonal_check(initialBoard, PLAYER1, 6, 5, -1, -1) == True

def test_TLBR_XY_true():
    from agents.common import check_connect_topleft_bottomright

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 6] = 1
    initialBoard[4, 5] = 1
    initialBoard[3, 4] = 1
    initialBoard[2, 3] = 1

    print(initialBoard)

    assert check_connect_topleft_bottomright(initialBoard, PLAYER1, 6, 0)

def test_BL_TR_true():
    from agents.common import check_connect_topright_bottomleft

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = 1
    initialBoard[4, 1] = 1
    initialBoard[3, 2] = 1
    initialBoard[2, 3] = 1

    print(initialBoard)

    assert check_connect_topright_bottomleft(initialBoard, PLAYER1, 0, 0) == True

def test_BL_TR_false():
    from agents.common import check_connect_topright_bottomleft

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = 1
    initialBoard[4, 1] = 1
    initialBoard[3, 2] = 1

    print(initialBoard)


    assert check_connect_topright_bottomleft(initialBoard, PLAYER1, 0, 0) == False


def test_end_state_win():
    from agents.common import check_end_state, GameState

    from agents.common import check_connect_topright_bottomleft

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    initialBoard[5, 0] = 1
    initialBoard[4, 1] = 1
    initialBoard[3, 2] = 1
    initialBoard[2, 3] = 1

    assert check_end_state(initialBoard, PLAYER1, 0) == GameState.IS_WIN

def test_end_state_still_playing():
    from agents.common import check_end_state, GameState

    initialBoard = np.ndarray(shape=(6, 7), dtype=BoardPiece)
    initialBoard.fill(NO_PLAYER)

    assert check_end_state(initialBoard, PLAYER1, 0) == GameState.STILL_PLAYING


def test_end_state_draw():
    from agents.common import check_end_state, GameState

    x = np.zeros((6, 7), dtype=int)
    x.fill(2)
    x[1::2, ::2] = 1
    x[::2, 1::2] = 1
    print(x)

    assert check_end_state(x, PLAYER1, 1) == GameState.IS_DRAW


def test_diagonal_neg():
    #str = "|==============|\n|O             |\n|X O           |\n|O X O         |\n|X X O O X     |\n|O X O X X     |\n|X O X X O     |\n|==============|\n|0 1 2 3 4 5 6 |"

    #board = string_to_board(str)

    board =  np.zeros((6, 7), dtype=int)
    board[0, 0] = PLAYER2
    board[1, 1] = PLAYER2
    board[2, 2] = PLAYER2
    board[3, 3] = PLAYER2

    assert check_connect_topleft_bottomright(board, PLAYER2, 2, 3) == True







