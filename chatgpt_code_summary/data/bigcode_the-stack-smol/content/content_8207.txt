import pygame  # Tested with pygame v1.9.6
from UIControls import Button
from constants import *
import numpy as np
import random
import time
import os
from nodes import bfs_node
import sys
import threading


###############################################
# Globals
###############################################

initial_cell_row = 0
initial_cell_col = 0
initial_cell_dragging = False

terminal_cell_row = ROWS - 1
terminal_cell_col = COLS - 1
terminal_cell_dragging = False

grid = np.ndarray((COLS, ROWS), np.int8)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

clear_button = Button((BUTTON_WIDTH * 0),
                      BUTTON_STRIP_TOP,
                      BUTTON_WIDTH,
                      BUTTON_STRIP_HEIGHT,
                      CLEAR_BUTTON_LABEL)
create_button = Button((BUTTON_WIDTH * 1),
                       BUTTON_STRIP_TOP,
                       BUTTON_WIDTH,
                       BUTTON_STRIP_HEIGHT,
                       CREATE_BUTTON_LABEL)
dfs_button = Button((BUTTON_WIDTH * 2),
                    BUTTON_STRIP_TOP,
                    BUTTON_WIDTH,
                    BUTTON_STRIP_HEIGHT,
                    DFS_BUTTON_LABEL)
bfs_button = Button((BUTTON_WIDTH * 3),
                    BUTTON_STRIP_TOP,
                    BUTTON_WIDTH,
                    BUTTON_STRIP_HEIGHT,
                    BFS_BUTTON_LABEL)
quit_button = Button((BUTTON_WIDTH * 4),
                     BUTTON_STRIP_TOP,
                     BUTTON_WIDTH,
                     BUTTON_STRIP_HEIGHT,
                     QUIT_BUTTON_LABEL)

processing = False


###############################################
# initialise()
###############################################

def initialise():
    global processing
    processing = True
    # Set all cells to EMPTY by default
    for col in range(COLS):
        for row in range(ROWS):
            grid[col, row] = EMPTY

    # Set the Initial and Terminal cells
    grid[initial_cell_col, initial_cell_row] = INITIAL
    grid[terminal_cell_col, terminal_cell_row] = TERMINAL
    # print(grid)

    processing = False


###############################################
# create_ui()
###############################################

def create_ui():
    screen.fill(BLACK)

    clear_button.draw(screen)
    create_button.draw(screen)
    dfs_button.draw(screen)
    bfs_button.draw(screen)
    quit_button.draw(screen)

    draw_grid()


###############################################
# draw_grid()
###############################################

def draw_grid():
    for col in range(COLS):
        for row in range(ROWS):
            # Only set the Initial cell if we are NOT dragging
            if (grid[col, row] == INITIAL and not initial_cell_dragging):
                draw_cell(INITIAL_CELL_COLOR, col, row)
            # Only set the Terminal cell if we are NOT dragging
            elif (grid[col, row] == TERMINAL and not terminal_cell_dragging):
                draw_cell(TERMINAL_CELL_COLOR, col, row)
            elif (grid[col, row] == WALL):
                draw_cell(WALL_CELL_COLOR, col, row)
            elif (grid[col, row] == VISITED):
                draw_cell(VISITED_CELL_COLOR, col, row)
            elif (grid[col, row] == PATH):
                draw_cell(PATH_CELL_COLOR, col, row)
            else:  # (grid[col, row] == EMPTY)
                draw_cell(EMPTY_CELL_COLOR, col, row)

    if (initial_cell_dragging):
        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        cell_col = int(mouse_x / CELL_WIDTH)
        cell_row = int(mouse_y / CELL_HEIGHT)
        # Check the current mouse-pointer for the dragging
        # motion is actually on the board
        if (valid_cell(cell_col, cell_row)):
            draw_cell(INITIAL_CELL_COLOR,
                      cell_col,
                      cell_row)
    elif (terminal_cell_dragging):
        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        cell_col = int(mouse_x / CELL_WIDTH)
        cell_row = int(mouse_y / CELL_HEIGHT)
        # Check the current mouse-pointer for the dragging motion
        # is actually on the board
        if (valid_cell(cell_col, cell_row)):
            draw_cell(TERMINAL_CELL_COLOR, cell_col, cell_row)


###############################################
# game_loop()
###############################################

def game_loop():
    game_exit = False
    clock = pygame.time.Clock()

    global initial_cell_row
    global initial_cell_col
    global initial_cell_dragging

    global terminal_cell_row
    global terminal_cell_col
    global terminal_cell_dragging

    while not game_exit:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) and (not processing):
                game_exit = True
            elif (event.type == pygame.MOUSEBUTTONDOWN) and (not processing):
                (mouse_x, mouse_y) = pygame.mouse.get_pos()
                cell_col = int(mouse_x / CELL_WIDTH)
                cell_row = int(mouse_y / CELL_HEIGHT)
                if (valid_cell(cell_col, cell_row)):
                    if (grid[cell_col, cell_row] == INITIAL):
                        # Set the flag for dragging the Initial cell
                        initial_cell_dragging = True
                    elif (grid[cell_col, cell_row] == TERMINAL):
                        # Set the flag for dragging the Terminal cell
                        terminal_cell_dragging = True
                    elif (not (initial_cell_dragging
                               or terminal_cell_dragging)):
                        # Otherwise, if we have clicked with mouse and
                        # we are not dragging anything, toggle
                        # the current cell between EMPTY and WALL
                        if (grid[cell_col, cell_row] == WALL):
                            grid[cell_col, cell_row] = EMPTY
                        elif (grid[cell_col, cell_row] == EMPTY):
                            grid[cell_col, cell_row] = WALL
            elif (event.type == pygame.MOUSEBUTTONUP) and (not processing):
                if clear_button.is_over(mouse_x, mouse_y):
                    thread = threading.Thread(target=initialise,
                                              args=())
                    thread.start()
                elif create_button.is_over(mouse_x, mouse_y):
                    thread = threading.Thread(target=create_maze,
                                              args=())
                    thread.start()
                elif dfs_button.is_over(mouse_x, mouse_y):
                    thread = threading.Thread(target=depth_first_search,
                                              args=())
                    thread.start()
                elif bfs_button.is_over(mouse_x, mouse_y):
                    thread = threading.Thread(target=breadth_first_search,
                                              args=())
                    thread.start()
                elif quit_button.is_over(mouse_x, mouse_y):
                    game_exit = True
                elif initial_cell_dragging:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()
                    cell_col = int(mouse_x / CELL_WIDTH)
                    cell_row = int(mouse_y / CELL_HEIGHT)
                    # Make sure we have not dragged the
                    # Initial cell off the screen
                    if (valid_cell(cell_col, cell_row)):
                        # Also make sure we aren't trying to drag Initial
                        # cell on top of Terminal cell
                        if (not((cell_col == terminal_cell_col) and
                                (cell_row == terminal_cell_row))):
                            grid[initial_cell_col, initial_cell_row] = EMPTY
                            initial_cell_col = cell_col
                            initial_cell_row = cell_row
                            grid[initial_cell_col, initial_cell_row] = INITIAL
                    # Whatever happens, cancel the dragging flag
                    initial_cell_dragging = False
                elif terminal_cell_dragging:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()
                    cell_col = int(mouse_x / CELL_WIDTH)
                    cell_row = int(mouse_y / CELL_HEIGHT)
                    # Make sure we have not dragged the
                    # Terminal cell off the screen
                    if (valid_cell(cell_col, cell_row)):
                        # Also make sure we aren't trying to drag Terminal
                        # cell on top of Initial cell
                        if (not((cell_col == initial_cell_col) and
                                (cell_row == initial_cell_row))):
                            grid[terminal_cell_col, terminal_cell_row] = EMPTY
                            terminal_cell_col = cell_col
                            terminal_cell_row = cell_row
                            grid[terminal_cell_col, terminal_cell_row] = TERMINAL
                    # Whatever happens, cancel the dragging flag
                    terminal_cell_dragging = False

        draw_grid()
        pygame.display.update()
        clock.tick(CLOCK_TICK)
    pygame.quit()


###############################################
# create_maze()
###############################################

def create_maze():

    ###############################################
    # make_holes()
    ###############################################

    def make_holes(col1, row1, col2, row2, vertical, horizontal):
        # print(f"\tmake_holes({col1},
        #                      {row1},
        #                      {col2},
        #                      {row2},
        #                      {vertical},
        #                      {horizontal})")

        all_lists = []

        list = []
        for row in range(row1, horizontal):
            if (has_horizontal_empty(vertical, row)):
                list.append((vertical, row))
        if (len(list) > 0):
            all_lists.append(list)

        list = []
        for row in range(horizontal + 1, row2):
            if (has_horizontal_empty(vertical, row)):
                list.append((vertical, row))
        if (len(list) > 0):
            all_lists.append(list)

        list = []
        for col in range(col1, vertical):
            if (has_vertical_empty(col, horizontal)):
                list.append((col, horizontal))
        if (len(list) > 0):
            all_lists.append(list)

        list = []
        for col in range(vertical + 1, col2):
            if (has_vertical_empty(col, horizontal)):
                list.append((col, horizontal))
        if (len(list) > 0):
            all_lists.append(list)

        if (len(all_lists) == 4):
            item_index_to_remove = random.randint(0, 3)
            del (all_lists[item_index_to_remove])

        for sub_list in all_lists:
            (hole_col, hole_row) = sub_list[
                random.randint(0, len(sub_list) - 1)]
            draw_cell(EMPTY_CELL_COLOR, hole_col, hole_row)
            grid[hole_col, hole_row] = EMPTY

    ###############################################
    # divide()
    ###############################################

    def divide(col1, row1, col2, row2):
        # print(f"divide({col1}, {row1}, {col2}, {row2})")
        vertical = col2
        if ((col2 - col1) > 2):
            vertical = int(((col2 - col1) / 2) + col1)

            for row in range(row1, row2):
                draw_cell(WALL_CELL_COLOR, vertical, row)
                grid[vertical, row] = WALL

        horizontal = row2
        if ((row2 - row1) > 2):
            horizontal = int(((row2 - row1) / 2) + row1)

            for col in range(col1, col2):
                draw_cell(WALL_CELL_COLOR, col, horizontal)
                grid[col, horizontal] = WALL

        # top-left
        new_col1 = col1
        new_row1 = row1
        new_col2 = vertical
        new_row2 = horizontal
        if (((new_col2 - new_col1) > 2) or ((new_row2 - new_row1) > 2)):
            (new_vertical, new_horizontal) = divide(new_col1,
                                                    new_row1,
                                                    new_col2,
                                                    new_row2)
            make_holes(new_col1,
                       new_row1,
                       new_col2,
                       new_row2,
                       new_vertical,
                       new_horizontal)

        # top-right
        new_col1 = vertical + 1
        new_row1 = row1
        new_col2 = col2
        new_row2 = horizontal
        if (((new_col2 - new_col1) > 2) or ((new_row2 - new_row1) > 2)):
            (new_vertical, new_horizontal) = divide(new_col1,
                                                    new_row1,
                                                    new_col2,
                                                    new_row2)
            make_holes(new_col1,
                       new_row1,
                       new_col2,
                       new_row2,
                       new_vertical,
                       new_horizontal)

        # bottom-left
        new_col1 = col1
        new_row1 = horizontal + 1
        new_col2 = vertical
        new_row2 = row2
        if (((new_col2 - new_col1) > 2) or ((new_row2 - new_row1) > 2)):
            (new_vertical, new_horizontal) = divide(new_col1,
                                                    new_row1,
                                                    new_col2,
                                                    new_row2)
            make_holes(new_col1,
                       new_row1,
                       new_col2,
                       new_row2,
                       new_vertical,
                       new_horizontal)

        # bottom-right
        new_col1 = vertical + 1
        new_row1 = horizontal + 1
        new_col2 = col2
        new_row2 = row2
        if (((new_col2 - new_col1) > 2) or ((new_row2 - new_row1) > 2)):
            (new_vertical, new_horizontal) = divide(new_col1,
                                                    new_row1,
                                                    new_col2,
                                                    new_row2)
            make_holes(new_col1,
                       new_row1,
                       new_col2,
                       new_row2,
                       new_vertical,
                       new_horizontal)

        time.sleep(SMALL_SLEEP)
        pygame.display.update()

        return (vertical, horizontal)

    global processing
    processing = True
    initialise()
    draw_grid()

    (new_vertical, new_horizontal) = divide(0, 0, COLS, ROWS)
    make_holes(0, 0, COLS, ROWS, new_vertical, new_horizontal)
    grid[initial_cell_col, initial_cell_row] = INITIAL
    grid[terminal_cell_col, terminal_cell_row] = TERMINAL

    processing = False


###############################################
# has_horizontal_neighbours()
###############################################

def has_horizontal_neighbours(col, row, cell_types):
    left_col = col - 1
    right_col = col + 1
    if (left_col >= 0) and (right_col < COLS):
        return ((grid[left_col, row] in cell_types) and
                (grid[right_col, row] in cell_types))

    return False


###############################################
# has_vertical_neighbours()
###############################################

def has_vertical_neighbours(col, row, cell_types):
    above_row = row - 1
    below_row = row + 1
    if (above_row >= 0) and (below_row < ROWS):
        return ((grid[col, above_row] in cell_types) and
                (grid[col, below_row] in cell_types))

    return False


###############################################
# has_horizontal_empty()
###############################################

def has_horizontal_empty(col, row):
    return has_horizontal_neighbours(col, row, [EMPTY, INITIAL, TERMINAL])


###############################################
# has_vertical_empty()
###############################################

def has_vertical_empty(col, row):
    return has_vertical_neighbours(col, row, [EMPTY, INITIAL, TERMINAL])


###############################################
# reset_maze()
###############################################

def reset_maze():
    """Resets any cells that are VISITED or PATH to EMPTY again, so that we
    can commence a search on a potentially partially completed board"""
    for col in range(COLS):
        for row in range(ROWS):
            grid[col, row] = EMPTY if (grid[col, row] in [VISITED, PATH]) else grid[col, row]


###############################################
# valid_cell()
###############################################

def valid_cell(col, row):
    return ((col >= 0) and (row >= 0) and (col < COLS) and (row < ROWS))


###############################################
# depth_first_search()
###############################################

def depth_first_search():

    ###############################################
    # check()
    ###############################################

    def check(col, row):
        if (valid_cell(col, row)):
            if (search(col, row)):
                return True
        return False

    ###############################################
    # search()
    ###############################################

    def search(col, row):
        print(f"search({col}, {row})")
        pygame.display.update()
        # time.sleep(SMALL_SLEEP)

        if (grid[col, row] == TERMINAL):
            return True
        if (grid[col, row] in [WALL, VISITED, PATH]):
            return False

        if (grid[col, row] != INITIAL):
            grid[col, row] = PATH
            draw_cell(PATH_CELL_COLOR, col, row)

            if (check(col - 1, row)):
                return True

            if (check(col + 1, row)):
                return True

            if (check(col, row - 1)):
                return True

            if (check(col, row + 1)):
                return True

            grid[col, row] = VISITED
            draw_cell(VISITED_CELL_COLOR, col, row)
            return False

    global processing
    processing = True
    reset_maze()
    draw_grid()

    if (check(initial_cell_col - 1, initial_cell_row)):
        processing = False
        return

    if (check(initial_cell_col + 1, initial_cell_row)):
        processing = False
        return

    if (check(initial_cell_col, initial_cell_row - 1)):
        processing = False
        return

    if (check(initial_cell_col, initial_cell_row + 1)):
        processing = False
        return
    processing = False


###############################################
# breadth_first_search()
###############################################

def breadth_first_search():

    ###############################################
    # search()
    ###############################################

    def search(nodes):

        ###############################################
        # check()
        ###############################################

        def check(next_col, next_row, sub_nodes):
            if (valid_cell(next_col, next_row)):
                if (grid[next_col, next_row] == TERMINAL):
                    backtrack_node = node

                    while (backtrack_node is not None):
                        if (backtrack_node.get_parent() is not None):
                            grid[backtrack_node.get_col(),
                                 backtrack_node.get_row()] = PATH
                            draw_cell(PATH_CELL_COLOR,
                                      backtrack_node.get_col(),
                                      backtrack_node.get_row())
                            pygame.display.update()
                        backtrack_node = backtrack_node.get_parent()

                    return True
                elif ((grid[next_col, next_row] != WALL) and
                      (grid[next_col, next_row] != VISITED) and
                      (grid[next_col, next_row] != INITIAL)):
                    grid[next_col, next_row] = VISITED
                    draw_cell(VISITED_CELL_COLOR, next_col, next_row)
                    pygame.display.update()
                    child_node = bfs_node(next_col, next_row, node)
                    sub_nodes.append(child_node)
            return False

        pygame.display.update()
        time.sleep(SMALL_SLEEP)

        sub_nodes = []

        for node in nodes:
            # print(f"\tNode at ({node.get_col()}, {node.get_row()})")
            if (check(node.get_col() - 1, node.get_row(), sub_nodes)):
                return

            if (check(node.get_col() + 1, node.get_row(), sub_nodes)):
                return

            if (check(node.get_col(), node.get_row() + 1, sub_nodes)):
                return

            if (check(node.get_col(), node.get_row() - 1, sub_nodes)):
                return

        if(len(sub_nodes) > 0):
            return search(sub_nodes)
        else:
            return False

    global processing
    processing = True
    reset_maze()
    draw_grid()

    nodes = []
    nodes.append(bfs_node(initial_cell_col, initial_cell_row, None))
    search(nodes)

    processing = False


###############################################
# draw_cell()
###############################################

def draw_cell(color, col, row):
    pygame.draw.rect(screen,
                     color,
                     (col * CELL_WIDTH,
                      row * CELL_HEIGHT,
                      CELL_WIDTH,
                      CELL_HEIGHT),
                     0)


###############################################
# main()
###############################################

def main():
    # Increase stack size (depth-first-search is stack intensive)
    sys.setrecursionlimit(10 ** 6)

    pygame.init()
    pygame.display.set_caption("Maze")

    initialise()
    create_ui()

    game_loop()


###############################################
# Startup
###############################################

if __name__ == "__main__":
    main()
