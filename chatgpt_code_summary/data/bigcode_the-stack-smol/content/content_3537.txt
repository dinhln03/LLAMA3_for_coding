import os
import sys

filename = __file__[:-5] + '-input'

with open(filename) as f:
    board = list(map(lambda s: list(map(int, list(s))), f.read().splitlines()))

max_row = len(board)
max_col = len(board[0])

def get_neighbors(row, col):
    n = []

    if(row > 0):
        n.append((row-1,col))
    if(row+1 < max_row):
        n.append((row+1,col))

    if(col > 0):
        n.append((row,col-1))
    if(col+1 < max_col):
        n.append((row,col+1))


    return n

low_points = []
basin_size = {}

for i, row in enumerate(board):
    for j, val in enumerate(row):
        neighbors = [board[r][c] for r,c in get_neighbors(i,j)]

        if all([val < elem for elem in neighbors ]):
            low_points.append((i,j))

for r,c in low_points:
    visited = []
    to_explore = [(r,c)]
    while len(to_explore) > 0:
        visited.append(to_explore[0])
        cur_r, cur_c = to_explore.pop(0)
        to_explore.extend([(r,c) for r,c in get_neighbors(cur_r, cur_c) if board[r][c] < 9 and (r,c) not in visited and (r,c) not in to_explore])

    basin_size[(r,c)] = len(visited)


largest_basins = sorted(basin_size, key=basin_size.get, reverse=True)[:3]

print(basin_size[largest_basins[0]]*basin_size[largest_basins[1]]*basin_size[largest_basins[2]])