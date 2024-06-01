import config as c
import random as r

def print_map(map_grid):
    print("= " * (len(map_grid) + 2))
    for row in map_grid:
        print("||", end='')
        print(*row, sep=" ", end='')
        print("||")
    print("= " * (len(map_grid) + 2))

# Builds map with all of one type of tile
# Should be WALL or FLOOR
def init_empty_map(dimension, default_tile):
    map_grid = []
    for i in range(dimension):
        map_grid.append([default_tile] * dimension)
    return map_grid

# def build_ruins(dimension, p_mod):
#     map_grid = init_empty_map(dimension, c.FLOOR)
#     build_dungeon_walls(map_grid, p_mod)
#     return map_grid

# Randomly populate wall tiles across an empty dungeon floor
def build_dungeon_walls(map_grid, p_mod):
    for y in range(0, len(map_grid)):
        for x in range(0, len(map_grid)):
            # Determine if wall tile will be populated
            if r.randint(0,100) / 100 < p_mod:
                map_grid[y][x] = c.WALL

def build_wall_clusters(map_grid, p_mod):
    for y in range(0, len(map_grid) - 1):
        for x in range(0, len(map_grid) - 1):
            # Determine if a few tiles will be populated
            if r.randint(0,100) / 100 < p_mod:
                build_cluster(map_grid, y, x)

# Populate a cluster of 2-3 tiles on the map
# Does not check for overlap of existing wall tiles
def build_cluster(map_grid, row, column):
    itr = r.randint(1,3)
    while itr > 0:
        map_grid[row][column] = c.WALL
        next_direction = r.choice(get_valid_cardinals(map_grid, row, column, False))
        row += c.CARDINAL_VECTORS[next_direction][c.Y_INDEX]
        column += c.CARDINAL_VECTORS[next_direction][c.X_INDEX]
        itr -= 1

# Returns a subset of cardinal directions which you could move from a given tile on a map
# 'diaganol' is a flag for whether or not to consider diaganol adjacency
def get_valid_cardinals(map_grid, row, column, diaganol):
    valid_cardinals = []
    if row > 0:
        valid_cardinals.append(c.NORTH)
    if column > 0:
        valid_cardinals.append(c.WEST)
    if row < len(map_grid) - 1:
        valid_cardinals.append(c.SOUTH)
    if column < len(map_grid) - 1:
        valid_cardinals.append(c.EAST)
    if diaganol:
        if row > 0 and column > 0:
            valid_cardinals.append(c.NORTHWEST)
        if row > 0 and column < len(map_grid) - 1:
            valid_cardinals.append(c.NORTHEAST)
        if row < len(map_grid) - 1 and column > 0:
            valid_cardinals.append(c.SOUTHWEST)
        if row < len(map_grid) - 1 and column < len(map_grid) - 1:
            valid_cardinals.append(c.SOUTHEAST)
    return valid_cardinals

# Clears all tiles of a given type, which have no adjacent matching tiles
# Default clear state is a FLOOR tile
# This considers diagonal adjacency
def remove_adjacentless_tiles(map_grid, tile_type):
    for y in range(0, len(map_grid)):
        for x in range(0, len(map_grid)):
            if map_grid[y][x] == tile_type and has_adjacent_tile(map_grid, y, x) is not True:
                map_grid[y][x] = c.FLOOR

# TODO Debug
def has_adjacent_tile(map_grid, y, x):
    tile_type = map_grid[y][x]
    cardinals = get_valid_cardinals(map_grid, y, x, True)
    for cardinal in cardinals:
        y_adj = y + c.CARDINAL_VECTORS[cardinal][c.Y_INDEX]
        x_adj = x + c.CARDINAL_VECTORS[cardinal][c.X_INDEX]
    if map_grid[y_adj][x_adj] == tile_type:
        return True
    return False