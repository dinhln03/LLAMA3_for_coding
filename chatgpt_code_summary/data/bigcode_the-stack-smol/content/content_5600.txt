from typing import Tuple

import yaml


class World:

    world = None
    """
    The first index is the Y coordinate, and the second index is the X coordinate
    :type world: List[List[int]]
    """

    width = None
    """
    :type width: int
    """

    height = None
    """
    :type height: int
    """

    def __init__(self, path):
        self.world = []
        self.load(path)

    def get_value(self, coordinates: Tuple[int, int]) -> int:
        return self.world[coordinates[0]][coordinates[1]]

    def load(self, path):
        self.world = []

        with open(path, 'r') as f:
            yml = yaml.load(f)

        self.height = yml['info']['height']
        self.width = yml['info']['width']

        # Create an empty world
        for y in range(0, self.height):
            self.world.append([])
            for x in range(0, self.width):
                self.world[y].append(0)

        row_index = -1
        for row in yml['data']:
            row_index += 1

            col_index = -2

            # If our row is an even row, it is a horizontal row.
            # All the streets have an offset of +1 on a horizontal row
            if row_index % 2 == 0:
                col_index += 1

            for col in row:
                col_index += 2

                self.world[row_index][col_index] = col
