from option import *
import tkinter as tk


class Block:
    """ Block class for each block  of the map """

    def __init__(self, x, y, char):
        self.x = x
        self.y = y
        self.char = char
        self.blockType = MAP_CHARS[char]
        self.texture = BLOCK_TEXTURE[self.blockType]
        self.collision = self.blockType in COLLISION_BLOCK
        self.firstDraw = True  # If the first time of drawing block

    def drawBlock(self, canvas):
        if self.firstDraw:  # If their is the first draw of the block
            self.display = []
            for texture in self.texture:
                if texture[0] == "rect":
                    self.display.append(
                        canvas.create_rectangle(
                            (self.x + texture[1]) * BLOCK_SIZE,
                            (self.y + texture[2]) * BLOCK_SIZE,
                            (self.x + texture[3]) * BLOCK_SIZE,
                            (self.y + texture[4]) * BLOCK_SIZE,
                            fill=texture[5],
                            width=0,
                        )
                    )
                elif texture[0] == "oval":
                    self.display.append(
                        canvas.create_oval(
                            (self.x + texture[1]) * BLOCK_SIZE,
                            (self.y + texture[2]) * BLOCK_SIZE,
                            (self.x + texture[3]) * BLOCK_SIZE,
                            (self.y + texture[4]) * BLOCK_SIZE,
                            fill=texture[5],
                            width=0,
                        )
                    )
                elif texture[0] == "polygone":
                    self.display.append(
                        canvas.create_polygon(
                            [
                                ((self.x + x) * BLOCK_SIZE, (self.y + y) * BLOCK_SIZE)
                                for x, y in texture[1]
                            ],
                            fill=texture[2],
                            width=0,
                        )
                    )
            self.firstDraw = False
            self.texture2 = self.texture
        elif self.texture == self.texture2:
            pass
        else:
            for texture in self.texture:
                canvas.delete(texture)
            self.firstDraw = True
            self.drawBlock(canvas)
            self.texture2 = self.texture


class MapParser:
    def __init__(self, source):
        self.coin = 0  # Number of coin in the map
        self.lines = source.split("\n")  # List of all the lines
        self._grid = []  # 2D array who contain the block

        # Filter the void line
        for line in range(len(self.lines) - 1):
            if self.lines[line] in ["", "\n", " ", "\t"]:
                del self.lines[line]

        # Making the map in self._grid
        for y, line in enumerate(self.lines):
            self._grid.append([])
            for x, char in enumerate(line):
                self._grid[y].append(Block(x, y, char))
                if self._grid[y][x].blockType == "playerspawn":
                    self.spawn = (x, y)
                elif self._grid[y][x].blockType == "coin":
                    self.coin += 1

        del self.lines

    def draw(self, canvas):
        for y in range(len(self._grid)):
            for x in range(len(self._grid[y])):
                self._grid[y][x].drawBlock(canvas)

    def __getitem__(self, index):
        return self._grid[index]

    def __len__(self):
        return len(self._grid)


class Player:
    def __init__(self, gMap):
        self.score = 0
        self.x = gMap.spawn[0]
        self.y = gMap.spawn[1]
        self.velocity = 0  # For simulate gravity
        self.color = PLAYER_COLOR
        self.firstDraw = True

    def drawPlayer(self, canvas):
        if self.firstDraw:
            self.display = canvas.create_oval(
                self.x * BLOCK_SIZE,
                self.y * BLOCK_SIZE,
                (self.x + 1) * BLOCK_SIZE - 1,
                (self.y + 1) * BLOCK_SIZE - 1,
                fill=self.color,
            )
            self.firstDraw = False
        else:
            canvas.itemconfigure(self.display, fill=self.color)
            canvas.coords(
                self.display,
                self.x * BLOCK_SIZE,
                self.y * BLOCK_SIZE,
                (self.x + 1) * BLOCK_SIZE - 1,
                (self.y + 1) * BLOCK_SIZE - 1,
            )
