# https://en.m.wikipedia.org/wiki/Box_Drawing
from random import randrange


class Board:
	# you only need to use update_board() outside this class.

	def __init__(self, size, mine_numbers):
		self.size = size
		self.default_content = " ◌ "
		self.board_data = self.create_board(self.size)
		self.mine_numbers = mine_numbers

	def create_board(self, size):
		return [[self.default_content for x in range(self.size)] for y in range(self.size)]

	def internal_board(self):
		internal_board = ""
		for x in range(len(self.board_data)):
			drawn_line = ""
			for y in range(len(self.board_data)):
				drawn_line += self.board_data[x][y]
				if y != len(self.board_data) - 1:
					drawn_line += "│"
			internal_board += drawn_line + "\n"
			if x < len(self.board_data) - 1:
				internal_board += "───┼" * \
					(len(self.board_data) - 1) + "───" + "\n"
		return internal_board

	def draw_board(self):
		internal_board = self.internal_board()
		drawn_board = ""

		# drawing the boarder around the internal board
		internal_board = internal_board.split("\n")
		drawn_board += "╔═══" + "╤═══" * (self.size - 1) + "╗" + "\n"
		for x in range(0, self.size * 2, 2):
			drawn_board += "║" + \
				internal_board[x] + "║" + f" :{int(x/2 +1)}" + "\n"
			if x != (self.size * 2) - 2:
				drawn_board += "╟" + internal_board[x + 1] + "╢" + "\n"
		drawn_board += "╚═══" + "╧═══" * (self.size - 1) + "╝" + "\n"

		for x in range(self.size):
			number = x + 1
			if number < 10:
				drawn_board += f"  {number} "
			else:
				drawn_board += f"  {number}"
		return drawn_board

	def generate_mines(self):
		mine_list = []

		while len(mine_list) < self.mine_numbers:
			x = randrange(self.size)
			y = randrange(self.size)

			if (x, y) not in mine_list:
				mine_list.append((x, y))
		return mine_list

	def generate_mine_board(self):
		mine_list = self.generate_mines()
		mine_board = ([[0 for y in range(self.size)] for x in range(self.size)])

		for mine in mine_list:
			#add a mine to the mine position, and add 1 to all adjecent spots
			x = mine[0]
			y = mine[1]

			mine_board[y][x] = "◉"  # negative = mine

			for x_ in range(x - 1, x + 2):
				for y_ in range(y - 1, y + 2):
					if 0 <= x_ < self.size and 0 <= y_ < self.size and mine_board[y_][x_] != "◉":
						mine_board[y_][x_] += 1
		return mine_board

	def find_valid_starting_mine_board(self, x, y):
		#making shure the first x,y input fits
		mine_board_candidate = []
		while True:
			mine_board_candidate = self.generate_mine_board()
			if mine_board_candidate[y - 1][x - 1] == 0:
				print(mine_board_candidate)
				break
		self.mine_board = mine_board_candidate

	def flood_fill(self, x, y):
		if self.board_data[y][x] == " ◌ " and self.mine_board[y][x] == "◉":
			raise Exception("the flood fill algo hit a mine, but it shouldn't because it will stop when it hits a number.")

		elif self.board_data[y][x] == " ◌ " and self.mine_board[y][x] > 0:
			self.board_data[y][x] = " " + str(self.mine_board[y][x]) + " "

		elif self.board_data[y][x] == " ◌ " and self.mine_board[y][x] == 0:
			self.board_data[y][x] = "   "

			for x_ in range(x - 1, x + 2):
				for y_ in range(y - 1, y + 2):
					if 0 <= x_ < self.size and 0 <= y_ < self.size:
						self.flood_fill(x_, y_)

	def reveal_board(self):
		for x in range(self.size):
			for y in range(self.size):
				if self.board_data[y][x] in [" ◌ ", " ▶ "]:
					if self.mine_board[y][x] == "◉":
						self.board_data[y][x] = " ◉ "
					elif self.mine_board[y][x] == 0:
						self.board_data[y][x] = "   "
					elif self.mine_board[y][x] > 0:
						self.board_data[y][x] = " " + str(self.mine_board[y][x]) + " "
		return self.draw_board()

	def check_winning(self):
		flag = True
		for x in range(self.size):
			for y in range(self.size):
				if self.board_data[y][x] == " ◌ " and self.mine_board[y][x] != "◉":
					flag = False
		return flag

	def update_board(self, position, flag=False):
		"""Takes position [x,y] as input
			returns a updated board as a string
		"""

		x = position[0] - 1
		y = position[1] - 1

		if flag == True:
			if self.board_data[y][x] == " ◌ ":
				self.board_data[y][x] = " ▶ "
			elif self.board_data[y][x] == " ▶ ":
				self.board_data[y][x] = " ◌ "
			return self.draw_board()

		if self.mine_board[y][x] == "◉":
			self.board_data[y][x] = " ◉ "
			return False

		elif isinstance(self.mine_board[y][x], int) and self.mine_board[y][x] > 0:
			self.board_data[y][x] = " " + str(self.mine_board[y][x]) + " "

		else:
			self.flood_fill(x, y)

		return self.draw_board()
