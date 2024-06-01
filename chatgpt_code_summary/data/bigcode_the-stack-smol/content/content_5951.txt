#!/usr/bin/env python3

from string import ascii_uppercase
from re import fullmatch
from time import sleep
from random import Random

# Default game presets.
testing_preset = {'height': 10, 'width': 10, '5_ships': 0, '4_ships': 0, '3_ships': 0, '2_ships': 2, '1_ships': 0, 'allow_mines': True, 'allow_moves': True, 'mine_turns': 5, 'p_type': 'Player', 'player_timer': 0}
normal_mode_preset = {'height': 10, 'width': 10, '5_ships': 1, '4_ships': 1, '3_ships': 2, '2_ships': 1, '1_ships': 0, 'allow_mines': False, 'allow_moves': False, 'mine_turns': None, 'p_type': 'CPU', 'player_timer': 5}
advanced_mode_preset = {'height': 15, 'width': 15, '5_ships': 2, '4_ships': 2, '3_ships': 2, '2_ships': 1, '1_ships': 0, 'allow_mines': True, 'allow_moves': True, 'mine_turns': 5, 'p_type': 'CPU', 'player_timer': 5}

# Miscellaneous global values.
letters = ascii_uppercase

# Global user-variables.
PAD_AMOUNT = 50


class Utils(object):
    """
    Utility class used for getting input and other common functions.

    Contains many functions to save space by condensing input and custom string formatting methods into one place.
    All methods are static, and do not modify parameters in-place.
    """
    @staticmethod
    def box_string(string, min_width=-1, print_string=False):
        """
        Place a string into an ASCII box.

        The result is placed inside of a ASCII box consisting of '+' characters for the corners and '-' characters for the edges.

        Parameters
        ----------
        string : str
            String to be boxed.
        min_width : int, optional
            Specifies that the box be of a certain minimum width. Defaults to input string width.
        print_string : bool, optional
            If True, prints the string after building it. Defaults to False.

        Returns
        -------
        str
            Input string with a box around it.
        """
        # Parameters.
        split_string = string.split('\n')
        height = len(split_string)
        length = max(min_width, *[len(x) for x in split_string])

        # String builder.
        result = '+' + '-' * (length + 2) + '+\n'
        for i in range(height):
            result += '| %s |\n' % split_string[i].center(length)
        result += '+' + '-' * (length + 2) + '+'

        # Print and return result.
        if print_string:
            print(result)
        return result

    @staticmethod
    def num_input(question, *choices):
        """
        Take user input based on several different options.

        The input question will be repeated until valid input is given.
        The choices will be displayed in order with a number next to them indicating their id.
        Responses can be given as the choice id or the full choice name.

        Parameters
        ----------
        question : str
            String to be displayed as the input question. Will be boxed with Utils#box_string before printing.
        *choices : *str
            Options for the user to choose from.

        Returns
        -------
        int
            Number of the answer choice, corresponding to the index of the choice in *choices.
        """
        error = ''
        while True:
            # Print question and ask for input.
            Utils.box_string((error + '\n' + question).strip(), print_string=True)
            for i in range(len(choices)):
                print('%d: %s' % (i, choices[i]))
            response = input('Response: ')

            # Test whether input is an integer or string.
            if fullmatch(r'\d+', response.strip()):
                to_int = int(response.strip())
                # Determine if input integer corresponds to one of the answer choices.
                if to_int < len(choices):
                    return to_int
                else:
                    error = 'ERROR: Invalid input! Input integer is not one of the available choices! Please try again.'
                continue
            else:
                # Determine if input string is one of the answer choices.
                for i in range(len(choices)):
                    if response.strip().lower() == choices[i].strip().lower():
                        return i
                error = 'ERROR: Invalid input! Input string is not one of the available choices! Please try again.'
                continue

    @staticmethod
    def string_input(question, condition=r'.+'):
        """
        Take string-based user input.

        The input question will be repeated until valid input is given, determined by the condition regex.

        Parameters
        ----------
        question : str
            String to be displayed as the input question. Will be boxed with Utils#box_string before printing.
        condition : r-string, optional
            Regex to test input string off of.

        Returns
        -------
        str
            Input string.
        """
        error = ''
        while True:
            # Print question and ask for input.
            Utils.box_string((error + '\n' + question).strip(), print_string=True)
            response = input()

            # Test if input is valid.
            if fullmatch(condition, response):
                return response
            else:
                error = 'ERROR: Invalid input! Please try again.'
                continue

    @staticmethod
    def print_settings(settings):
        """
        Pretty-print a settings dictionary.

        Parameters
        ----------
        settings : dict
            The settings dictionary to pretty-print.

        Returns
        -------
            None
        """
        Utils.box_string('Current Settings', print_string=True)

        print('Grid Size:')
        print('\tWidth: %d' % settings['width'])
        print('\tHeight: %d' % settings['height'])

        print('Ship Amount:')
        print('\t5-Long Ships: %d' % settings['5_ships'])
        print('\t4-Long Ships: %d' % settings['4_ships'])
        print('\t3-Long Ships: %d' % settings['3_ships'])
        print('\t2-Long Ships: %d' % settings['2_ships'])
        print('\t1-Long Ships: %d' % settings['1_ships'])

        print('Special Abilities:')
        print('\tShip Moving: %s' % str(settings['allow_moves']))
        print('\tMines: %s' % str(settings['allow_mines']))
        if settings['allow_mines']:
            print('\tTurns Between Mines: %d' % settings['mine_turns'])

        print('Game Type: Player vs. %s' % settings['p_type'])

    @staticmethod
    def grid_pos_input(height, width, question='Enter a Position:'):
        """
        Take user-input in coordinate form.

        The input question will be repeated until valid input is given.
        The input must be a valid coordinate in battleship form (r'[A-Z]\d+').
        The input coordinate must be inside of the grid defined by height and width.

        Parameters
        ----------
        height : int
            Specifies the height of the grid.
        width : int
            Specifies the width of the grid.
        question : str, optional
            String to be displayed as the input question. Will be boxed with Utils#box_string before printing. Defaults to 'Enter a Position'.

        Returns
        -------
        tuple
            Contains the following:
                int
                    Height-aligned position (y-position) of input.
                int
                    Width-aligned position (x-position) of input.
        """
        error = ''
        while True:
            # Print the question and ask for input.
            Utils.box_string((error + '\n' + question).strip(), print_string=True)
            loc = input().upper()

            # Test if input is a valid coordinate and is in the grid.
            if not fullmatch(r'[A-Z][1-2]?[0-9]', loc):
                error = 'ERROR: Invalid input! Input string is not a valid coordinate! Please try again.'
                continue
            elif loc[0] in letters[:height] and 0 < int(loc[1:]) <= width:
                return letters.index(loc[0]), int(loc[1:]) - 1
            else:
                error = 'ERROR: Invalid input! Input string is not in the grid! Please try again.'
                continue


class BattleshipGame(object):
    """
    Class that handles game execution and running.

    Controls game setup based off of a certain settings preset.
    Handles all input and output for the game.

    Attributes
    ----------
    settings : dict
        Settings that the game is running based off of.
    height : int
        Height of the grids used for the game.
    width : int
        Width of the grids used for the game.
    p1_grid : list
        Two dimensional list of ints containing player 1's board.
    p1_grid_2 : list
        Two dimensional list of ints containing player 1's guesses.
    p1_ships : list
        List of player 1's ship dicts with position, direction, and size data.
    p2_grid : list
        Two dimensional list of ints containing player 2's board.
    p2_grid_2 : list
        Two dimensional list of ints containing player 2's guesses.
    p2_ships : list
        List of player 2's ship dicts with position, direction, and size data.
    p2_cpu : bool
        True if player 2 is not a human player, False otherwise.
    turn : int
        Current turn number.
    p1_mines : int
        Current amount of mines available to Player 1.
    p2_mines : int
        Current amount of mines available to Player 2.
    p1_move : str
        Return message to display to Player 2 on their turn.
    p2_move : str
        Return message to display to Player 1 on their turn.
    """
    def __init__(self, settings):
        """
        Constructor for the BattleshipGame class.

        Parameters
        ----------
        settings : dict
            Settings to create the game based off of.
        """
        # Grid attributes.
        self.settings = settings
        self.height = settings['height']
        self.width = settings['width']

        # Player 1 grids.
        self.p1_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.p1_grid_2 = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.p1_ships = []

        # Player 2 grids.
        self.p2_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.p2_grid_2 = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.p2_ships = []

        # Miscellaneous attributes.
        self.p2_cpu = settings['p_type'] == 'CPU'
        self.turn = 0
        self.p1_mines = 0
        self.p2_mines = 0
        self.p1_move = ''
        self.p2_move = ''

        # CPU attributes.
        self.cpu_data = {'shots': [], 'misses': [], 'hits': [], 'p1_ships': None}

    def update_board(self, player):
        """
        Update both grids for a player.

        Adds new ships and puts them into the right locations.

        Parameters
        ----------
        player : int
            Determines which player's grids to print. Zero-indexed.
        """
        # Place ships into grid, if not already.
        if player == 0:  # Player 1
            board = self.p1_grid
            for ship in self.p1_ships:
                if not ship['setup']:
                    if ship['direction'] == 0:
                        for i in range(ship['size']):
                            if not (1 <= board[ship['y_pos']][ship['x_pos'] + i] <= 26 or board[ship['y_pos']][ship['x_pos'] + i] == 26):
                                board[ship['y_pos']][ship['x_pos'] + i] = ship['num'] + 1
                    else:
                        for j in range(ship['size']):
                            if not (1 <= board[ship['y_pos'] + j][ship['x_pos']] <= 26 or board[ship['y_pos'] + j][ship['x_pos']] == 26):
                                board[ship['y_pos'] + j][ship['x_pos']] = ship['num'] + 1
                    ship['setup'] = True
        else:  # Player 2
            board = self.p2_grid
            for ship in self.p2_ships:
                if not ship['setup']:
                    if ship['direction'] == 0:
                        for i in range(ship['size']):
                            if not (1 <= board[ship['y_pos']][ship['x_pos'] + i] <= 26 or board[ship['y_pos']][ship['x_pos'] + i] == 26):
                                board[ship['y_pos']][ship['x_pos'] + i] = ship['num'] + 1
                    else:
                        for j in range(ship['size']):
                            if not (1 <= board[ship['y_pos'] + j][ship['x_pos']] <= 26 or board[ship['y_pos'] + j][ship['x_pos']] == 26):
                                board[ship['y_pos'] + j][ship['x_pos']] = ship['num'] + 1
                    ship['setup'] = True

    def print_board(self, player):
        """
        Pretty-print the current boards of a player.

        Prints both boards for a player, along with coordinate references, titles, and boxes around the grids.

        Parameters
        ----------
        player : int
            Determines which player's grids to print. Zero-indexed.

        Returns
        -------
        str
            Same as the string that is printed.
        """
        # Characters to use while printing.
        characters = '.' + letters + '*0#'  # 0:Null, 1-26:Ships, 27:Hit, 28:Miss, 29:Mine

        # Update board.
        self.update_board(player)

        # Get boards to print.
        if player == 0:  # Player 1
            board = self.p1_grid
            board_2 = self.p1_grid_2
        else:  # Player 2
            board = self.p2_grid
            board_2 = self.p2_grid_2

        # Build header.
        result = '    +' + '-' * (self.width * 2 + 1) + '+' + '-' * (self.width * 2 + 1) + '+\n'
        result += '    |' + 'Your Board'.center(self.width * 2 + 1) + '|' + 'Enemy Board'.center(self.width * 2 + 1) + '|\n'
        result += '    +' + '-' * (self.width * 2 + 1) + '+' + '-' * (self.width * 2 + 1) + '+\n'

        # Build x-coordinate reference.
        if self.width > 9:
            result += '    | ' + ' '.join([str(x + 1).rjust(2)[0] for x in range(self.width)]) + ' | ' + ' '.join([str(x + 1).rjust(2)[0] for x in range(self.width)]) + ' |\n'
        result += '    | ' + ' '.join([str(x + 1).rjust(2)[1] for x in range(self.width)]) + ' | ' + ' '.join([str(x + 1).rjust(2)[1] for x in range(self.width)]) + ' |\n'
        result += '+---+' + '-' * (self.width * 2 + 1) + '+' + '-' * (self.width * 2 + 1) + '+\n'

        # Build y-coordinate reference and grid.
        for i in range(self.height):
            result += '| ' + letters[i] + ' | ' + ' '.join([characters[x] for x in board[i]]) + ' | ' + ' '.join([characters[x] for x in board_2[i]]) + ' |\n'
        result += '+---+' + '-' * (self.width * 2 + 1) + '+' + '-' * (self.width * 2 + 1) + '+'

        # Print and return result.
        print(result)
        return result

    def setup_ship(self, pos, direction, player, count, size):
        """
        Create a ship.

        Creates a ship dictionary based on positional, directional, player, and size data and tests if placement is legal.

        Parameters
        ----------
        pos : tuple
            (y,x) coordinate pair of top-left corner of the ship.
        direction : int
            Determines the direction of the ship:
                0: Horizontal.
                1: Vertical.
        player : int
            Determines which player to assign the ship to. Zero-indexed.
        count : int
            Current ship count for internal tracking use.
        size : int
            Length of the ship.

        Returns
        -------
        str
            Error string if an error occurred, None otherwise.
        """
        try:
            # Test if the ship does not overlap another ship.
            if player == 0:  # Player 1
                board = self.p1_grid
                if direction == 0:
                    for i in range(size):
                        if board[pos[0]][pos[1] + i] != 0:
                            return 'ERROR: You cannot place a ship on top of another!'
                else:
                    for j in range(size):
                        if board[pos[0] + j][pos[1]] != 0:
                            return 'ERROR: You cannot place a ship on top of another!'
            else:  # Player 2
                board = self.p2_grid
                if direction == 0:
                    for i in range(size):
                        if board[pos[0]][pos[1] + i] != 0:
                            return 'ERROR: You cannot place a ship on top of another!'
                else:
                    for j in range(size):
                        if board[pos[0] + j][pos[1]] != 0:
                            return 'ERROR: You cannot place a ship on top of another!'
        except IndexError:
            # Catch if ship would be placed out-of-bounds.
            return 'ERROR: You must place a ship inside the grid boundaries!'

        # Create the ship's dictionary and append it to the player's ship list.
        if player == 0:
            self.p1_ships.append({'num': count, 'size': size, 'x_pos': pos[1], 'y_pos': pos[0], 'direction': direction, 'setup': False, 'health': size, 'hits': []})
        else:
            self.p2_ships.append({'num': count, 'size': size, 'x_pos': pos[1], 'y_pos': pos[0], 'direction': direction, 'setup': False, 'health': size, 'hits': []})

        return None

    def setup_ships(self, size, player, count):
        """
        Setup all the ships of a particular size for a certain player.

        Sets up all of the length-n size ships for a player.
        Count is not updated in-place.

        Parameters
        ----------
        size : int
            Length of the ships.
        player : int
            Determines which player to assign the ships to. Zero-indexed.
        count : int
            Current ship count for internal tracking use.

        Returns
        -------
        int
            The updated cumulative ship count.
        """
        # Setup number of ships based on value defined in game settings.
        for i in range(self.settings['%d_ships' % size]):
            error = ''
            while True:
                # Print current board for player reference.
                self.print_board(player)

                # Take ship details from player.
                pos = Utils.grid_pos_input(self.height, self.width, question=(error + '\nWhere do you want to place ship \'%s\' (%d-long)?' % (letters[count], size)).strip())
                direction = Utils.num_input('Which direction?', 'Horizontal', 'Vertical')

                # Determine if the ship needs to be inputted again.
                error = self.setup_ship(pos, direction, player, count, size)
                if error is None:
                    break
            count += 1

        # Return updated cumulative ship total.
        return count

    def p1_turn(self):
        """
        Execute a turn for Player 1.

        Handles input and output for the turn and updates both player's grids.

        Returns
        -------
        bool
            True if game ends after the move, False otherwise
        """
        print('\n' * PAD_AMOUNT)  # Pad previous output.
        Utils.box_string('Player 1\'s Turn', min_width=self.width * 4 + 5, print_string=True)
        self.p1_move = ''

        # Test if Player 2 is a human.
        if not self.p2_cpu:
            # Alert Player 2 to look away.
            Utils.box_string('Player 2, please look away.', min_width=self.width * 4 + 5, print_string=True)
            sleep(self.settings['player_timer'])

        self.print_board(0)

        # Notify player if a ship moved.
        if self.p2_move != '':
            Utils.box_string('Note: ' + self.p2_move, min_width=self.width * 4 + 5, print_string=True)

        # Determine input method based on possible actions.
        if self.settings['allow_moves']:
            if self.settings['allow_mines'] and self.p1_mines > 0:
                action = Utils.num_input('What do you want to do?', 'Fire Missile', 'Move a Ship', 'Clear Misses', 'Clear Hits', 'Place a Mine')
            else:
                action = Utils.num_input('What do you want to do?', 'Fire Missile', 'Move a Ship', 'Clear Misses', 'Clear Hits')
            if action == 0:  # Fire Missile
                error = ''
                while True:
                    y_pos, x_pos = Utils.grid_pos_input(self.height, self.width, question=(error+'\nWhere do you want to fire?').strip())

                    if True in [(y_pos, x_pos) in self.p2_ships[x]['hits'] for x in range(len(self.p2_ships))] or self.p1_grid_2[y_pos][x_pos] > 26:
                        error = 'ERROR: You already guessed there!'
                        continue
                        
                    if self.p2_grid[y_pos][x_pos] > 26:
                        error = 'ERROR: You already guessed there!'
                        continue

                    if self.p2_grid[y_pos][x_pos] != 0:
                        Utils.box_string('Direct Hit!', min_width=self.width * 4 + 5, print_string=True)

                        # Update ship.
                        self.p2_ships[self.p2_grid[y_pos][x_pos] - 1]['health'] -= 1
                        self.p2_ships[self.p2_grid[y_pos][x_pos] - 1]['hits'].append((y_pos, x_pos))

                        # Test if ship still stands.
                        if self.p2_ships[self.p2_grid[y_pos][x_pos] - 1]['health'] == 0:
                            Utils.box_string('You sunk a ship!', min_width=self.width * 4 + 5, print_string=True)

                        # Update grid.
                        self.p1_grid_2[y_pos][x_pos] = 27
                        self.p2_grid[y_pos][x_pos] = 27
                    else:
                        Utils.box_string('Miss!', min_width=self.width * 4 + 5, print_string=True)

                        # Update grid.
                        self.p1_grid_2[y_pos][x_pos] = 28
                        self.p2_grid[y_pos][x_pos] = 28
                    break
            elif action == 1:  # Move Ship
                error = ''
                ship_num = -1
                while True:
                    ship_num = letters.index(Utils.string_input((error + '\nWhich ship do you want to move?').strip(), condition=('[A-%sa-%s]' % (letters[len(self.p1_ships) - 1], letters[len(self.p1_ships) - 1].lower()))).upper())
                    ship = self.p1_ships[ship_num]
                    if ship['health'] == 0:
                        error = 'ERROR: That ship is sunk!'
                        continue
                    move_direction = Utils.num_input('Which direction do you want to move it?', 'Up', 'Down', 'Left', 'Right')
                    error = ''
                    try:
                        if move_direction < 2:  # Up or down.
                            true_dir = -1 if move_direction == 0 else 1
                            board = self.p1_grid
                            if ship['direction'] == 0:
                                for i in range(ship['size']):
                                    # Check if another ship is there.
                                    for ship2 in self.p1_ships:
                                        if ship2['direction'] == 0:
                                            for k in range(ship2['size']):
                                                if ship2['num'] != ship_num and ship2['y_pos'] == ship['y_pos'] + true_dir and ship2['x_pos'] + k == ship['x_pos'] + i:
                                                    error = 'ERROR: You cannot move your ship there!'
                                                    continue
                                        else:
                                            for l in range(ship2['size']):
                                                if ship2['num'] != ship_num and ship2['y_pos'] + l == ship['y_pos'] + true_dir and ship2['x_pos'] == ship['x_pos'] + i:
                                                    error = 'ERROR: You cannot move your ship there!'
                                                    continue

                                    if (1 <= board[ship['y_pos'] + true_dir][ship['x_pos'] + i] <= 26 or board[ship['y_pos'] + true_dir][ship['x_pos'] + i] == 29) and (board[ship['y_pos'] + true_dir][ship['x_pos'] + i] != ship_num + 1) or ship['y_pos'] + true_dir < 0 or ship['y_pos'] >= self.height:
                                        error = 'ERROR: You cannot move your ship there!'
                            else:
                                for j in range(ship['size']):
                                    # Check if another ship is there.
                                    for ship2 in self.p1_ships:
                                        if ship2['direction'] == 0:
                                            for k in range(ship2['size']):
                                                if ship2['num'] != ship_num and ship2['y_pos'] == ship['y_pos'] + j + true_dir and ship2['x_pos'] + k == ship['x_pos']:
                                                    error = 'ERROR: You cannot move your ship there!'
                                                    continue
                                        else:
                                            for l in range(ship2['size']):
                                                if ship2['num'] != ship_num and ship2['y_pos'] + l == ship['y_pos'] + j +  true_dir and ship2['x_pos'] == ship['x_pos']:
                                                    error = 'ERROR: You cannot move your ship there!'
                                                    continue

                                    if (1 <= board[ship['y_pos'] + j + true_dir][ship['x_pos']] <= 26 or board[ship['y_pos'] + j + true_dir][ship['x_pos']] == 29) and (board[ship['y_pos'] + j + true_dir][ship['x_pos']] != ship_num + 1) or ship['y_pos'] + j + true_dir < 0 or ship['y_pos'] >= self.height:
                                        error = 'ERROR: You cannot move your ship there!'
                            if error == '':
                                self.p1_ships[ship_num]['setup'] = False
                                self.p1_ships[ship_num]['y_pos'] += true_dir
                                self.p1_move = 'Player 1 just moved a ship ' + ('up!' if move_direction == 0 else 'down!')

                                # Update board positions
                                if ship['direction'] == 0:
                                    for i in range(ship['size'] - 1):
                                        board[ship['y_pos'] + true_dir][ship['x_pos'] + i] = 0
                                else:
                                    for j in range(ship['size'] - 1):
                                        board[ship['y_pos'] + j + true_dir][ship['x_pos']] = 0
                                break
                        else:  # Left or right.
                            true_dir = -1 if move_direction == 2 else 1
                            board = self.p1_grid
                            if ship['direction'] == 0:
                                for i in range(ship['size']):
                                    # Check if another ship is there.
                                    for ship2 in self.p1_ships:
                                        if ship2['direction'] == 0:
                                            for k in range(ship2['size']):
                                                if ship2['num'] != ship_num and ship2['y_pos'] == ship['y_pos'] and ship2['x_pos'] + k == ship['x_pos'] + i + true_dir:
                                                    error = 'ERROR: You cannot move your ship there!'
                                                    continue
                                        else:
                                            for l in range(ship2['size']):
                                                if ship2['num'] != ship_num and ship2['y_pos'] + l == ship['y_pos'] and ship2['x_pos'] == ship['x_pos'] + i + true_dir:
                                                    error = 'ERROR: You cannot move your ship there!'
                                                    continue

                                    if (1 <= board[ship['y_pos']][ship['x_pos'] + i + true_dir] <= 26 or board[ship['y_pos']][ship['x_pos'] + i + true_dir] == 29) and (board[ship['y_pos']][ship['x_pos'] + i + true_dir] != ship_num + 1) or ship['x_pos'] + i + true_dir < 0 or ship['x_pos'] >= self.width:
                                        error = 'ERROR: You cannot move your ship there!'
                            else:
                                for j in range(ship['size']):
                                    # Check if another ship is there.
                                    for ship2 in self.p1_ships:
                                        if ship2['direction'] == 0:
                                            for k in range(ship2['size']):
                                                if ship2['num'] != ship_num and ship2['y_pos'] == ship['y_pos'] + j and ship2['x_pos'] + k == ship['x_pos'] + true_dir:
                                                    error = 'ERROR: You cannot move your ship there!'
                                                    continue
                                        else:
                                            for l in range(ship2['size']):
                                                if ship2['num'] != ship_num and ship2['y_pos'] + l == ship['y_pos'] + j and ship2['x_pos'] == ship['x_pos'] + true_dir:
                                                    error = 'ERROR: You cannot move your ship there!'
                                                    continue

                                    if (1 <= board[ship['y_pos'] + j][ship['x_pos'] + true_dir] <= 26 or board[ship['y_pos'] + j][ship['x_pos'] + true_dir] == 29) and (board[ship['y_pos'] + j][ship['x_pos'] + true_dir] != ship_num + 1) or ship['x_pos'] + true_dir < 0 or ship['x_pos'] >= self.width:
                                        error = 'ERROR: You cannot move your ship there!'
                            if error == '':
                                self.p1_ships[ship_num]['setup'] = False
                                self.p1_ships[ship_num]['x_pos'] += true_dir
                                self.p1_move = 'Player 1 just moved a ship to the ' + ('left!' if move_direction == 2 else 'right!')

                                # Update board positions.
                                if ship['direction'] == 0:
                                    for i in range(ship['size'] - 1):
                                        board[ship['y_pos']][ship['x_pos'] + i + true_dir] = 0
                                else:
                                    for j in range(ship['size'] - 1):
                                        board[ship['y_pos'] + j][ship['x_pos'] + true_dir] = 0
                                break
                    except IndexError:
                        error = 'ERROR: You cannot move your ship there!'

                # Update board positions again, just in case.
                for i in range(self.height):
                    for j in range(self.width):
                        if board[i][j] == ship_num + 1:
                            board[i][j] = 0

                self.p1_ships[ship_num]['hits'] = []

                self.update_board(0)
            elif action == 2:  # Clear Misses
                for i in range(self.height):
                    for j in range(self.width):
                        if self.p1_grid_2[i][j] == 28:
                            self.p1_grid_2[i][j] = 0
                return self.p1_turn()
            elif action == 3:  # Clear Hits
                for i in range(self.height):
                    for j in range(self.width):
                        if self.p1_grid_2[i][j] == 27:
                            self.p1_grid_2[i][j] = 0
                return self.p1_turn()
            else:  # Place Mine
                error = ''
                while True:
                    y_pos, x_pos = Utils.grid_pos_input(self.height, self.width, question=(error + '\nWhere do you want to place the mine?').strip())
                    if self.p2_grid[y_pos][x_pos] == 29:
                        error = 'ERROR: You already placed a mine there!'
                        continue

                    if 1 <= self.p2_grid[y_pos][x_pos] <= 26:
                        ship_num = self.p2_grid[y_pos][x_pos] - 1
                        self.p2_ships[ship_num]['health'] = 0
                        for i in range(self.height):
                            for j in range(self.width):
                                if self.p2_grid[i][j] == ship_num + 1:
                                    self.p2_grid[i][j] = 27
                        Utils.box_string('You sunk a ship!', min_width=self.width * 4 + 5, print_string=True)

                    self.p2_grid[y_pos][x_pos] = 29
                    self.p1_grid_2[y_pos][x_pos] = 29

                    self.p1_mines -= 1

                    break
        else:
            error = ''
            while True:
                y_pos, x_pos = Utils.grid_pos_input(self.height, self.width, question=(error + '\nWhere do you want to fire?').strip())

                if self.p1_grid_2[y_pos][x_pos] != 0:
                    error = 'ERROR: You already guessed there!'
                    continue
                    
                if self.p2_grid[y_pos][x_pos] > 26:
                    error = 'ERROR: You already guessed there!'
                    continue

                if self.p2_grid[y_pos][x_pos] != 0:
                    Utils.box_string('Direct Hit!', min_width=self.width * 4 + 5, print_string=True)

                    # Update ship.
                    self.p2_ships[self.p2_grid[y_pos][x_pos] - 1]['health'] -= 1
                    self.p2_ships[self.p2_grid[y_pos][x_pos] - 1]['hits'].append((y_pos, x_pos))

                    # Test if ship still stands.
                    if self.p2_ships[self.p2_grid[y_pos][x_pos] - 1]['health'] == 0:
                        Utils.box_string('You sunk a ship!', min_width=self.width * 4 + 5, print_string=True)

                    # Update grid.
                    self.p1_grid_2[y_pos][x_pos] = 27
                    self.p2_grid[y_pos][x_pos] = 27
                else:
                    Utils.box_string('Miss!', min_width=self.width * 4 + 5, print_string=True)

                    # Update grid.
                    self.p1_grid_2[y_pos][x_pos] = 28
                    self.p2_grid[y_pos][x_pos] = 28
                break

        # End turn.
        Utils.box_string('Your turn is now over.', print_string=True)
        sleep(self.settings['player_timer'])

        # Detect if game is over.
        return sum([x['health'] for x in self.p2_ships]) == 0

    def p2_turn(self):
        """
        Execute a turn for Player 2.

        Handles input and output for the turn and updates both player's grids.

        Returns
        -------
        bool
            True if game ends after the move, False otherwise
        """
        print('\n' * PAD_AMOUNT)  # Pad previous output.
        Utils.box_string('Player 2\'s Turn', min_width=self.width * 4 + 5, print_string=True)
        self.p2_move = ''

        # Test if Player 2 is a human.
        if not self.p2_cpu:  # Player is a human
            # Alert Player 1 to look away.
            Utils.box_string('Player 1, please look away.', min_width=self.width * 4 + 5, print_string=True)
            sleep(self.settings['player_timer'])

            self.print_board(1)
            if self.p1_move != '':
                Utils.box_string('Note: ' + self.p1_move, min_width=self.width * 4 + 5, print_string=True)

            # Determine input method based on possible actions.
            if self.settings['allow_moves']:
                if self.settings['allow_mines'] and self.p2_mines > 0:
                    action = Utils.num_input('What do you want to do?', 'Fire Missile', 'Move a Ship', 'Clear Misses', 'Clear Hits', 'Place a Mine')
                else:
                    action = Utils.num_input('What do you want to do?', 'Fire Missile', 'Move a Ship', 'Clear Misses', 'Clear Hits')
                if action == 0:  # Fire Missile
                    error = ''
                    while True:
                        y_pos, x_pos = Utils.grid_pos_input(self.height, self.width, question=(error+'\nWhere do you want to fire?').strip())

                        if True in [(y_pos, x_pos) in self.p1_ships[x]['hits'] for x in range(len(self.p1_ships))] or self.p2_grid_2[y_pos][x_pos] > 26:
                            error = 'ERROR: You already guessed there!'
                            continue

                        if self.p1_grid[y_pos][x_pos] > 26:
                            error = 'ERROR: You already guessed there!'
                            continue

                        if self.p1_grid[y_pos][x_pos] != 0:
                            Utils.box_string('Direct Hit!', min_width=self.width * 4 + 5, print_string=True)

                            # Update ship.
                            self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['health'] -= 1
                            self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['hits'].append((y_pos, x_pos))

                            # Test if ship still stands.
                            if self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['health'] == 0:
                                Utils.box_string('You sunk a ship!', min_width=self.width * 4 + 5, print_string=True)

                            # Update grid.
                            self.p2_grid_2[y_pos][x_pos] = 27
                            self.p1_grid[y_pos][x_pos] = 27
                        else:
                            Utils.box_string('Miss!', min_width=self.width * 4 + 5, print_string=True)

                            # Update grid.
                            self.p2_grid_2[y_pos][x_pos] = 28
                            self.p1_grid[y_pos][x_pos] = 28
                        break
                elif action == 1:  # Move Ship
                    error = ''
                    ship_num = -1
                    while True:
                        ship_num = letters.index(Utils.string_input((error + '\nWhich ship do you want to move?').strip(), condition=('[A-%sa-%s]' % (letters[len(self.p1_ships) - 1], letters[len(self.p1_ships) - 1].lower()))).upper())
                        ship = self.p2_ships[ship_num]
                        if ship['health'] == 0:
                            error = 'ERROR: That ship is sunk!'
                            continue
                        move_direction = Utils.num_input('Which direction do you want to move it?', 'Up', 'Down', 'Left', 'Right')
                        error = ''
                        try:
                            if move_direction < 2:  # Up or down.
                                true_dir = -1 if move_direction == 0 else 1
                                board = self.p2_grid
                                if ship['direction'] == 0:
                                    for i in range(ship['size']):
                                        # Check if another ship is there.
                                        for ship2 in self.p2_ships:
                                            if ship2['direction'] == 0:
                                                for k in range(ship2['size']):
                                                    if ship2['num'] != ship_num and ship2['y_pos'] == ship['y_pos'] + true_dir and ship2['x_pos'] + k == ship['x_pos'] + i:
                                                        error = 'ERROR: You cannot move your ship there!'
                                                        continue
                                            else:
                                                for l in range(ship2['size']):
                                                    if ship2['num'] != ship_num and ship2['y_pos'] + l == ship['y_pos'] + true_dir and ship2['x_pos'] == ship['x_pos'] + i:
                                                        error = 'ERROR: You cannot move your ship there!'
                                                        continue

                                        if (1 <= board[ship['y_pos'] + true_dir][ship['x_pos'] + i] <= 26 or board[ship['y_pos'] + true_dir][ship['x_pos'] + i] == 29) and (board[ship['y_pos'] + true_dir][ship['x_pos'] + i] != ship_num + 1) or ship['y_pos'] + true_dir < 0 or ship['y_pos'] >= self.height:
                                            error = 'ERROR: You cannot move your ship there!'
                                else:
                                    for j in range(ship['size']):
                                        # Check if another ship is there.
                                        for ship2 in self.p2_ships:
                                            if ship2['direction'] == 0:
                                                for k in range(ship2['size']):
                                                    if ship2['num'] != ship_num and ship2['y_pos'] == ship['y_pos'] + j + true_dir and ship2['x_pos'] + k == ship['x_pos']:
                                                        error = 'ERROR: You cannot move your ship there!'
                                                        continue
                                            else:
                                                for l in range(ship2['size']):
                                                    if ship2['num'] != ship_num and ship2['y_pos'] + l == ship['y_pos'] + j + true_dir and ship2['x_pos'] == ship['x_pos']:
                                                        error = 'ERROR: You cannot move your ship there!'
                                                        continue

                                        if (1 <= board[ship['y_pos'] + j + true_dir][ship['x_pos']] <= 26 or board[ship['y_pos'] + j + true_dir][ship['x_pos']] == 29) and (board[ship['y_pos'] + j + true_dir][ship['x_pos']] != ship_num + 1) or ship['y_pos'] + j + true_dir < 0 or ship['y_pos'] >= self.height:
                                            error = 'ERROR: You cannot move your ship there!'
                                if error == '':
                                    self.p2_ships[ship_num]['setup'] = False
                                    self.p2_ships[ship_num]['y_pos'] += true_dir
                                    self.p2_move = 'Player 2 just moved a ship ' + ('up!' if move_direction == 0 else 'down!')

                                    # Update board positions
                                    if ship['direction'] == 0:
                                        for i in range(ship['size'] - 1):
                                            board[ship['y_pos'] + true_dir][ship['x_pos'] + i] = 0
                                    else:
                                        for j in range(ship['size'] - 1):
                                            board[ship['y_pos'] + j + true_dir][ship['x_pos']] = 0
                                    break
                            else:  # Left or right.
                                true_dir = -1 if move_direction == 2 else 1
                                board = self.p2_grid
                                if ship['direction'] == 0:
                                    for i in range(ship['size']):
                                        # Check if another ship is there.
                                        for ship2 in self.p2_ships:
                                            if ship2['direction'] == 0:
                                                for k in range(ship2['size']):
                                                    if ship2['num'] != ship_num and ship2['y_pos'] == ship['y_pos'] and ship2['x_pos'] + k == ship['x_pos'] + i + true_dir:
                                                        error = 'ERROR: You cannot move your ship there!'
                                                        continue
                                            else:
                                                for l in range(ship2['size']):
                                                    if ship2['num'] != ship_num and ship2['y_pos'] + l == ship['y_pos'] and ship2['x_pos'] == ship['x_pos'] + i + true_dir:
                                                        error = 'ERROR: You cannot move your ship there!'
                                                        continue

                                        if (1 <= board[ship['y_pos']][ship['x_pos'] + i + true_dir] <= 26 or board[ship['y_pos']][ship['x_pos'] + i + true_dir] == 29) and (board[ship['y_pos']][ship['x_pos'] + i + true_dir] != ship_num + 1) or ship['x_pos'] + i + true_dir < 0 or ship['x_pos'] >= self.width:
                                            error = 'ERROR: You cannot move your ship there!'
                                else:
                                    for j in range(ship['size']):
                                        # Check if another ship is there.
                                        for ship2 in self.p2_ships:
                                            if ship2['direction'] == 0:
                                                for k in range(ship2['size']):
                                                    if ship2['num'] != ship_num and ship2['y_pos'] == ship['y_pos'] + j and ship2['x_pos'] + k == ship['x_pos'] + true_dir:
                                                        error = 'ERROR: You cannot move your ship there!'
                                                        continue
                                            else:
                                                for l in range(ship2['size']):
                                                    if ship2['num'] != ship_num and ship2['y_pos'] + l == ship['y_pos'] + j and ship2['x_pos'] == ship['x_pos'] + true_dir:
                                                        error = 'ERROR: You cannot move your ship there!'
                                                        continue

                                        if (1 <= board[ship['y_pos'] + j][ship['x_pos'] + true_dir] <= 26 or board[ship['y_pos'] + j][ship['x_pos'] + true_dir] == 29) and (board[ship['y_pos'] + j][ship['x_pos'] + true_dir] != ship_num + 1) or ship['x_pos'] + true_dir < 0 or ship['x_pos'] >= self.width:
                                            error = 'ERROR: You cannot move your ship there!'
                                if error == '':
                                    self.p2_ships[ship_num]['setup'] = False
                                    self.p2_ships[ship_num]['x_pos'] += true_dir
                                    self.p2_move = 'Player 2 just moved a ship to the ' + ('left!' if move_direction == 2 else 'right!')

                                    # Update board positions
                                    if ship['direction'] == 0:
                                        for i in range(ship['size'] - 1):
                                            board[ship['y_pos']][ship['x_pos'] + i + true_dir] = 0
                                    else:
                                        for j in range(ship['size'] - 1):
                                            board[ship['y_pos'] + j][ship['x_pos'] + true_dir] = 0
                                    break
                        except IndexError:
                            error = 'ERROR: You cannot move your ship there! (INDEX ERROR)'

                    # Update board positions again, just in case.
                    for i in range(self.height):
                        for j in range(self.width):
                            if board[i][j] == ship_num + 1:
                                board[i][j] = 0

                    self.p2_ships[ship_num]['hits'] = []

                    self.update_board(1)
                elif action == 2:  # Clear Misses
                    for i in range(self.height):
                        for j in range(self.width):
                            if self.p2_grid_2[i][j] == 28:
                                self.p2_grid_2[i][j] = 0
                    return self.p2_turn()
                elif action == 3:  # Clear Hits
                    for i in range(self.height):
                        for j in range(self.width):
                            if self.p2_grid_2[i][j] == 27:
                                self.p2_grid_2[i][j] = 0
                    return self.p2_turn()
                else:  # Place Mine
                    error = ''
                    while True:
                        y_pos, x_pos = Utils.grid_pos_input(self.height, self.width, question=(error + '\nWhere do you want to place the mine?').strip())
                        if self.p1_grid[y_pos][x_pos] == 29:
                            error = 'ERROR: You already placed a mine there!'
                            continue

                        if 1 <= self.p1_grid[y_pos][x_pos] <= 26:
                            ship_num = self.p1_grid[y_pos][x_pos] - 1
                            self.p1_ships[ship_num]['health'] = 0
                            for i in range(self.height):
                                for j in range(self.width):
                                    if self.p1_grid[i][j] == ship_num + 1:
                                        self.p1_grid[i][j] = 27
                            Utils.box_string('You sunk a ship!', min_width=self.width * 4 + 5, print_string=True)

                        self.p1_grid[y_pos][x_pos] = 29
                        self.p2_grid_2[y_pos][x_pos] = 29

                        self.p2_mines -= 1

                        break
            else:
                error = ''
                while True:
                    y_pos, x_pos = Utils.grid_pos_input(self.height, self.width, question=(error + '\nWhere do you want to fire?').strip())

                    if self.p2_grid_2[y_pos][x_pos] != 0:
                        error = 'ERROR: You already guessed there!'
                        continue

                    if self.p1_grid[y_pos][x_pos] > 26:
                        error = 'ERROR: You already guessed there!'
                        continue

                    if self.p1_grid[y_pos][x_pos] != 0:
                        Utils.box_string('Direct Hit!', min_width=self.width * 4 + 5, print_string=True)

                        # Update ship.
                        self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['health'] -= 1
                        self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['hits'].append((y_pos, x_pos))

                        # Test if ship still stands.
                        if self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['health'] == 0:
                            Utils.box_string('You sunk a ship!', min_width=self.width * 4 + 5, print_string=True)

                        # Update grid.
                        self.p2_grid_2[y_pos][x_pos] = 27
                        self.p1_grid[y_pos][x_pos] = 27
                    else:
                        Utils.box_string('Miss!', min_width=self.width * 4 + 5, print_string=True)

                        # Update grid.
                        self.p2_grid_2[y_pos][x_pos] = 28
                        self.p1_grid[y_pos][x_pos] = 28
                    break
        else:  # Player is CPU
            # Alert Player 1 of CPU turn.
            Utils.box_string('CPU is deciding...', min_width=self.width * 4 + 5, print_string=True)
            sleep(2)

            rng = Random()
            while True:
                pos = (rng.randrange(self.height), rng.randrange(self.width))
                y_pos, x_pos = pos
                if self.p1_grid[y_pos][x_pos] != 0:
                    # Update ship.
                    self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['health'] -= 1
                    self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['hits'].append((y_pos, x_pos))

                    # Test if ship still stands.
                    if self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['health'] == 0:
                        self.cpu_data['p1_ships']['%d_ships' % self.p1_ships[self.p1_grid[y_pos][x_pos] - 1]['size']] -= 1

                    # Update grid.
                    self.p2_grid_2[y_pos][x_pos] = 27
                    self.p1_grid[y_pos][x_pos] = 27
                else:
                    # Update grid.
                    self.p2_grid_2[y_pos][x_pos] = 28
                    self.p1_grid[y_pos][x_pos] = 28
                break

        # End turn.
        Utils.box_string('Your turn is now over.', print_string=True)
        sleep(self.settings['player_timer'])

        # Detect if game is over.
        return sum([x['health'] for x in self.p1_ships]) == 0

    def start_game(self):
        """
        Start a new game.

        Starts a game with the settings provided in the constructor.
        All game code is contained here, with relevant helper methods also called here.
        Every game has two stages: Setup and Play.

        Returns
        -------
        int
            Winning player's number. Zero-indexed.
        """
        # Setup Phase:
        # In this stage, both players choose where to place their ships.
        print('\n' * PAD_AMOUNT)  # Pad previous output.
        Utils.box_string('Setup Phase', min_width=self.width * 4 + 5, print_string=True)

        Utils.box_string('Player 1\'s Turn', min_width=self.width * 4 + 5, print_string=True)

        # Test if Player 2 is a human.
        if not self.p2_cpu:
            # Alert Player 2 to look away.
            Utils.box_string('Player 2, please look away.', min_width=self.width * 4 + 5, print_string=True)
            sleep(self.settings['player_timer'])

        # Player 1
        Utils.box_string('Player 1 Setup', min_width=self.width * 4 + 5, print_string=True)
        p1_ship_count = 0
        for i in range(5):
            p1_ship_count = self.setup_ships(i + 1, 0, p1_ship_count)

        # Test if Player 2 is a human.
        if self.p2_cpu:  # Player 2 is CPU
            # Setup CPU data.
            self.cpu_data['p1_ships'] = {}
            for size in range(1, 6):
                self.cpu_data['p1_ships']['%d_ships' % size] = self.settings['%d_ships' % size]

            # Setup ships.
            p2_ship_count = 0
            rng = Random()
            for size in range(1, 6):
                count = 0
                # Setup number of ships based on value defined in game settings.
                for i in range(self.settings['%d_ships' % size]):
                    while True:
                        # Generate ship details.
                        pos = (rng.randrange(self.height), rng.randrange(self. width))
                        direction = rng.randrange(2)

                        # Determine if the ship needs to be randomized again.
                        error = self.setup_ship(pos, direction, 1, p2_ship_count + count, size)
                        if error is None:
                            print('Placed ship ' + str(p2_ship_count + count) + ' at ' + str(pos) + ' with direction ' + str(direction) + ' with size ' + str(size))
                            break
                    count += 1

                # Update cumulative ship total.
                p2_ship_count += count

        else:  # Player 2 is a human
            print('\n' * PAD_AMOUNT)  # Pad previous output.
            Utils.box_string('Player 2\'s Turn', min_width=self.width * 4 + 5, print_string=True)

            # Alert Player 1 to look away.
            Utils.box_string('Player 1, please look away.', min_width=self.width * 4 + 5, print_string=True)
            sleep(self.settings['player_timer'])

            # Player 2
            Utils.box_string('Player 2 Setup', min_width=self.width * 4 + 5, print_string=True)
            p2_ship_count = 0
            for i in range(5):
                p2_ship_count = self.setup_ships(i + 1, 1, p2_ship_count)

        # Update both boards.
        self.update_board(0)
        self.update_board(1)

        # Play Phase:
        # In this stage, the game itself is played.
        Utils.box_string('Play Phase', min_width=self.width * 4 + 5, print_string=True)

        # Main game loop.
        winner = None
        while True:
            if self.settings['mine_turns'] is not None and self.turn % (self.settings['mine_turns'] * 2) == 0:
                self.p1_mines += 1
                self.p2_mines += 1

            if self.turn % 2 == 0:
                if self.p1_turn():
                    winner = 1
                    break
            else:
                if self.p2_turn():
                    winner = 2
                    break
            self.turn += 1

        # Print winner.
        Utils.box_string('Player %d won!' % winner, min_width=self.width * 4 + 5, print_string=True)

        return winner


def create_game(gm):
    """
    Configure and create a game.

    Creates a game with base settings equivalent to one of the default presets.
    Allows user to customize the settings before starting the game.

    Parameters
    ----------
    gm : int
        Game type to replicate:
            0: Normal mode.
            1: Advanced mode.

    Returns
    -------
    BattleshipGame
        Game instance with user-chosen settings.
    """
    print('\n' * PAD_AMOUNT)  # Pad previous output.

    # Choose and print default settings.
    if gm == 0:
        Utils.box_string('Normal Mode', print_string=True)
        settings = normal_mode_preset
    elif gm == 1:
        Utils.box_string('Advanced Mode', print_string=True)
        settings = advanced_mode_preset
    else:  # TODO: REMOVE TESTING MODE
        Utils.box_string('Testing Mode', print_string=True)
        settings = testing_preset

    # Print current settings.
    Utils.print_settings(settings)

    # Change settings, if applicable.
    if Utils.num_input('Would you like to change the settings?', 'No', 'Yes') == 1:
        while True:
            # Determine which setting group to modify.
            setting = Utils.num_input('Settings', 'Grid Size', 'Ship Amount', 'Special Abilities', 'Game Type', 'Exit')

            # Modify setting groups.

            if setting == 0:  # Grid Size
                # Take grid dimensions.
                settings['width'] = int(Utils.string_input('Grid Width (5-26)', condition=r'^[5-9]$|^1[0-9]$|^2[0-6]$'))
                settings['height'] = int(Utils.string_input('Grid Height (5-26)', condition=r'^[5-9]$|^1[0-9]$|^2[0-6]$'))

            elif setting == 1:  # Ship Amount
                while True:
                    # Take ship amounts.
                    settings['5_ships'] = int(Utils.string_input('5-Long Ships (0-9)', condition=r'[0-9]'))
                    settings['4_ships'] = int(Utils.string_input('4-Long Ships (0-9)', condition=r'[0-9]'))
                    settings['3_ships'] = int(Utils.string_input('3-Long Ships (0-9)', condition=r'[0-9]'))
                    settings['2_ships'] = int(Utils.string_input('2-Long Ships (0-9)', condition=r'[0-9]'))
                    settings['1_ships'] = int(Utils.string_input('1-Long Ships (0-9)', condition=r'[0-9]'))

                    # Test if ship amounts are valid.
                    count = settings['5_ships'] + settings['4_ships'] + settings['3_ships'] + settings['2_ships'] + settings['1_ships']
                    if count == 0:
                        Utils.box_string('You must have at least one ship!', print_string=True)
                    elif count > 26:
                        Utils.box_string('You have put in too many ships! (max 26)', print_string=True)
                    elif settings['5_ships'] * 5 + settings['4_ships'] * 4 + settings['3_ships'] * 3 + settings['2_ships'] * 2 + settings['1_ships'] > settings['width'] * settings['height']:
                        Utils.box_string('Your ships will not fit inside of the board!', print_string=True)
                    else:
                        break

            elif setting == 2:  # Special Abilities
                # Take abilities.
                settings['allow_moves'] = Utils.num_input('Ship Moving', 'Enable', 'Disable') == 0
                if settings['allow_moves']:
                    settings['allow_mines'] = Utils.num_input('Mines', 'Enable', 'Disable') == 0
                settings['mine_turns'] = int(Utils.string_input('Turns Between Mines', condition=r'\d+')) if settings['allow_mines'] else None

            elif setting == 3:  # Game Type
                # Take game type.
                settings['p_type'] = ['CPU', 'Player'][Utils.num_input('Game Type', 'CPU', 'Player')]

            # Print updated settings.
            Utils.print_settings(settings)

            if setting == 4:  # Exit
                break

    return BattleshipGame(settings)

# Test if code is run independently or on repl.it.
if __name__ == '__main__' or __name__ == 'builtins':
    print('\n' * PAD_AMOUNT)  # Pad previous output.
    Utils.box_string('Welcome to Battleship!', print_string=True)

    passed_settings = None
    while True:
        # Create game.
        gamemode = Utils.num_input('Which gamemode do you want to play?', 'Normal', 'Advanced', 'testing')  # TODO: REMOVE TESTING MODE
        if passed_settings is not None:
            bs = BattleshipGame(passed_settings)
        else:
            bs = create_game(gamemode)
            passed_settings = bs.settings

        # Play game.
        bs.start_game()

        # Determine if the game should be played again.
        again = Utils.num_input('Do you want to play again?', 'Yes [Same Settings]', 'Yes [Different Settings]', 'No')
        if again == 0:
            pass
        elif again == 1:
            passed_settings = None
        else:
            break
