"""Interfaces for interactively entering guesses."""
import curses
import time

import click

from wordle_cheater.interface_base import WordleCheaterUI


class CursesInterface(WordleCheaterUI):
    """Interface for using the curses library to enter guesses and display solutions.

    Attributes
    ----------
    guesses : list of WordleLetter objects
        The currently entered guesses.
    entering_letters : bool
        Whether or not we are currently entering guesses.
    """

    @classmethod
    def init_and_run(cls, *args, **kwargs):
        """Instantiate and run `self.main()` using `curses.wrapper`.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to the CursesInterface constructor.
        **kwargs : dict, optional
            Keyword arguments to be passed to the CursesInterface constructor.

        Returns
        -------
        CursesInterface object
            An instance of the CursesInterface class.
        """
        ui = cls(*args, **kwargs)
        curses.wrapper(ui.main)
        return ui

    def main(self, stdscr):
        """Run the interface.

        Should typically be called using `curses.wrapper`.

        Parameters
        ----------
        stdscr : curses.Window object
            The curses screen which the user interacts with.
        """
        self.stdscr = stdscr
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # White on black
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # Black on yellow
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Black on green
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_RED)  # Black on red

        height, width = stdscr.getmaxyx()
        self.results_window = curses.newwin(
            height - 12, width, 12, 0
        )  # window for printing results

        x0 = width // 2 - 3
        y0 = 5
        self.print_title()
        self.enter_letters(x0=x0, y0=y0)
        self.print_results()
        self.set_cursor_visibility(False)
        self.get_key()

    def center_print(self, y, string, *args, **kwargs):
        """Print in the center of the screen.

        Parameters
        ----------
        y : int
            The vertical location at which to print.
        string : str
            The string to print.
        *args : tuple
            Additional arguments to be passed to `stdscr.addstr`.
        **kwargs : dict, optional
            Keyword arguments to be passed to `stdscr.addstr`.
        """
        height, width = self.stdscr.getmaxyx()

        str_length = len(string)
        x_mid = width // 2

        self.stdscr.addstr(y, x_mid - str_length // 2, string, *args, **kwargs)

    def print_title(self):
        """Print title and instructions."""
        self.center_print(1, "Wordle Cheater :(", curses.A_BOLD)
        self.center_print(2, "Enter guesses below.")
        self.center_print(3, "spacebar: change color", curses.A_DIM)

    def print_results(self, sep="     "):
        """Print possible solutions given guesses.

        Parameters
        ----------
        sep : str, optional
            The string to display between each possible solution.
        """
        height, width = self.results_window.getmaxyx()
        max_rows = height - 1  # -1 to account for "Possible solutions" header
        cols = width // (5 + len(sep))

        out_str = self.get_results_string(max_rows=max_rows, max_cols=cols, sep=sep)

        self.results_window.clear()
        self.results_window.addstr(0, 0, "Possible solutions:", curses.A_UNDERLINE)
        self.results_window.addstr(1, 0, out_str)
        self.results_window.refresh()

    def print(self, x, y, string, c=None):
        """Print `string` at coordinates `x`, `y`.

        Parameters
        ----------
        x : int
            Horizontal position at which to print the string.
        y : int
            Height at which to print the string.
        string : str
            The string to print.
        c : str, {None, 'black', 'yellow', 'green', 'red'}
            The color in which to print.  Must be one of
            ['black', 'yellow', 'green', 'red'] or None. If `c` is None, it should
            print in the default color pair.
        """
        if c is None:
            self.stdscr.addstr(y, x, string)

        elif c == "black":
            self.stdscr.addstr(y, x, string, curses.color_pair(1))

        elif c == "yellow":
            self.stdscr.addstr(y, x, string, curses.color_pair(2))

        elif c == "green":
            self.stdscr.addstr(y, x, string, curses.color_pair(3))

        elif c == "red":
            self.stdscr.addstr(y, x, string, curses.color_pair(4))

        else:
            raise ValueError(
                "`c` must be one of ['black', 'yellow', 'green', 'red'] or none."
            )

    def sleep(self, ms):
        """Temporarily suspend execution.

        Parameters
        ----------
        ms : int
            Number of miliseconds before execution resumes.
        """
        curses.napms(ms)
        self.stdscr.refresh()

    def move_cursor(self, x, y):
        """Move cursor to position `x`, `y`.

        Parameters
        ----------
        x : int
            Desired horizontal position of cursor.
        y : int
            Desired vertical position of cursor.
        """
        self.stdscr.move(y, x)

    def set_cursor_visibility(self, visible):
        """Set cursor visibility.

        Parameters
        ----------
        visible : bool
            Whether or not the cursor is visible.
        """
        curses.curs_set(visible)

    def get_key(self):
        """Get a key press.

        Returns
        -------
        key : str
            The key that was pressed.
        """
        return self.stdscr.getkey()

    def is_enter(self, key):
        """Check if `key` is the enter/return key.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        is_enter : bool
            True if `key` is the enter or return key, False otherwise.
        """
        if key == curses.KEY_ENTER or key == "\n" or key == "\r":
            return True

        else:
            return False

    def is_backspace(self, key):
        """Check if `key` is the backspace/delete key.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        is_backspace : bool
            True if `key` is the backspace or delete key, False otherwise.
        """
        if key == curses.KEY_BACKSPACE or key == "\b" or key == "\x7f":
            return True

        else:
            return False


class ClickInterface(WordleCheaterUI):
    """Interface for using Click alone to enter letters and see solutions.

    Parameters
    ----------
    max_rows : int, optional
        The maximum rows of possible solutions to print.
    max_cols : int, optional
        The maximum columns of possible solutions to print.
    x0 : int, optional
        The leftmost position where guesses will be entered.
    y0 : int, optional
        The topmost position where guesses will be entered.
    esc : str, optional
        The ANSI escape code for the terminal.

    Attributes
    ----------
    guesses : list of WordleLetter
        The currently entered guesses.
    entering_letters : bool
        Whether or not we are currently entering guesses.
    max_rows : int, optional
        The maximum rows of possible solutions to print.
    max_cols : int, optional
        The maximum columns of possible solutions to print.
    x0 : int, optional
        The leftmost position where guesses will be entered.
    y0 : int, optional
        The topmost position where guesses will be entered.
    esc : str, optional
        The ANSI escape code for the terminal.
    line_lengths : list of int
        The highest x value we've printed to per line.  For example, if we've printed
        two lines, the first one up to x=5 and the second up to x=3, then
        `line_lengths = [5, 3]`.
    curs_xy
    """

    def __init__(self, max_rows=10, max_cols=8, x0=4, y0=4, esc="\033"):
        self.max_rows = max_rows  # Maximum rows of results to print
        self.max_cols = max_cols  # Maximum columns of results to print
        self.x0 = x0  # Initial x position of guesses
        self.y0 = y0  # Initial y position of guesses
        self.esc = esc  # ANSI escape code
        self._curs_xy = (0, 0)  # cursor position
        self.line_lengths = [0]  # Highest x values we've hit per line
        super().__init__()

    @property
    def curs_xy(self):
        """Location of cursor."""
        return self._curs_xy

    @curs_xy.setter
    def curs_xy(self, xy):
        """Update max line lengths when we update cursor position."""
        x, y = xy
        if y > len(self.line_lengths) - 1:
            self.line_lengths += [0 for i in range(y - len(self.line_lengths) + 1)]

        if x > self.line_lengths[y]:
            self.line_lengths[y] = x

        self._curs_xy = xy

    def main(self):
        """Run the interface."""
        try:
            self.print_title()
            self.enter_letters(x0=self.x0, y0=self.y0)
            self.print_results()

        finally:
            self.set_cursor_visibility(True)

    def print_title(self):
        """Print title and instructions."""
        self.print(0, 0, "Wordle Cheater :(", bold=True)
        self.print(0, 1, "Enter guesses below.")
        self.print(0, 2, "spacebar: change color", dim=True)

    def print_results(self):
        """Print possible solutions given guesses."""
        # If we're still entering letters, don't do anything
        if self.entering_letters:
            return

        out_str = self.get_results_string(
            max_rows=self.max_rows, max_cols=self.max_cols, sep="     "
        )

        self.move_cursor(0, self.curs_xy[1] + 1)
        click.secho("Possible solutions:", underline=True)
        click.echo(out_str)

    def print(self, x, y, string, c=None, *args, **kwargs):
        """Print `string` at coordinates `x`, `y`.

        Parameters
        ----------
        x : int
            Horizontal position at which to print the string.
        y : int
            Height at which to print the string.
        string : str
            The string to print.
        c : str, {None, 'black', 'yellow', 'green', 'red'}
            The color in which to print.  Must be one of
            ['black', 'yellow', 'green', 'red'] or None. If `c` is None, it should
            print in the default color pair.
        *args : tuple
            Additional arguments to be passed to `click.secho`.
        **kwargs : dict, optional
            Keyword arguments to be passed to `click.secho`.
        """
        # Move cursor to x, y so we can print there
        self.move_cursor(x, y)

        if c is None:
            click.secho(string, nl=False, *args, **kwargs)

        elif c == "black":
            click.secho(string, fg="white", bg="black", nl=False)

        elif c == "yellow":
            click.secho(string, fg="black", bg="yellow", nl=False)

        elif c == "green":
            click.secho(string, fg="black", bg="green", nl=False)

        elif c == "red":
            click.secho(string, fg="black", bg="red", nl=False)

        else:
            raise ValueError(
                "`c` must be one of ['black', 'yellow', 'green', 'red'] or none."
            )

        self.curs_xy = (self.curs_xy[0] + len(string), self.curs_xy[1])

    def sleep(self, ms):
        """Temporarily suspend execution.

        Parameters
        ----------
        ms : int
            Number of miliseconds before execution resumes.
        """
        time.sleep(ms / 1000)

    def move_cursor(self, x, y):
        """Move cursor to position `x`, `y`.

        Parameters
        ----------
        x : int
            Desired horizontal position of cursor.
        y : int
            Desired vertical position of cursor.
        """
        # Check if we want to move cursor up (decreasing y)
        if self.curs_xy[1] > y:
            click.echo(f"{self.esc}[{self.curs_xy[1] - y}A", nl=False)

        # Check if we want to move cursor down (increasing y)
        elif self.curs_xy[1] < y:
            # Check if we need to add new lines to screen
            if len(self.line_lengths) - 1 < y:
                # First arrow down as far as possible
                click.echo(
                    f"{self.esc}[{(len(self.line_lengths) - 1) - self.curs_xy[1]}B",
                    nl=False,
                )

                # Now add blank lines
                click.echo("\n" * (y - (len(self.line_lengths) - 1)), nl=False)

                # New line, so definitely need to print spaces to move x
                click.echo(" " * x, nl=False)
                self.curs_xy = (x, y)
                return

            else:
                # Should just arrow down to not overwrite stuff
                click.echo(f"{self.esc}[{y - self.curs_xy[1]}B", nl=False)

        # Check if we want to move cursor left (decreasing x)
        if self.curs_xy[0] > x:
            click.echo(f"{self.esc}[{self.curs_xy[0] - x}D", nl=False)

        # Check if we want to move cursor right (increasing x)
        elif self.curs_xy[0] < x:
            # Check if we need to add space to right of cursor
            if self.line_lengths[y] > x:
                # First arrow to the right as far as possible
                click.echo(
                    f"{self.esc}[{self.line_lengths[y] - self.curs_xy[0]}C", nl=False
                )

                # Now add blank spaces
                click.echo(" " * (x - self.line_lengths[y]), nl=False)

            else:
                # Should just arrow to right to not overwrite stuff
                click.echo(f"{self.esc}[{x - self.curs_xy[0]}C", nl=False)

        self.curs_xy = (x, y)

    def set_cursor_visibility(self, visible):
        """Set cursor visibility.

        Parameters
        ----------
        visible : bool
            Whether or not the cursor is visible.
        """
        if visible:
            click.echo(f"{self.esc}[?25h", nl=False)

        else:
            click.echo(f"{self.esc}[?25l", nl=False)

    def get_key(self):
        """Get a key press.

        Returns
        -------
        key : str
            The key that was pressed.
        """
        return click.getchar()

    def is_enter(self, key):
        """Check if `key` is the enter/return key.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        is_enter : bool
            True if `key` is the enter or return key, False otherwise.
        """
        if key == "\r" or key == "\n":
            return True

        else:
            return False

    def is_backspace(self, key):
        """Check if `key` is the backspace/delete key.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        is_backspace : bool
            True if `key` is the backspace or delete key, False otherwise.
        """
        if key == "\b" or key == "\x7f":
            return True

        else:
            return False


if __name__ == "__main__":
    # curses_ui = CursesInterface()
    # curses.wrapper(curses_ui.main)
    click_ui = ClickInterface()
    click_ui.main()
