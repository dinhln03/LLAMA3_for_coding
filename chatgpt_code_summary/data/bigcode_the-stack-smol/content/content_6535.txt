
def demo():
    """Output:
    ---------⌝
    ----------
    ----?????-
    ----------
    ----------
    --!!!-----
    --!!!-----
    ----------
    ----------
    ⌞---------
    """
    n = 10

    # Construction is easy:
    grid = {}

    # Assignment is easy:
    grid[(0, 0)] = "⌞"
    grid[(n - 1, n - 1)] = "⌝"

    # Helper functions that just work on the dictionary:
    fill(grid, "!", start=(2, 3), stop=(5, 5))
    fill(grid, "?", start=(4, 7), stop=(9, 8))

    print(stringify(grid, n))


def fill(grid: dict, value: str, start=(0, 0), stop=(0, 0)):
    """Using product allows for flatter loops."""
    from itertools import product

    for coord in product(range(start[0], stop[0]), range(start[1], stop[1])):
        grid[coord] = value


def stringify(grid: dict, n: int) -> str:
    """Stringify with (0, 0) in the lower-left corner."""
    rows = []
    for y in reversed(range(n)):
        row = []
        for x in range(n):
            value = grid.get((x, y), "-")
            row.append(value)

        rows.append(row)

    return "\n".join("".join(row) for row in rows)


if __name__ == "__main__":
    demo()
